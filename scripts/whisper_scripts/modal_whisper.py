import json
import logging
import time

from modal import Image, Mount, Secret, Stub, Volume
import whisper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

stub = Stub("audio-transcriber")
volume = Volume.from_name("transcribe_volume", create_if_missing=True)

app_image = (
    Image.debian_slim(python_version="3.11", force_build=False)
    .apt_install("git")
    .pip_install(
        "git+https://github.com/openai/whisper.git",
        "ffmpeg-python",
        "loguru==0.6.0",
        "torchaudio==2.1.0",
    )
    .apt_install("ffmpeg")
    .pip_install("ffmpeg-python")
)


def split_silences(
    path: str, min_segment_length: float = 30.0, min_silence_length: float = 1.0
):
    import re
    import ffmpeg

    silence_end_re = re.compile(
        r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))"
    )

    try:
        metadata = ffmpeg.probe(path)
        duration = float(metadata["format"]["duration"])

        reader = (
            ffmpeg.input(str(path))
            .filter("silencedetect", n="-10dB", d=min_silence_length)
            .output("pipe:", format="null")
            .run_async(pipe_stderr=True)
        )

        cur_start = 0.0
        num_segments = 0

        while True:
            line = reader.stderr.readline().decode("utf-8")
            if not line:
                break
            match = silence_end_re.search(line)
            if match:
                silence_end, silence_dur = match.group("end"), match.group("dur")
                split_at = float(silence_end) - (float(silence_dur) / 2)

                if (split_at - cur_start) < min_segment_length:
                    continue

                yield cur_start, split_at
                cur_start = split_at
                num_segments += 1

        if duration > cur_start:
            yield cur_start, duration
            num_segments += 1
        print(f"Split {path} into {num_segments} segments")

    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr}")
        raise

    except Exception as e:
        print(f"Error while splitting silences: {str(e)}")
        raise


@stub.function(
    image=app_image,
    volumes={"/results": volume},
    cpu=2,
    timeout=3000,
)
def transcribe_segment(
    start: float,
    end: float,
    audio_filepath: str,
    model: str = "large",
):
    import tempfile
    import time

    import ffmpeg
    import torch
    import whisper

    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        try:
            (
                ffmpeg.input(str(audio_filepath))
                .filter("atrim", start=start, end=end)
                .output(f.name)
                .overwrite_output()
                .run(quiet=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr}")
            raise

        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"

        model = whisper.load_model(
            model, device=device, download_root="/results/weights/"
        )
        result = model.transcribe(
            f.name,
            language="fr",
            fp16=use_gpu,
            condition_on_previous_text=False,
        )

    print(
        f"Transcribed segment {start:.2f} to {end:.2f} ({end - start:.2f}s duration) in {time.time() - t0:.2f} seconds."
    )

    for segment in result["segments"]:
        segment["start"] += start
        segment["end"] += start

    return result


# @stub.function(
#     image=app_image,
#     timeout=1200,
#     volumes={"/results": volume},
# )
# def transcribe_audio(audio_filepath: str, result_path: str, model: str = "base"):
#     volume.reload()
#     segment_gen = split_silences(str(audio_filepath))
#     output_text = ""
#     output_segments = []
#     for result in transcribe_segment.starmap(
#         segment_gen, kwargs=dict(audio_filepath=audio_filepath, model=model)
#     ):
#         output_text += result["text"]
#         output_segments += result["segments"]

#     result = {
#         "text": output_text,
#         "segments": output_segments,
#         "language": "fr",
#     }

#     print(f"Writing transcription to {result_path}")
#     with open(result_path, "w") as f:
#         json.dump(result, f, indent=4)

#     volume.commit()


@stub.function(
    image=app_image,
    timeout=3000,
    volumes={"/results": volume},
)
async def transcribe_audio(audio_filepath: str, result_path: str, model: str = "base"):
    volume.reload()
    segment_gen = split_silences(str(audio_filepath))

    print(f"Writing transcription to {result_path}")
    with open(result_path, "w") as f:
        f.write('{"text": "", "segments": [], "language": "fr"}\n')

    def process_segments():
        for result in transcribe_segment.starmap(
            segment_gen, kwargs=dict(audio_filepath=audio_filepath, model=model)
        ):
            yield result

    with open(result_path, "a") as f:
        for result in process_segments():
            text = result["text"]
            segments = result["segments"]
            f.write(f'{{"text": "{text}", "segments": {json.dumps(segments)}}}\n')
            f.flush()

    volume.commit()


@stub.local_entrypoint()
def main():
    start = time.time()
    audio_filepath = "/results/audios/cours_3-5.mp3"
    result_path = "/results/cours_3-5_transcription.json"
    model = "large"
    transcribe_audio.remote(audio_filepath, result_path, model)
    completion_time = time.time() - start
    logger.info(f"Transcription completed in: {completion_time}")


# modal volume ls transcribe_volume /
# modal volume put transcribe_volume data/GES824/audios /
# modal volume get transcribe_volume cours_2-1_transcription.json

# With whisper v3 - large, and CPU
# transcribed audio file of 1:23:37 in 7m20s
# modal cost $0.96 + $0.53 + $1.02 + $0.24 = 2.75 US$

# transcribed 59m08s in 5m18s
# cost $0.74 CPU + $0.94 Memory = $1.68

# transcribed 3:38:07 in 21m14s
# cost ~ 2.85 + 2.19 = 5.05 a bit more than that.

# transcribed 25m16 in 7m23s min=1.0sec
