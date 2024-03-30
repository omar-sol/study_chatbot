import modal
from modal import Image
import whisper

stub = modal.Stub("model-upload")
volume = modal.Volume.from_name("transcribe_volume", create_if_missing=True)


image = (
    Image.debian_slim(python_version="3.11")
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


@stub.function(
    volumes={"/model": volume},
    image=image,
)
def f():
    print("hello")
    volume.reload()
    device = "cpu"
    model = whisper.load_model("large", device=device, download_root="/model/weights/")
    volume.commit()  # Persist changes


@stub.function(
    volumes={"/model": volume},
    image=image,
)
def g():
    volume.reload()  # Fetch latest changes
    device = "cpu"
    model = whisper.load_model("large", device=device, download_root="/model/weights/")
    print("Done")


@stub.local_entrypoint()
def main():
    f.remote()
    g.remote()


# modal volume ls transcribe_volume /
# modal volume put transcribe_volume data/GES824/audios /
# modal volume get transcribe_volume cours_2-1_transcription.json
