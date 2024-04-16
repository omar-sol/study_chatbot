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
        "torchaudio==2.1.0",
    )
)


@stub.function(
    volumes={"/model": volume},
    image=image,
)
def f():
    print("hello")
    volume.reload()
    device = "cpu"
    model = whisper.load_model(
        "large-v3", device=device, download_root="/model/weights/"
    )
    volume.commit()


@stub.local_entrypoint()
def main():
    f.remote()
