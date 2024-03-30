from modal import asgi_app, Image, Stub, Secret, Mount

stub = Stub("chunks-GES824")

image = (
    Image.debian_slim(force_build=True)
    .apt_install("git")
    .pip_install(
        "-U",
        "pydantic>=2.6",
        "fastapi",
        "uvicorn",
        "numpy",
        "cohere",
        "python-dotenv",
    )
)


@stub.function(
    image=image,
    secrets=[Secret.from_name("my-personal-secrets")],
    mounts=[
        Mount.from_local_dir("data/GES824/embeds", remote_path="/embeds"),
        Mount.from_local_python_packages("get_chunks_api"),
    ],
    keep_warm=0,
)
@asgi_app()
def fastapi_app():
    from get_chunks_api import app

    return app
