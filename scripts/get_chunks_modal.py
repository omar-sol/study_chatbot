from modal import asgi_app, Image, Stub, Secret, Mount

stub = Stub("chunks-GES800")

image = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install(
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
    secrets=[Secret.from_name("my-custom-secret")],
    mounts=[
        Mount.from_local_dir("data/GES800/embeds", remote_path="/embeds"),
        Mount.from_local_python_packages("get_chunks_api"),
    ],
    keep_warm=0,
)
@asgi_app()
def fastapi_app():
    from get_chunks_api import app

    return app
