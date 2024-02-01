# School-Chatbot

This projects creates an API for the a school chatbot.

## Installation

Create a new Python environment

```bash
python -m venv env
```

Activate the environment

```bash
source env/bin/activate
```

Install the dependencies

```bash
pip install -r requirements.txt
```

## Usage

```bash
export OPENAI_API_KEY=...
export COHERE_API_KEY=...
```

```bash
uvicorn main:app --reload
```

## Create a Docker image

Build the image

```bash
docker build --platform linux/amd64 -t school_chunks .
```

(Optional) Test the image (needs environment variables in a .env file)

```bash
export OPENAI_API_KEY=...
export COHERE_API_KEY=...
```

```bash
docker run -it --name school_chunks-container -p 80:80 --env-file .env school_chunks
```

Tag the image

```bash
docker tag school_chunks gcr.io/PROJECT_ID/school_chunks:latest
```

Push to Google Container Registry

```bash
docker push gcr.io/PROJECT_ID/school_chunks:latest
```
