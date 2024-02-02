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

## Create cohere account, get api key 

## Create openAi Account, get api key

## Create env file 

```env
OPENAI_API_KEY=...
COHERE_API_KEY=...
```

## Get your own pdf into the data folder 

## Execute script pdf_to_images.py
```bash
python pdf_to_images.py --path_to_pdf "data/pdf_exemple.pdf" --path_to_img "data/images" 
```

## Execute extraction_gpt-4-vision.py
```bash
python extraction_gpt-4-vision.py --path_to_folder "data/images"
```

## Execute compute_embeddings.ipynb

# Local Test

## Start the api 
```bash
uvicorn src.main:app --reload
```
## Call the api  
```bash
curl -N -X POST "http://127.0.0.1:8000/get_chunks/" -H "Content-Type: application/json" -d '{"user_input": "YOUR QUESTION HERE"}'
```

# Deploy 

## Connect to google cloud 
```ps1
gcloud auth login
```
## Create config for pushing the container 
```ps1
gcloud auth configure-docker
```
 
## Crete a google cloud project 
### Google Container Registry API

## insert the name of the project in the build_push file 

## Run the script build_push
```ps1
./build_push.ps1
```

```bash
./build_push.sh
```

- Creer google cloud run service (choisir l'image pusher, mettre les clés env, autorizer call public, augmenter le nombre de core et 4 Gb ram)

- OpenaiGPT  action : import url (fast api /docs), ajouter le serveur, ajouter privacy policy






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
