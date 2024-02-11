import os
from pathlib import Path
import logging
import json
import uuid

import cohere
import numpy as np
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Security, Depends
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from src.myfirebase import get_current_user
from src.pdf_to_imgs import convert_pdf_to_images
from src.extraction_gpt4_vision import img_to_text
from src.text_to_embed import file_text_to_embed
from src.retrieve_chunks import retrieve_chunks

load_dotenv()
api_key: str | None = os.getenv("COHERE_API_KEY")

app = FastAPI()
es = Elasticsearch(["http://localhost:9200"])

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/hello-world/")
async def hellow_world(current_user: Optional[dict] = Depends(get_current_user)):
    print("current user auth", not not current_user)
    if current_user:
        # If authenticated, use the user's name from the decoded token
        user_name = current_user.get("email", "World")  # Adjust key as needed based on the token's content
        return {"message": f"Hello {user_name}"}
    else:
        # If not authenticated, return a default message
        return {"message": "Hello World"}

# Directory where uploaded files will be stored
UPLOAD_DIR = Path("data/")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/test-upload/")
async def test_upload(file: UploadFile = File(...)):
    contents = await file.read()
    print(type(contents))
    print(f"Uploaded file size: {len(contents)} bytes")
    return {"filename": file.filename, "size": len(contents)}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), current_user: Optional[dict] = Depends(get_current_user)):
    if current_user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    email = current_user.get("email", uuid.uuid4())
    try:
        
        await process_file(file, email)
        return JSONResponse(status_code=200, content={"message": f"File '{file.filename}' uploaded successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {e}"})

def create_dirs(email: str, filename: str) -> str:
    name, _ = os.path.splitext(filename)

    # Create a directory to store the file
    file_path = "data/" + email + "/" + name + "/file/"  
    file_dir = Path(file_path)
    file_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a directory to store the images
    img_path = "data/" + email +  "/" + name + "/images/"  
    images_dir = Path(img_path)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create a directory to store the text
    output_path = "data/" + email +  "/" + name + "/output/" 
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a directory to store the embedings
    # TODO Store embedings in a database
    embedings_path = "data/" + email + "/" + name + "/embedings/"
    embedings_dir = Path(embedings_path)
    embedings_dir.mkdir(parents=True, exist_ok=True)

    return (name, file_path, img_path, output_path, embedings_path)

def save_file(contents : bytes, file_location: str) -> None:
    # Save the file
    with open(file_location, "wb+") as file_object:
        file_object.write(contents)

async def process_file(file : UploadFile, email: str) : 

    # Create a directory to store the images
    name, file_path, img_path, output_path, embedings_path = create_dirs(email, file.filename)

    # Read content of the file 
    if file.filename.endswith('.pdf'):
        contents = await file.read()
    save_file(contents, file_path + file.filename)

    # Convert the PDF to images
    convert_pdf_to_images(contents, img_path)

    # extraction gpt4
    img_to_text(img_path, output_path)
    
    # Embedding
    file_text_to_embed(output_path, embedings_path)


class AnswerRequest(BaseModel):
    user_input: str = Field(
        min_length=1,
        description="The user's input question.",
        examples=["What is the Mistral 7B model?"],
    )
    sigle_cours: list[str] = Field(
        description="The course's sigle.",
        examples=["INF1005A"],
    )

@app.post("/get_chunks/")
def retrieve_chunks_endpoint(request: AnswerRequest) -> Response:
    return Response(
        retrieve_chunks(request.user_input, request.sigle_cours),
        media_type="application/json",
    )


@app.post("/search")
async def search(query_vector: list, tags: list):
    response = es.search(index="my_vectors", body={
        # Your search query here
    })
    return {"hits": response["hits"]["hits"]}





