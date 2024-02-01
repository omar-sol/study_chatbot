import os
import logging
import json

import cohere
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, Header, HTTPException, Security
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
api_key: str | None = os.getenv("COHERE_API_KEY")
embeds_dataset = np.load("embeddings.npy")

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            # Parse each line as a JSON object
            data.append(json.loads(line))
    return data


file_path = "output.jsonl"
contents = load_jsonl(file_path)


def cosine_similarity_matrix(vectors, query_vec):
    dot_product = np.dot(vectors, query_vec)

    norms_vectors = np.linalg.norm(vectors, axis=1)
    norm_query_vec = np.linalg.norm(query_vec)

    if norm_query_vec == 0 or np.any(norms_vectors == 0):
        raise ValueError(
            "Cosine similarity is not defined when one or both vectors are zero vectors."
        )

    similarity = dot_product / (norms_vectors * norm_query_vec)
    return similarity


def retrieve_chunks(user_input: str = ""):
    logger.info(f"User input: {user_input}")

    if not user_input.endswith("\n"):
        user_input += "\n"

    model_name = "embed-multilingual-v3.0"
    input_type_embed = "search_query"

    if api_key is None:
        raise ValueError("Please set the COHERE_API_KEY environment variable.")
    co = cohere.Client(api_key)

    # Get the embeddings
    query_embed: list[list[float]] = co.embed(
        texts=[user_input], model=model_name, input_type=input_type_embed
    ).embeddings
    query_array = np.array(query_embed)
    query_array = query_array.reshape(-1)

    similarity_results = np.zeros((embeds_dataset.shape[0],), dtype=np.float32)
    try:
        similarity_results = cosine_similarity_matrix(embeds_dataset, query_array)
    except ValueError as e:
        print(e)

    sorted_indices = np.argsort(similarity_results)[::-1]
    relevant_contents = [contents[i] for i in sorted_indices[:5]]

    return json.dumps(relevant_contents, ensure_ascii=False)


class AnswerRequest(BaseModel):
    user_input: str = Field(
        min_length=1,
        description="The user's input question.",
        examples=["What is the Mistral 7B model?"],
    )


@app.post("/get_chunks/")
def retrieve_chunks_endpoint(request: AnswerRequest) -> Response:
    return Response(
        retrieve_chunks(request.user_input),
        media_type="application/json",
    )
