import json
import logging
import os

import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import cohere
from dotenv import load_dotenv

load_dotenv()
api_key: str | None = os.getenv("COHERE_API_KEY")

PDF_EMBEDS_DATASET = np.load("/embeds/embeddings_GES824.npy")
PDF_CONTENTS_FILE_PATH = "/embeds/summaries_GES824.jsonl"

# ONLY FOR AUDIO
AUDIO_CONTENTS_FILE_PATH = "/embeds/summaries_audio_GES824.jsonl"
AUDIO_EMBEDS_DATASET = np.load("/embeds/embeddings_audio_GES824.npy")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Study Chatbot", description="A chatbot to help students study")

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
            data.append(json.loads(line))
    return data


contents = load_jsonl(PDF_CONTENTS_FILE_PATH)
audio_contents = load_jsonl(AUDIO_CONTENTS_FILE_PATH)


def cosine_similarity_matrix(vectors, query_vec):
    dot_product = np.dot(vectors, query_vec)
    norms_vectors = np.linalg.norm(vectors, axis=1)
    norm_query_vec = np.linalg.norm(query_vec)
    if norm_query_vec == 0 or np.any(norms_vectors == 0):
        logger.error(
            "Cosine similarity is not defined when one or both vectors are zero vectors."
        )
        raise ValueError(
            "Cosine similarity is not defined when one or both vectors are zero vectors."
        )
    similarity = dot_product / (norms_vectors * norm_query_vec)
    return similarity


async def retrieve_chunks(user_input: str = ""):
    logger.info(f"User input: {user_input}")

    if not user_input.endswith("\n"):
        user_input += "\n"

    model_name = "embed-multilingual-v3.0"
    input_type_embed = "search_query"

    if api_key is None:
        logger.error("Please set the COHERE_API_KEY environment variable.")
        raise ValueError("Please set the COHERE_API_KEY environment variable.")

    async with cohere.AsyncClient(api_key) as co:
        query_embed = await co.embed(
            texts=[user_input], model=model_name, input_type=input_type_embed
        )

    query_array = np.array(query_embed.embeddings)
    query_array = query_array.reshape(-1)

    # ONLY FOR PDF
    similarity_results_pdf = np.zeros((PDF_EMBEDS_DATASET.shape[0],), dtype=np.float32)
    try:
        similarity_results_pdf = cosine_similarity_matrix(
            PDF_EMBEDS_DATASET, query_array
        )
    except ValueError as e:
        logger.error(e)
        print(e)

    sorted_indices_pdf = np.argsort(similarity_results_pdf)[::-1]
    relevant_contents_pdf: list = [contents[i] for i in sorted_indices_pdf[:5]]

    for d in relevant_contents_pdf:
        d.pop("image_path", None)

    # return json.dumps(relevant_contents_pdf, ensure_ascii=False)

    # ONLY FOR AUDIO
    similarity_results_audio = np.zeros(
        (AUDIO_EMBEDS_DATASET.shape[0],), dtype=np.float32
    )
    try:
        similarity_results_audio = cosine_similarity_matrix(
            AUDIO_EMBEDS_DATASET, query_array
        )
    except ValueError as e:
        logger.error(e)
        print(e)

    sorted_indices_audio = np.argsort(similarity_results_audio)[::-1]
    relevant_contents_audio: list = [
        audio_contents[i] for i in sorted_indices_audio[:5]
    ]

    # Combine PDF and Audio results into one JSON object
    combined_results = {
        "notes_pdf": relevant_contents_pdf,
        "audio_transcripts": relevant_contents_audio,
    }

    combined_json = json.dumps(combined_results, ensure_ascii=False)

    return combined_json


class AnswerRequest(BaseModel):
    user_input: str = Field(
        ...,
        min_length=1,
        description="The user's input question.",
    )


@app.post("/get_chunks/")
async def retrieve_chunks_endpoint(request: AnswerRequest) -> Response:
    chunks = await retrieve_chunks(request.user_input)
    return Response(chunks, media_type="application/json")
