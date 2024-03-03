import json
import logging
import os
import numpy as np
import cohere
from dotenv import load_dotenv

load_dotenv()
api_key: str | None = os.getenv("COHERE_API_KEY")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def retrieve_chunks(user_input: str, sigle_cours: list[str]):
    embeds_dataset = np.load("data/frank/embeddings.npy")  # 90,1028
    file_path = "data/frank/texts/output.jsonl"
    contents = load_jsonl(file_path)
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


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            # Parse each line as a JSON object
            data.append(json.loads(line))
    return data


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
