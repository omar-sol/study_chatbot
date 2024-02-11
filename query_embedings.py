import cohere
import os
import numpy as np
from openai import OpenAI

QUERY="Le devoir 1 vaut pour combien de pourcentage?"

api_key: str | None = os.getenv("COHERE_API_KEY")

# Now we'll set up the cohere client.
if api_key is None:
    raise ValueError("Please set the COHERE_API_KEY environment variable.")
co = cohere.Client(api_key)

def prompt_query(file_path: str, user_input: str, model :str="gpt-3.5-turbo-1106", max_retries : int = 1):
    get_embeddings(file_path)
    extract_tags_function_call(user_input, model, max_retries)

def extract_tags_function_call(user_input: str, model :str="", max_retries : int = 1):
    client = OpenAI()
    try:
        extracted_details = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world class teacher in the field of management and supervision of teams working on projects. You will answer questions asked by students. You will use theory provided to you in the form of slides.",
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
            max_tokens=1000,
        )
        return extracted_details

    except Exception as e:
        print(e)
        return f"Error generating post with the OpenAI API: {e}"

def get_embeddings(file_path: str, model_name: str = "gpt-3.5-turbo-1106", input_type_embed: str = "search_document"):    
    # Get the embeddings
    query_embed: list[list[float]] = co.embed(
        texts=[QUERY], model=model_name, input_type=input_type_embed
    ).embeddings
    query_array = np.array(query_embed)
    query_array = query_array.reshape(-1)

    embeds_dataset = np.load(file_path) # 'embeddings.npy'

    similarity_results = np.zeros((embeds_dataset.shape[0],), dtype=np.float32)
    try:
        similarity_results = cosine_similarity_matrix(embeds_dataset, query_array)
    except ValueError as e:
        print(e)

    sorted_indices = np.argsort(similarity_results)[::-1]
    relevant_contents = [contents[i] for i in sorted_indices[:5]]
    for i in range(8):
        print(similarity_results[sorted_indices[i]], contents[sorted_indices[i]])


def cosine_similarity_matrix(vectors, query_vec):
    dot_product = np.dot(vectors, query_vec)
    
    norms_vectors = np.linalg.norm(vectors, axis=1)
    norm_query_vec = np.linalg.norm(query_vec)
    
    if norm_query_vec == 0 or np.any(norms_vectors == 0):
        raise ValueError("Cosine similarity is not defined when one or both vectors are zero vectors.")
    
    similarity = dot_product / (norms_vectors * norm_query_vec)
    return similarity