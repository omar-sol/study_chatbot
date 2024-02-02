import json
import os
import cohere
import numpy as np
from dotenv import load_dotenv

load_dotenv()

model_name = "embed-multilingual-v3.0"
api_key: str | None = os.getenv("COHERE_API_KEY")
input_type_embed = "search_document"

# Now we'll set up the cohere client.
if api_key is None:
    raise ValueError("Please set the COHERE_API_KEY environment variable.")
co = cohere.Client(api_key)

def file_text_to_embed(file_path: str, output_path: str):
    contents = load_jsonl(file_path + "output.jsonl")
    text_contents = [slide['content'] for slide in contents]
    # Get the embeddings
    embeds: list[list[float]] = co.embed(
        texts=text_contents, model=model_name, input_type=input_type_embed
    ).embeddings

    array_embeds = np.array(embeds)
    np.save(output_path + "embeddings.npy", array_embeds)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            data.append(json.loads(line))
    return data

