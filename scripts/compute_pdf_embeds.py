import argparse
import json
import os
import logging

import cohere
import numpy as np
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            # Parse each line as a JSON object
            data.append(json.loads(line))
    return data


def main(dir_path):
    dir_name = os.path.basename(dir_path)
    file_path = os.path.join(dir_path, f"embeds/summaries_{dir_name}.jsonl")
    contents = load_jsonl(file_path)
    text_contents = [slide["content"] for slide in contents]

    model_name = "embed-multilingual-v3.0"
    api_key: str | None = os.getenv("COHERE_API_KEY")
    input_type_embed = "search_document"

    if api_key is None:
        raise ValueError("Please set the COHERE_API_KEY environment variable.")

    co = cohere.Client(api_key)

    # Split the text_contents list into chunks of 96 texts
    chunk_size = 96
    text_chunks = [
        text_contents[i : i + chunk_size]
        for i in range(0, len(text_contents), chunk_size)
    ]

    # Initialize an empty list to store the embeddings
    embeds = []

    # Iterate over each chunk and make the API request
    for chunk in text_chunks:
        chunk_embeds: list[list[float]] = co.embed(
            texts=chunk, model=model_name, input_type=input_type_embed
        ).embeddings  # type: ignore
        embeds.extend(chunk_embeds)

    print(type(embeds), len(embeds), len(embeds[0]))
    array_embeds = np.array(embeds)
    print(array_embeds.shape)

    save_path = os.path.join(dir_path, f"embeds/embeddings_{dir_name}.npy")
    logger.info(f"Saving embeddings to {save_path}")
    np.save(save_path, array_embeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for PDF files")
    parser.add_argument(
        "--dir_path", help="The relative path to the directory containing the PDF files"
    )
    args = parser.parse_args()

    main(args.dir_path)
