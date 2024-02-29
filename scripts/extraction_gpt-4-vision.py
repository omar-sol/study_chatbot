import argparse
import os
import time
import json
import logging
import base64

from tqdm import tqdm
import instructor
from openai import OpenAI, AsyncOpenAI
import tiktoken
from dotenv import load_dotenv


load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def save_to_jsonl(data, output_file_path):
    with open(output_file_path, "a") as file:  # 'a' mode for appending to the file
        json.dump(data, file)
        file.write("\n")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_tags_function_call(base64_image: str, model: str, max_retries: int = 0):
    # client = instructor.patch(OpenAI(), mode=instructor.function_calls.Mode.JSON_SCHEMA)
    client = instructor.patch(OpenAI())
    try:
        extracted_details = client.chat.completions.create(
            model=model,
            # response_model=ExtractionModel,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world class extractor of information from images and figures created for students. You only extract the information, so avoid chat like answers. Write in french.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Quelle est l'information présenté dans la diapositive? Écrit un résumé de l'information.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            max_retries=max_retries,
            max_tokens=2000,
        )
        return extracted_details

    except Exception as e:
        print(e)
        # print(f"{job_details._raw_response}\n")
        return f"Error generating post with the OpenAI API: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_folder",
        type=str,
        required=True,
        help="Path to folder containing images",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output JSONL file"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help="Number of retries for the OpenAI API",
    )

    args = parser.parse_args()
    folder_path = args.path_to_folder
    output_path = args.output_path

    image_files = [
        file
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file))
    ]
    sorted_image_files = sorted(
        image_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    with tqdm(sorted_image_files, desc="Processing Images") as t:
        for file in t:
            full_path = os.path.join(folder_path, file)
            t.set_description(f"Processing {full_path}")

            base64_image = encode_image(full_path)
            model = "gpt-4-vision-preview"
            extracted_details = extract_tags_function_call(
                base64_image, model, args.max_retries
            )
            print(extracted_details.choices[0].message.content)
            dict = {
                "module": folder_path,
                "slide_number": file,
                "content": extracted_details.choices[0].message.content,
            }
            save_to_jsonl(dict, output_path)


if __name__ == "__main__":
    main()
