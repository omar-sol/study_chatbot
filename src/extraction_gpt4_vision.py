import argparse
import os
import json
import base64
import fitz  # PyMuPDF
import io
import base64

from tqdm import tqdm
import instructor
from openai import OpenAI, AsyncOpenAI
import tiktoken
from dotenv import load_dotenv

load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def img_to_text(images_path : str, output_file_path : str, model="gpt-4-vision-preview", max_retries=1):
    print("img_to_text")

    image_files = [
        file
        for file in os.listdir(images_path)
        if os.path.isfile(os.path.join(images_path, file))
    ]
    sorted_image_files = sorted(
        image_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    for file in sorted_image_files:
        full_path = os.path.join(images_path, file)
        base64_image = encode_image(full_path)
        extracted_details = extract_tags_function_call(
            base64_image, model, max_retries
        )
        print(extracted_details.choices[0].message.content)
        dict = {
            "module": images_path,
            "slide_number": file,
            "content": extracted_details.choices[0].message.content,
        }
        save_to_jsonl(dict, output_file_path + "output.jsonl") 

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_tags_function_call(base64_image: str, model: str, max_retries: int = 0):
    client = instructor.patch(OpenAI())
    try:
        extracted_details = client.chat.completions.create(
            model=model,
            # response_model=pm.ImageModel,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world class extractor of information from images and figures created for students. You only extract information, so avoid chat like answers.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What are the informations in this image? finish with a short resume.",
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
            max_tokens=1000,
        )
        return extracted_details

    except Exception as e:
        print(e)
        return f"Error generating post with the OpenAI API: {e}"
    

def save_to_jsonl(data, output_file_path):
    with open(output_file_path, "a") as file:  # 'a' mode for appending to the file
        json.dump(data, file)
        file.write("\n")



