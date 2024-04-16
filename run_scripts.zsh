#!/bin/zsh

# Set the variable for the directory
dir="ATS900"

# Execute the first Python script
python scripts/pdf_to_images.py --input_folder_path data/$dir

# Execute the second Python script
python scripts/async_calls_openai.py --requests_filepath data/$dir --save_filepath data/$dir/embeds/summaries_$dir.jsonl

# Execute the third Python script
python scripts/compute_pdf_embeds.py --dir_path data/$dir
