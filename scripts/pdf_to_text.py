import PyPDF2
import argparse
import os

def convert_pdf_to_text(path_to_pdf, output_folder) :   
    # Open the PDF file
    with open(path_to_pdf, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        texts = []
        # Iterate over each page and extract text
        for page in reader.pages:
            # print(page.extract_text())
            texts.append(page.extract_text())
        
        for i,text in enumerate(texts):
            # Save the texts in the output_folder
            output_filename = f"{output_folder}/output_p{i}.txt" 
            with open(output_filename, 'w', encoding='utf-8') as file:
                file.write(text)

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_pdf",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--path_to_text",
        type=str,
        required=True,
    )
        
    args = parser.parse_args()
    path_to_pdf = args.path_to_pdf
    path_to_text = args.path_to_text
    
    if not os.path.exists(path_to_text):
        os.makedirs(path_to_text)

    convert_pdf_to_text(path_to_pdf, path_to_text)

if __name__ == "__main__":
    main()
