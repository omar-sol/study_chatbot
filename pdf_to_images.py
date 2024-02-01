import fitz  # PyMuPDF
import os


def convert_pdf_to_images(pdf_path, output_folder):
    # Open the PDF file
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        # Get the page
        page = doc.load_page(page_num)

        # Render page to an image (pixmap)
        pix = page.get_pixmap()

        # Define the output filename
        image_filename = f"{output_folder}/page_{page_num + 1}.png"

        # Save the image
        pix.save(image_filename)

        print(f"Saved {image_filename}")

    # Close the document
    doc.close()


# Usage
PDF_PATH = "data/"  # Replace with your PDF file path
OUTPUT_FOLDER_PATH = "data/"  # Replace with your desired output folder path

if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

convert_pdf_to_images(PDF_PATH, OUTPUT_FOLDER_PATH)
