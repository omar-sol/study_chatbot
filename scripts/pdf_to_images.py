import fitz  # PyMuPDF
import os


def convert_pdf_to_images(pdf_path, output_folder, pdf_name):
    # Open the PDF file
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        # Get the page
        page = doc.load_page(page_num)

        # Render page to an image (pixmap)
        pix = page.get_pixmap()

        # Define the output filename
        image_filename = f"{output_folder}/{pdf_name}/page_{page_num + 1}.png"

        # Save the image
        pix.save(image_filename)

        print(f"Saved {image_filename}")

    # Close the document
    doc.close()


# Main script
input_folder_path = "data/GES811"  # Folder containing PDF files
output_folder_path = "data/GES811"  # Base folder to store output images

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for file in os.listdir(input_folder_path):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(input_folder_path, file)
        pdf_name = os.path.splitext(file)[0].replace(
            " ", "_"
        )  # Replace spaces with underscores
        pdf_output_folder = os.path.join(output_folder_path, pdf_name)

        if not os.path.exists(pdf_output_folder):
            os.makedirs(pdf_output_folder)

        convert_pdf_to_images(pdf_path, output_folder_path, pdf_name)
