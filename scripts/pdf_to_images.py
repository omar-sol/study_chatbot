import fitz  # PyMuPDF
import os


def convert_pdf_to_images(pdf_path, output_folder, pdf_name, dpi=72):
    # Open the PDF file
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        # Get the page
        page = doc.load_page(page_num)

        # Render page to an image (pixmap) with specified dpi
        pix = page.get_pixmap(dpi=dpi)  # type: ignore

        # Define the output filename
        image_filename = f"{output_folder}/{pdf_name}/page_{page_num + 1}.png"

        # Save the image
        pix.save(image_filename)

        print(f"Saved {image_filename}")

    # Close the document
    doc.close()


INPUT_FOLDER_PATH = "data/GES800"  # Folder containing PDF files

for file in filter(lambda f: f.endswith(".pdf"), os.listdir(INPUT_FOLDER_PATH)):
    pdf_path = os.path.join(INPUT_FOLDER_PATH, file)
    pdf_name = os.path.splitext(file)[0].replace(" ", "_")
    pdf_output_folder = os.path.join(INPUT_FOLDER_PATH, pdf_name)

    if not os.path.exists(pdf_output_folder):
        os.makedirs(pdf_output_folder)
        convert_pdf_to_images(pdf_path, INPUT_FOLDER_PATH, pdf_name, dpi=72)
