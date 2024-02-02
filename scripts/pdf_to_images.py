import fitz  # PyMuPDF
import os
import argparse

def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    # Open the PDF file
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        # Get the page
        page = doc.load_page(page_num)

        # Set zoom factor based on desired DPI
        zoom = dpi / 72  # Default DPI in PDF is 72
        matrix = fitz.Matrix(zoom, zoom)

        # Render page to an image (pixmap) with the zoom matrix
        pix = page.get_pixmap(matrix=matrix)

        # Define the output filename
        image_filename = f"{output_folder}/page_{page_num + 1}.png"

        # Save the image
        pix.save(image_filename)

        print(f"Saved {image_filename}")

    # Close the document
    doc.close()

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_pdf",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--path_to_img",
        type=str,
        required=True,
    )
        
    args = parser.parse_args()
    path_to_pdf = args.path_to_pdf
    path_to_images = args.path_to_img
    
    if not os.path.exists(path_to_images):
        os.makedirs(path_to_images)

    convert_pdf_to_images(path_to_pdf, path_to_images)

if __name__ == "__main__":
    main()
