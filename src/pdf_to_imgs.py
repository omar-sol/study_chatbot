import fitz
import os
import argparse

def convert_pdf_to_images(contents, output_folder, dpi=300):

    # Open the PDF file from the bytes object
    doc = fitz.open(stream=contents, filetype="pdf")
    
    pixs = []
    for page_num in range(len(doc)):
        # Get the page
        page = doc.load_page(page_num)

        # Set zoom factor based on desired DPI
        zoom = dpi / 72  # Default DPI in PDF is 72
        matrix = fitz.Matrix(zoom, zoom)

        # Render page to an image (pixmap) with the zoom matrix
        pix = page.get_pixmap(matrix=matrix)

        # Append the pixmap to the list
        pixs.append(pix)

        # Define the output filename
        image_filename = f"{output_folder}/page_{page_num + 1}.png"

        # Save the image
        pix.save(image_filename)

        print(f"Saved {image_filename}")

    # Close the document
    doc.close()

    # return pixs

