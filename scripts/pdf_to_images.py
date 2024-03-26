import fitz  # PyMuPDF
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF pages into a folder with images.",
    )
    parser.add_argument(
        "--input_folder_path",
        type=str,
        help="The folder containing PDF files to convert.",
    )
    args = parser.parse_args()

    input_folder_path = args.input_folder_path

    for file in filter(lambda f: f.endswith(".pdf"), os.listdir(input_folder_path)):
        pdf_path = os.path.join(input_folder_path, file)
        pdf_name = os.path.splitext(file)[0].replace(" ", "_")
        pdf_output_folder = os.path.join(input_folder_path, pdf_name)

        if not os.path.exists(pdf_output_folder):
            os.makedirs(pdf_output_folder)
            convert_pdf_to_images(pdf_path, input_folder_path, pdf_name, dpi=72)
            logger.info(f"Converted {pdf_path} to {pdf_output_folder}")
        else:
            logger.info(f"Skipping {pdf_path} because it already exists.")

    logger.info("Finished processing all PDF files.")


if __name__ == "__main__":
    main()
