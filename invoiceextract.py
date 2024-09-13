import os
import yaml
import email
import json
import pdfplumber
from imapclient import IMAPClient
import time
from email.header import decode_header
import logging
import os
import streamlit as st
import json
import pdfplumber
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient

import math
import fitz  # PyMuPDF
import cv2
import numpy as np
from deskew import determine_skew
from PIL import Image
import json
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient, BlobBlock
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from typing import Tuple
from openai import AzureOpenAI, OpenAIError
import pandas as pd
from tempfile import NamedTemporaryFile

api_key = st.secrets["api_key"]
azure_endpoint = st.secrets["azure_endpoint"]
api_version = st.secrets["api_version"]
deployment_name = st.secrets["deployment_name"]

# Load environment variables


# Initialize the AzureOpenAI client
client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version
)
 
# Deskewing
def rotate(image: np.ndarray, angle: float, background: tuple) -> np.ndarray:
    old_height, old_width = image.shape[:2]
    angle_radian = math.radians(angle)
    new_width = int(abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width))
    new_height = int(abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height))
    image_center = (old_width / 2, old_height / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (new_width - old_width) / 2
    rot_mat[0, 2] += (new_height - old_height) / 2
    rotated_image = cv2.warpAffine(image, rot_mat, (new_width, new_height), borderValue=background)
    return rotated_image
 
def deskew_image(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    return angle
 
def deskew_pdf(input_pdf_path, output_pdf_path):
    document = fitz.open(input_pdf_path)
    deskewed_images = []
 
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap(dpi=100)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
 
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]
        if width > height:
            img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
 
        angle = deskew_image(img_cv)
 
        if angle is not None:
            deskewed_img_cv = rotate(img_cv, angle, (255, 255, 255))
        else:
            print(f"Could not determine skew for page {page_num}. Skipping deskewing.")
            deskewed_img_cv = img_cv
 
        deskewed_img = Image.fromarray(cv2.cvtColor(deskewed_img_cv, cv2.COLOR_BGR2RGB))
        deskewed_images.append(deskewed_img)
 
    deskewed_images[0].save(output_pdf_path, save_all=True, append_images=deskewed_images[1:], quality=100, dpi=(300, 300))
    print(f"Deskewed PDF saved to {output_pdf_path}")
 
def split_pdf(input_pdf_path):
    document = fitz.open(input_pdf_path)
    pdf_chunks = []
   
    for i in range(len(document)):
        output_pdf = fitz.open()  # Create a new PDF in memory
        output_pdf.insert_pdf(document, from_page=i, to_page=i)
       
        output_pdf_path = f'{input_pdf_path}_page_{i + 1}.pdf'
        output_pdf.save(output_pdf_path)
        pdf_chunks.append(output_pdf_path)
        output_pdf.close()
   
    document.close()
    return pdf_chunks
 
def extract_text_and_coordinates_from_pdf(pdf_path, client, raw_output_path):
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-read", document=f)
        result = poller.result()
 
    # Save raw coordinates
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        raw_content = [page for page in result.pages]
        json.dump(raw_content, f, default=lambda o: o.__dict__, indent=2)
 
    # Extract text
    lines = []
    for page in result.pages:
        for line in page.lines:
            lines.append(line.content)
    return "\n".join(lines)
 
def calculate_tolerance(elements):
    heights = []
    for e in elements:
        polygon = e['polygon']
        if len(polygon) >= 2:
           
            y_coords = [point[1] for point in polygon]  # Assuming polygon is a list of [x, y]
            height = max(y_coords) - min(y_coords)
            heights.append(height)
    avg_height = sum(heights) / len(heights) if heights else 0
    return avg_height * 0.5
 
def group_by_position(elements):
    if not elements:
        return []
 
    vertical_tolerance = calculate_tolerance(elements)
    grouped_lines = []
    current_group = []
    last_top = None
 
    sorted_elements = sorted(elements, key=lambda e: min(point[1] for point in e['polygon']))
 
    for element in sorted_elements:
        top = min(point[1] for point in element['polygon'])
        left = min(point[0] for point in element['polygon'])
 
        if last_top is None or abs(top - last_top) <= vertical_tolerance:
            current_group.append(element)
        else:
            current_group.sort(key=lambda e: min(point[0] for point in e['polygon']))
            line_text = ' '.join(e['content'] for e in current_group)
            grouped_lines.append(line_text)
            current_group = [element]
 
        last_top = top
 
    if current_group:
        current_group.sort(key=lambda e: min(point[0] for point in e['polygon']))
        line_text = ' '.join(e['content'] for e in current_group)
        grouped_lines.append(line_text)
 
    return grouped_lines
 
def save_grouped_lines_to_file(pages, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for page_number, elements in sorted(pages.items()):
            grouped_lines = group_by_position(elements)
            for line in grouped_lines:
                file.write(line + '\n')
           
            file.write(f"\nPAGE_SEPARATOR (Page {page_number})\n\n")

def organize_elements_by_page(raw_file_contents):
    pages = {}
    for page in raw_file_contents:
        page_number = page.get('page_number')
        if page_number not in pages:
            pages[page_number] = []
        for line in page.get('lines', []):
            pages[page_number].append({
                'content': line.get('content'),
                'polygon': line.get('polygon')
            })
    return pages
 
CHUNK_SIZE = 4 * 1024 * 1024  
 
def upload_file_in_chunks(blob_service_client, container_name, blob_name, file_path, chunk_size):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    block_list = []
    file_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as file:
        chunk_num = 0
        while True:
            chunk_data = file.read(chunk_size)
            if not chunk_data:
                break
            chunk_id = f'{chunk_num:06d}'  # Generate a unique block ID for each chunk
            blob_client.stage_block(block_id=chunk_id, data=chunk_data)
            block_list.append(BlobBlock(block_id=chunk_id))
            chunk_num += 1
 
    # Commit the blocks as a single blob
    blob_client.commit_block_list(block_list)
    print(f"Uploaded {file_path} in chunks to {blob_name} in Azure Blob Storage.")

def upload_to_blob(response_json, container_name, blob_service_client):
    try:
        # Ensure the output folder exists
        local_output_folder = "output"
        if not os.path.exists(local_output_folder):
            os.makedirs(local_output_folder)

        # Create a temporary file to store the JSON data
        with NamedTemporaryFile(delete=False, mode='w', suffix='.json', dir=local_output_folder) as temp_file:
            temp_file.write(response_json)
            temp_file_path = temp_file.name

        print(f"Saved JSON data to temporary file: {temp_file_path}")
        
        # Upload the temp file to Azure Blob Storage
        blob_name = f'output/{os.path.basename(temp_file_path)}'
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        file_size = os.path.getsize(temp_file_path)

        if file_size > CHUNK_SIZE:
            print(f"Uploading {temp_file_path} in chunks to Azure Blob Storage.")
            upload_file_in_chunks(blob_service_client, container_name, blob_name, temp_file_path, CHUNK_SIZE)
        else:
            with open(temp_file_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            print(f"Uploaded {temp_file_path} to Azure Blob Storage in 'output' folder.")

    except AzureError as e:
        logging.error(f"Failed to upload {temp_file_path} to Azure Blob Storage: {str(e)}")
    except Exception as ex:
        logging.error(f"Unexpected error while uploading {temp_file_path}: {str(ex)}")
    finally:
        # Clean up the temporary file after upload
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Temporary file {temp_file_path} deleted.")

def process_pdfs(analysis_client, file_path):
    for input_pdf_path in file_path:
        base_filename = os.path.splitext(os.path.basename(input_pdf_path))[0]
 
        # Step 1: Deskew the PDF
        deskewed_pdf_path = f'{base_filename}_deskewed.pdf'
        deskew_pdf(input_pdf_path, deskewed_pdf_path)
 
        # Step 2: Split the deskewed PDF into pages
        pdf_pages = split_pdf(deskewed_pdf_path)
 
        # Step 3: Initialize the final output file path
        final_output_path = f'{base_filename}_grouped_response.txt'
 
        # Ensure the final output file is cleared before appending new content
        if os.path.exists(final_output_path):
            os.remove(final_output_path)
 
        # Initialize a list to accumulate grouped lines from all chunks
        all_grouped_lines = []
 
        for page_num, page_path in enumerate(pdf_pages, start=1):
            raw_output_path = f"{page_path}_raw_output.json"
 
            print(f"Processing page {page_num} of {len(pdf_pages)}: {page_path}")
            extracted_text = extract_text_and_coordinates_from_pdf(page_path, analysis_client, raw_output_path)
            print(f"Extracted text from page {page_num}:\n{extracted_text}")
 
            with open(raw_output_path, 'r', encoding='utf-8') as f:
                raw_file_contents = json.load(f)
 
            pages = organize_elements_by_page(raw_file_contents)
 
            grouped_lines = []
            for page_number, elements in sorted(pages.items()):
                grouped_lines.extend(group_by_position(elements))
 
            all_grouped_lines.append("\n".join(grouped_lines))
            all_grouped_lines.append("\nPAGE_SEPARATOR\n")
 
            os.remove(raw_output_path)
 
        with open(final_output_path, 'w', encoding='utf-8') as file:
            file.write("\n".join(all_grouped_lines))

 
        # upload_to_blob(final_output_path, container_name, blob_service_client)
        os.remove(final_output_path)
 
        os.remove(deskewed_pdf_path)
        for page in pdf_pages:
            os.remove(page)

        return all_grouped_lines

# Function to extract information from referral letters using GPT-35-Turbo
def extract_referral_information(text):
    if not text or text.strip() == "":
        return {"Error": "No text provided"}

    try:
        # Run the synchronous OpenAI completion method
        response = client.chat.completions.create(
            model=deployment_name,
            temperature=0.7,
            max_tokens=4000,
            top_p=1.0,  # Ensure top_p is set to 1.0 for deterministic behavior
            frequency_penalty=0.0,
            presence_penalty=0.0,
            messages=[
            {
                "role": "system",
                "content": '''You are an AI tasked with converting extracted text from a PDF document into a complete and well-structured JSON format. Your goal is to capture every detail from the text in a JSON format, with the correct hierarchy and structure. Follow these instructions carefully:

1. Identify all sections in the document, such as company details, invoice information, customer details, payment details, service information, taxes, and terms and conditions.
2. For each section:
   - Use clear, descriptive keys for each piece of information (e.g., "company_name," "invoice_number," "payment_terms").
   - Use nested objects for sub-sections (e.g., address details, bank details) and arrays for lists of repeating items (e.g., multiple charges, terms).
3. Ensure all details are included without omission. If a section is repeated or has multiple items, include them all in the appropriate format (e.g., an array of objects).
4. Retain the hierarchical relationships and context, preserving the structure as seen in the text.
5. Ensure the output is valid JSON format.

Convert the following extracted text into a detailed and comprehensive JSON object:

'''

},
            {"role": "user", "content": f"{text}"}
        ]
        )
#             messages=[
#             {
#                 "role": "system",
#                 "content": '''You are an AI assistant converting extracted text from a PDF into a structured JSON format. Your goal is to capture every section, detail, and relationship in the text accurately. Follow these instructions carefully:

# 1. Identify all distinct sections in the text, such as:
#    - **Company Details**: Include all company-related information like name, PAN, GST, and address.
#    - **Customer Information**: Capture all customer-related details, such as name, billing address, and order details.
#    - **Invoice Information**: Extract details like invoice number, invoice date, due date, and payment terms.
#    - **Payment Details**: Include all payment-related instructions, beneficiary information, bank details, and TDS instructions.
#    - **Service Information**: Detail the type of service provided, service ID, category, bandwidth, and charges.
#    - **Tax Information**: Capture all tax-related details, including breakdowns of SGST, CGST, and total amounts.
#    - **Terms and Conditions**: List each term and condition as an array of strings, including all sub-points.

# 2. Organize the information hierarchically, using:
#    - Nested objects for sub-sections (e.g., detailed address components, bank details).
#    - Arrays for lists or multiple items (e.g., multiple terms and conditions, line items).

# 3. Make sure no information is omitted. Include every piece of data, even if it seems minor. Use clear and descriptive keys for each item.

# 4. The output must be in a valid JSON format.

# Convert the following extracted text into a detailed JSON format:

# '''

# },
#             {"role": "user", "content": f"{text}"}
#         ]
#         )



        # Parse the response
        generated_text = response.choices[0].message.content
        return generated_text

    except OpenAIError as e:
        return {"Error": f"Error: {e}"}


def main():
    st.title("Invoice Information Extraction")
    
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Display the selected file name
        st.write(f"Uploaded file: {uploaded_file.name}")
        
        # Save the file temporarily
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        # Azure Blob and Analysis client initialization
        blob_service_client = BlobServiceClient.from_connection_string(
            st.secrets["azure"]["blob_connection_string"]
        )
        analysis_client = DocumentAnalysisClient(
            endpoint=st.secrets["azure"]["analysis_endpoint"], 
            credential=AzureKeyCredential(st.secrets["azure"]["analysis_api_key"])
        )
        
        # Call your process_pdfs function
        file_path = [temp_pdf_path]
        extracted_text = process_pdfs(analysis_client, file_path)

        if isinstance(extracted_text, list):
            extracted_text = " ".join(extracted_text)
        
        # Call the GPT-35-Turbo extraction method
        extracted_info = extract_referral_information(extracted_text)

        # Validate the extracted information and handle non-JSON data
        if extracted_info:
            try:
                json_data = json.loads(extracted_info)
                st.write("Extracted Invoice Information:")
                st.json(json_data)  # Display valid JSON
            except json.JSONDecodeError:
                # st.error("Extracted info is not valid JSON. Showing raw data:")
                # st.text(extracted_info)  # Show raw extracted data
                st.json({"extracted_info": extracted_info})

                # # Optionally convert the raw text to a simple JSON structure
                # st.json({"extracted_info": extracted_info})

            # Option to download the extracted data
            st.download_button(
                label="Download Extracted Information as JSON",
                data=json.dumps({"extracted_info": extracted_info}, indent=2),
                file_name="extracted_info.json",
                mime="application/json"
            )
        else:
            st.error("No information extracted.")

        # Option to upload the file to Azure Blob Storage
        if st.button("Upload to Azure Blob Storage"):
            upload_to_blob(extracted_info, "your_container_name", blob_service_client)
            st.success("File uploaded to Azure Blob Storage.")

if __name__ == "__main__":
    main()



    
 

 