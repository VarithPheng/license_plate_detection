import easyocr
import os
import re

# Initialize the OCR reader with GPU support
reader = easyocr.Reader(['en'], gpu=True)

def clean_and_filter_text(text):
    """
    Clean and filter the OCR recognized text to match specific patterns.
    """
    # Remove spaces and make uppercase for consistency
    text = text.upper().replace(' ', '')

    # Replace ":" with "-" for license plate formatting
    text = text.replace(":", "-")
    
    # Define allowed patterns
    patterns = [
        r'^[A-Z0-9]{2,3}-\d{4}$',  # Pattern for license plate (e.g., "2BY-7369")
        r'^PHNOM$',                # Pattern for "PHNOM"
        r'^PENH$'                  # Pattern for "PENH"
    ]
    
    # Check if text matches any of the allowed patterns
    for pattern in patterns:
        if re.match(pattern, text):
            return text
    
    return None  # Return None if it doesn't match any pattern

def extract_text_from_image(image_path):
    """
    Read the license plate text directly from the image and write the result to text.txt.
    """
    # Perform OCR directly on the image without pre-processing
    detections = reader.readtext(image_path)

    # Collect recognized and filtered text
    extracted_text = ""
    for detection in detections:
        _, text, _ = detection
        filtered_text = clean_and_filter_text(text)
        
        if filtered_text:  # Only add if the text matches the patterns
            extracted_text += filtered_text + "\n"

    # Save the extracted text to text.txt in UTF-8 encoding
    with open("text.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print("Extracted and filtered text has been saved to text.txt.")
    print("Filtered text:\n", extracted_text)

# Specify the path to your license plate image
image_path = '6.jpg'

# Extract text from the image
extract_text_from_image(image_path)
