import cv2
import pytesseract
import re
import os

# Define the folder containing the images
folder_path = "test_output"

# Tesseract configuration for optimal performance
custom_config = r'--oem 3 --psm 6'

# Output file for cleaned text
cleaned_output_file = "all_cleaned_texts.txt"

# Function to clean unwanted symbols and trailing characters
def clean_text(text):
    # Keep only alphanumeric, Khmer script, spaces, and hyphens
    cleaned = re.sub(r'[^a-zA-Z0-9\u1780-\u17FF\s-]', '', text)
    # Remove trailing isolated "ន" or extra spaces
    cleaned = re.sub(r'\s+[ន]*$', '', cleaned.strip())
    return cleaned

# Function to process an image and extract cleaned text
def process_image(image_path):
    # Load and resize the image for better OCR performance
    image = cv2.imread(image_path)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert to grayscale and apply adaptive thresholding
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Perform OCR on the thresholded image
    raw_text = pytesseract.image_to_string(thresh, config=custom_config, lang='khm+eng')

    # Clean and format the OCR output
    lines = raw_text.strip().split('\n')
    cleaned_lines = [clean_text(line) for line in lines if clean_text(line)]

    return "\n".join(cleaned_lines)

# Open the cleaned output file in write mode with UTF-8 encoding
with open(cleaned_output_file, "w", encoding="utf-8") as cleaned_f:
    # Iterate through all images in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
            image_path = os.path.join(folder_path, filename)

            # Extract cleaned text from the image
            cleaned_text = process_image(image_path)

            # Write the cleaned text for each image into the output file
            cleaned_f.write(f"\n[Cleaned Text from {filename}]:\n{cleaned_text}\n")

print(f"All cleaned texts saved to {cleaned_output_file}")
