import cv2
import pytesseract
import os

# Define the folder containing the images
folder_path = "plates"

# Tesseract configuration for optimal performance
custom_config = r'--oem 3 --psm 6'

# Output file for raw text (single .txt file)
output_file = "all_raw_texts.txt"

# Open the output file in write mode with UTF-8 encoding
with open(output_file, "w", encoding="utf-8") as f:
    # Iterate through all images in the "plates" folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
            image_path = os.path.join(folder_path, filename)

            # Load and preprocess the image
            image = cv2.imread(image_path)

            # Resize the image for better recognition
            image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(image, (5, 5), 0)

            # Convert to grayscale and apply adaptive thresholding
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Perform OCR on the preprocessed image
            raw_text = pytesseract.image_to_string(thresh, config=custom_config, lang='khm+eng')

            # Write the OCR result for each image into the same file
            f.write(f"\n[Raw Text from {filename}]:\n")
            f.write(raw_text + "\n")

print(f"All raw texts saved to {output_file}")
