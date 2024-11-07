# import cv2
# import pytesseract
# import os

# # Define the folder containing the images and output folder
# folder_path = "plates"
# output_folder = "output_raw"

# # Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Tesseract configuration for optimal performance
# custom_config = r'--oem 3 --psm 6'

# # Iterate through all images in the "plates" folder
# for filename in os.listdir(folder_path):
#     if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
#         image_path = os.path.join(folder_path, filename)

#         # Load and preprocess the image
#         image = cv2.imread(image_path)

#         # Resize the image for better recognition
#         image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

#         # Apply Gaussian blur to reduce noise
#         blurred = cv2.GaussianBlur(image, (5, 5), 0)

#         # Convert to grayscale and apply adaptive thresholding
#         gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
#         thresh = cv2.adaptiveThreshold(
#             gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#         )

#         # Perform OCR on the preprocessed image
#         raw_text = pytesseract.image_to_string(thresh, config=custom_config, lang='khm+eng')

#         # Save the raw text to a .txt file
#         output_file = os.path.join(output_folder, f"{filename}_raw.txt")
#         with open(output_file, "w", encoding="utf-8") as f:
#             f.write(raw_text)

#         print(f"Saved raw text to {output_file}")
import cv2
import pytesseract
import re
import os

# Define the folder containing the images and output folder
folder_path = "plates"
output_folder = "output_cleaned"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Tesseract configuration for optimal performance
custom_config = r'--oem 3 --psm 6'

# Define a function to clean up common OCR errors
def clean_text(text):
    # Remove trailing or isolated characters (like "ន" at the end)
    cleaned = re.sub(r'\s+[ន]*$', '', text.strip())
    return cleaned

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

        # Apply the clean_text function to each line of the OCR result
        lines = raw_text.strip().split('\n')
        cleaned_lines = [clean_text(line) for line in lines]

        # Join cleaned lines into a single string
        cleaned_text = "\n".join(cleaned_lines)

        # Save the cleaned text to a .txt file
        output_file = os.path.join(output_folder, f"{filename}_cleaned.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print(f"Saved cleaned text to {output_file}")
