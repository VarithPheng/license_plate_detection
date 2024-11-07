import cv2
import easyocr
import os

# Define input folder containing images and output folder
input_folder = 'test'  # Folder containing images to process
text_file = 'text.txt'    # Text file to store OCR results



# Initialize EasyOCR with English language model (add 'km' if you want Khmer)
reader = easyocr.Reader(['en'])  # Add 'km' to the list if Khmer text is needed

# Open the text file in write mode
with open(text_file, 'w', encoding='utf-8') as f:
    # Loop through all image files in the input folder
    for image_file in os.listdir(input_folder):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):  # Check for valid image formats
            image_path = os.path.join(input_folder, image_file)
            
            # Load the image
            image = cv2.imread(image_path)

            # Perform OCR using EasyOCR directly on the original image
            result = reader.readtext(image, detail=0)

            # Save OCR results to the text file
            if result:
                f.write(f"Image: {image_file}\n")
                for line in result:
                    f.write(line + "\n")
                f.write("-" * 50 + "\n")  # Separator for readability

print("Processing complete. Text saved to 'text.txt'.")