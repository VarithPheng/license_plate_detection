import os
import cv2
import torch
from craft_text_detector import Craft

# Paths
input_folder = 'images'  # Folder containing input images
output_file = 'text_boxes.txt'  # Output file to save detected bounding boxes

# Initialize CRAFT
craft = Craft(output_dir='craft_output', crop_type="poly", cuda=torch.cuda.is_available())


# Open output file to write bounding box results
with open(output_file, 'w', encoding='utf-8') as f:
    # Process each image in the input folder
    for image_file in os.listdir(input_folder):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_folder, image_file)

            # Perform text detection with CRAFT
            prediction_result = craft.detect_text(image_path)

            # Get bounding boxes for detected text
            bboxes = prediction_result["boxes"]

            # Read the image with OpenCV for further processing (if needed)
            image = cv2.imread(image_path)

            # Write the image name and bounding boxes to the output file
            f.write(f"Image: {image_file}\n")
            for bbox in bboxes:
                f.write(f"Bounding box: {bbox}\n")
            f.write("-" * 50 + "\n")  # Separator for readability

# Clean up by unloading CRAFT models
craft.unload_craftnet_model()
craft.unload_refinenet_model()

print("Processing complete. Bounding boxes saved to 'text_boxes.txt'.")
