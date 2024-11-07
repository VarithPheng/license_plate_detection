from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import os

# Step 1: Load your trained YOLOv8 model
model = YOLO('model/yolo.pt')  # Path to your trained weights

# Step 2: Define the path to the directory containing test images
image_folder = "images"  # Replace with your folder path

# Step 3: List all image files in the directory
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Step 4: Create a folder to store cropped images (if it doesn't exist)
cropped_folder = "cropped"
os.makedirs(cropped_folder, exist_ok=True)

# Step 5: Iterate over each image file
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(image_folder, image_file)

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to load image {image_file}. Skipping...")
        continue  # Skip if the image can't be loaded

    # Perform detection using YOLOv8
    results = model(image)[0]  # Get the first result

    # Extract predictions from YOLOv8 results
    xyxy = results.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Class IDs

    # Initialize Supervision Detections object
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidences,
        class_id=class_ids
    )

    # Step 6: Crop detected objects and save them in the 'cropped' folder
    for idx, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, box)  # Extract coordinates and convert to int
        cropped_img = image[y1:y2, x1:x2]  # Crop the detected object

        # Save the cropped image without annotations
        cropped_image_path = os.path.join(cropped_folder, f"{image_file}_crop_{idx}.jpg")
        cv2.imwrite(cropped_image_path, cropped_img)
        print(f"Cropped image saved: {cropped_image_path}")

    # Initialize annotators for the display image only (optional)
    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator()

    # Get class names from YOLO model (for labels)
    labels = [model.names.get(class_id, f"Class {class_id}") for class_id in class_ids]

    # Annotate the image for display purposes only
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # Display the annotated image (optional)
    # cv2.imshow(f"Annotated Image - {image_file}", annotated_image)
    # cv2.waitKey(0)  # Wait until a key is pressed to close the window
    # cv2.destroyAllWindows()

print("Processing completed.")
