import supervision as sv
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Initialize YOLOv8 model (you can replace 'yolov8n.pt' with your custom-trained model)
model = YOLO('../model/yolo.pt')  # Use 'yolov8x.pt' or any other YOLOv8 model if needed

# Path to the directory containing images
image_folder = r"F:\tii\licenseplatedetection\images"

# List all image files in the directory
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

# Iterate over each image file
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(image_folder, image_file)
    
    # Load the image
    image = cv2.imread(image_path)

    # Perform detection
    results = model(image)[0]  # Get the first batch result

    # Extract predictions in YOLO format (x1, y1, x2, y2, confidence, class_id)
    detections = np.array(results.boxes.data.cpu())

    # Separate bounding boxes, confidences, and class IDs
    xyxy = detections[:, :4]
    confidences = detections[:, 4]
    class_ids = detections[:, 5].astype(int)

    # Initialize Detections object for supervision
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidences,
        class_id=class_ids
    )

    # Initialize annotators
    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator()

    # Annotate the image
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    labels = [model.names[class_id] for class_id in class_ids]  # Get labels from YOLO model
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # Display the annotated image
    sv.plot_image(image=annotated_image, size=(16, 16))

    # Optionally save the annotated image
    # cv2.imwrite(f"annotated_{image_file}", annotated_image)
