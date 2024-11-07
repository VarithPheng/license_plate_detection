from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLOv8 model
    model = YOLO('model/yolov8s.pt')

    # Enhanced training settings with adjustments for the 3060 Ti
    model.train(
        data='F:/tii/licenseplatedetection/datasets/data.yaml',  # Full path to data.yaml
        epochs=100,                # Increase epochs for more training, possibly improve accuracy
        imgsz=800,                 # Adjusted image size to reduce VRAM usage
        batch=16,                  # Reduced batch size to fit into 8 GB VRAM
        name='carplate_yolo_4',
        project='runs/detect',
        device=0,                  # Use the first GPU
        half=True,                 # Mixed precision, faster on compatible GPUs
        workers=8,                 # Parallel data loading
        optimizer='AdamW',         # AdamW optimizer, may yield better results for object detection
        lr0=0.0005,                # Initial learning rate, monitor for stability
        val=True,                  # Enable validation after each epoch
    )

    # Validate the model after training
    model.val()
