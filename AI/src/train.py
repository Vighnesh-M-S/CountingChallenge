from ultralytics import YOLO

# Load model (use 'yolov8n-seg.pt' for small model)
model = YOLO('yolov8n-seg.pt')

# Train model
model.train(
    data='AI/data/nuts_bolts.yaml',  # Dataset config file
    epochs=100,
    imgsz=640,
    batch=8,
    name='nuts_bolts_segmentation'
)
