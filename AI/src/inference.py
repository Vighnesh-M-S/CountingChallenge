import cv2
from ultralytics import YOLO
import os

model = YOLO('AI/model/best.pt')  # Path to your trained weights
image_path = 'AI/data/test/sample.jpg'
results = model(image_path)

# Count objects and draw masks
for r in results:
    print(f"Counted Objects: {len(r.boxes)}")
    result_image = r.plot()
    cv2.imwrite('AI/data/output/segmented_sample.jpg', result_image)