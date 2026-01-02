from ultralytics import YOLO
import cv2
from pathlib import Path
MODEL_PATH = r"runs/detect/train2/weights/best.pt"
IMAGE_PATH = r"<input img path> "
OUT_PATH   = r"<specify the output path"
model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH, conf=0.25)
r = results[0]
img = r.orig_img.copy()
boxes = r.boxes.xyxy
for i, box in enumerate(boxes, start=1):
    x1, y1, x2, y2 = box.tolist()[:4]
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    cv2.putText(
        img,
        str(i),
        (cx, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),  
        2,
        cv2.LINE_AA,
    )
h, w = img.shape[:2]
text = f"Total shrimp: {len(boxes)}"
cv2.putText(img, text,
            (20, h - 20),    
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 3, cv2.LINE_AA)
print(f"Detected shrimps: {len(boxes)}")
cv2.imwrite(OUT_PATH, img)
print(f"Saved numbered image to: {OUT_PATH}")