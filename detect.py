from ultralytics import YOLO
import cv2

class HandwritingDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image)[0]
        boxes = []

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            # assume class 0 = handwriting
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))

        return boxes

    def crop(self, image, boxes):
        return [image[y1:y2, x1:x2] for x1,y1,x2,y2 in boxes]
