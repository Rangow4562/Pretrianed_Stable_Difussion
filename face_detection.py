from ultralytics import YOLO
import torch
import cv2
import numpy as np

class FaceDetection:
    def __init__(self, model_path="yolov8n-face.pt", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.set_device(0) if torch.cuda.is_available() else None
        else:
            self.device = device

        self.model = YOLO(model_path).to(self.device)

    def detect_face(self, image):
        self.results = self.model(image)
        classes = [self.model.names[int(box.data[0][-1])] for result in self.results for box in result.boxes]
        return classes
    
    def crop_face(self, image, padding=150):
        cropped_image = None
        image = np.array(image)
        if self.results and len(self.results[0].boxes.xyxy) > 0:
            box = self.results[0].boxes.xyxy[0]
            x_min, y_min, x_max, y_max = map(int, box)
            new_x_min = max(0, x_min - padding)
            new_y_min = max(0, y_min - padding)
            new_x_max = min(image.shape[1], x_max + padding)
            new_y_max = min(image.shape[0], y_max + padding)
            cropped_image = image[new_y_min:new_y_max, new_x_min:new_x_max]
        return cropped_image
    
if __name__ == "__main__":
    detector = FaceDetection()
    image_path = "images.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_classes = detector.detect_face(image)
    print("Number of detected classes:", len(detected_classes))
