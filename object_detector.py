from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, image):
        results = self.model(image)
        return results
