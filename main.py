from object_detector import ObjectDetector
from color_detector import ColorDetector
from utils import load_image, preprocess_image
import cv2

def draw_bounding_box(image, bbox, label, confidence):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def scale_bounding_box(bbox, original_shape, processed_shape):
    orig_h, orig_w = original_shape[:2]
    proc_h, proc_w = processed_shape[:2]
    x_scale = orig_w / proc_w
    y_scale = orig_h / proc_h

    x1, y1, x2, y2 = bbox
    return [x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale]

def main(image_path):
    image = load_image(image_path)
    original_shape = image.shape
    processed_image = preprocess_image(image)
    processed_shape = processed_image.shape

    object_detector = ObjectDetector(model_path='yolov8x.pt')
    color_detector = ColorDetector()

    results = object_detector.detect_objects(processed_image)

    for result in results:
        for box in result.boxes:
            confidence = box.conf.item()
            if confidence > 0.90:  # Filter boxes with confidence > 90%
                bbox = box.xyxy[0].tolist()
                bbox = scale_bounding_box(bbox, original_shape, processed_shape)
                class_name = object_detector.model.names[int(box.cls.item())]
                color = color_detector.detect_color(image, bbox)
                draw_bounding_box(image, bbox, f'{class_name}, {color}', confidence)
                print(f"Detected {class_name} with bounding box {bbox} and color {color} (confidence: {confidence:.2f})")

    output_path = "output_" + image_path
    cv2.imwrite(output_path, image)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
    else:
        main(sys.argv[1])
