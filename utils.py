import cv2

def load_image(image_path):
    return cv2.imread(image_path)

def preprocess_image(image, img_size=640):
    image = cv2.resize(image, (img_size, img_size))
    return image
