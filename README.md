
# Intelligent Object and Color Identification System

This project demonstrates object detection and color detection using YOLOv8 and OpenCV. The project includes scripts for detecting objects in images and identifying their colors using segmentation techniques.

## Project Structure

```
Intelligent-Object-and-Color-Identification-System
│   README.md
│   requirements.txt 
├───main.py
├───object_detector.py
├───color_detector.py
├───utils.py
├───images 
│   ├───output_alfa.jpg
│   ├───output_laferarri.png
```
 
## Installation

1. Clone the repository:

```bash
git clone https://github.com/MrSaeidSeyfi/Intelligent-Object-and-Color-Identification-System.git
cd Intelligent-Object-and-Color-Identification-System
```

2. Create a virtual environment and activate it:

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the object and color detection on an image, use the following command:

```bash
python main.py <image_path>
```

Replace `<image_path>` with the path to the image file you want to process.

Example:

```bash
python main.py images/laferarri.png
```

## Color Detection Approach

The color detection approach in this project involves several key steps:

1. **Image Segmentation**: The `ColorDetector` class segments the object from the background using a combination of Graph Cut and edge-based segmentation. This isolates the object, making it easier to analyze its color.

2. **Color Mapping**: Once the object is segmented, the image is mapped to the closest predefined colors. This is achieved by comparing each pixel's color in the segmented area to a set of predefined colors using Euclidean distance in the RGB color space.

3. **Dominant Color Detection**: The most frequent color within the segmented region is identified as the dominant color. This is done by counting the occurrences of each color and selecting the one with the highest count.

4. **Final Output**: The detected color is associated with the detected object, and the bounding box along with the color label is drawn on the original image. This provides a clear visual representation of both the object and its color.
 
## License

This project is licensed under the MIT License.
