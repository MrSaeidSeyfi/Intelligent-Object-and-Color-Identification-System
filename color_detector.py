import cv2
import numpy as np
import matplotlib.pyplot as plt

class ColorDetector:
    def __init__(self):
        self.color_labels = {
            'black': [0, 0, 0], 'white': [255, 255, 255], 'grey': [128, 128, 128],
            'silver': [192, 192, 192], 'blue': [0, 0, 255], 'red': [255, 0, 0],
            'green': [0, 255, 0], 'brown': [165, 42, 42], 'beige': [245, 245, 220],
            'golden': [255, 215, 0], 'bordeaux': [128, 0, 0], 'yellow': [255, 255, 0],
            'orange': [255, 165, 0], 'violet': [148, 0, 211]
        }
        self.color_labels_rgb = np.array(list(self.color_labels.values()))

    def convert_to_closest_color(self, image):
        print("Converting image to closest predefined colors...")
        image_reshaped = image.reshape((-1, 3))
        distances = np.linalg.norm(image_reshaped[:, np.newaxis] - self.color_labels_rgb, axis=2)
        closest_indices = np.argmin(distances, axis=1)
        mapped_image = self.color_labels_rgb[closest_indices].reshape(image.shape).astype(np.uint8)
        return mapped_image

    def segment_car_body(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        print(f"Performing Graph Cut segmentation on bounding box: {bbox}")

        # Graph Cut segmentation
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (x1, y1, x2 - x1, y2 - y1)
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask_graphcut = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        graphcut_img = image * mask_graphcut[:, :, np.newaxis]

        print("Graph Cut segmentation completed. Finding most frequent color...")
        # Find the most frequent color in the segmented region
        segmented_pixels = graphcut_img[mask_graphcut > 0].reshape(-1, 3)
        if len(segmented_pixels) == 0:
            most_frequent_color = [0, 0, 0]  # Default to black if no pixels are found
        else:
            unique_colors, counts = np.unique(segmented_pixels, axis=0, return_counts=True)
            most_frequent_color = unique_colors[np.argmax(counts)]
        print(f"Most frequent color in segmented region: {most_frequent_color}")

        # Replace all colors in the segmented image with the most frequent color
        graphcut_img[mask_graphcut > 0] = most_frequent_color

        print("Performing edge-based segmentation...")
        # Edge-based segmentation
        cropped_img = image[y1:y2, x1:x2]
        blurred_img = cv2.GaussianBlur(cropped_img, (5, 5), 0)
        edges = cv2.Canny(blurred_img, 50, 150)
        mask_edges = cv2.dilate(edges, None, iterations=2)
        mask_edges = cv2.erode(mask_edges, None, iterations=2)
        mask_edges = cv2.bitwise_not(mask_edges)
        edge_segmented_img = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_edges)

        mask_edges_full = np.zeros_like(mask)
        mask_edges_full[y1:y2, x1:x2] = mask_edges

        print("Combining Graph Cut and edge-based segmentations...")
        # Combine both segmentations by finding the intersection
        combined_mask = cv2.bitwise_and(mask_graphcut, mask_edges_full)
        segmented_img = cv2.bitwise_and(image, image, mask=combined_mask)

        return graphcut_img, edge_segmented_img, segmented_img, combined_mask

    def detect_color(self, image, bbox):
        mapped_image = self.convert_to_closest_color(image)
        graphcut_img, edge_segmented_img, combined_img, combined_mask = self.segment_car_body(mapped_image, bbox)

        print("Calculating the dominant color in the combined segmented image...")
        # Calculate the dominant color in the segmented car body region only
        segmented_pixels = combined_img[combined_mask > 0].reshape(-1, 3)
        if len(segmented_pixels) == 0:
            dominant_color = [0, 0, 0]  # Default to black if no pixels are found
            color_name = "black"
        else:
            unique_colors, counts = np.unique(segmented_pixels, axis=0, return_counts=True)
            dominant_color = unique_colors[np.argmax(counts)]
            color_name = self.find_closest_color(dominant_color)
        print(f"Dominant color in the segmented image: {dominant_color}")

        # Display results using matplotlib
        fig, axs = plt.subplots(1, 6, figsize=(30, 5))
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(cv2.cvtColor(graphcut_img, cv2.COLOR_BGR2RGB))
        axs[1].set_title('GraphCut Segmented')
        axs[1].axis('off')

        axs[2].imshow(cv2.cvtColor(edge_segmented_img, cv2.COLOR_BGR2RGB))
        axs[2].set_title('Edge-Based Segmented')
        axs[2].axis('off')

        axs[3].imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        axs[3].set_title('Combined Segmented')
        axs[3].axis('off')
 
        color_rgb = self.color_labels.get(color_name)
        color_square = np.zeros((100, 100, 3), dtype=np.uint8)
        color_square[:] = color_rgb
        axs[4].set_title(f'Dominant Color: {color_name}')
        axs[4].imshow(color_square)


        # Log individual masks
        axs[5].imshow(combined_mask, cmap='gray')
        axs[5].set_title('Combined Mask')
        axs[5].axis('off')

        plt.show()

        print(f"Dominant color: {dominant_color}, Closest color name: {color_name}")
        return color_name

    def find_closest_color(self, bgr_color):
        # Convert BGR to RGB
        rgb_color = bgr_color[::-1]
        print(f"Converted BGR {bgr_color} to RGB {rgb_color}")
        min_distance = float('inf')
        closest_color = None
        for color_name, rgb_value in self.color_labels.items():
            distance = np.linalg.norm(np.array(rgb_color) - np.array(rgb_value))
            print(f"Comparing {rgb_color} with {color_name}: distance = {distance}")
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        return closest_color
 