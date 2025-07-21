"""
Image segmentation module for Colilert Quanti-Tray analysis.

This module provides functionality to segment Colilert Quanti-Tray plates from images
using GrabCut algorithm and contour detection techniques. The Quanti-Tray is used
for water quality testing to detect E. coli and coliform bacteria.
"""

import cv2
import numpy as np


def segment_plate(image_path):
    """
    Segment and extract the Colilert Quanti-Tray from an image.
    
    This function uses the GrabCut algorithm to perform foreground/background
    segmentation, followed by contour detection to find the largest connected
    component (assumed to be the Quanti-Tray). The result is a cropped image
    containing only the segmented tray area.
    
    Args:
        image_path (str): Path to the input image file containing the Quanti-Tray.
        
    Returns:
        numpy.ndarray: Cropped BGR image containing only the segmented tray area.
        
    Raises:
        FileNotFoundError: If the image file cannot be loaded.
        ValueError: If no contours are found in the segmented image.

    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Initialize GrabCut algorithm with rectangular region of interest
    rect = (100, 100, img.shape[1] - 100, img.shape[0] - 100)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut algorithm for foreground/background segmentation
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = img * mask2[:, :, np.newaxis]

    # Convert to grayscale and create binary mask
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours to identify the tray boundary
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the segmented image!")

    # Select the largest contour (assumed to be the Quanti-Tray)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create mask from the largest contour and apply it
    new_mask = np.zeros_like(gray)
    cv2.drawContours(new_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to extract the tray region
    result = cv2.bitwise_and(img, img, mask=new_mask)

    # Crop to the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = result[y:y+h, x:x+w]

    return cropped


if __name__ == "__main__":
    # Example usage for testing the segmentation function
    file_path = "data/raw/image1.jpg"
    segmented_image = segment_plate(file_path)
    cv2.imwrite("data/processed/segmented_image.jpg", segmented_image)
    print("Quanti-Tray segmentation completed successfully.")