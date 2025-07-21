"""
Wells detection module for Colilert Quanti-Tray analysis.

This module provides functionality to detect and classify positive wells
in Colilert Quanti-Tray images. The system detects wells that have turned
yellow/fluorescent indicating the presence of E. coli and coliform bacteria
in water samples. Wells are classified by size (large and small) for proper
MPN calculation according to IDEXX Colilert methodology.
"""

import cv2
import numpy as np


def clean_mask(mask):
    """
    Clean a binary mask using morphological operations.
    
    Applies opening followed by closing operations to remove noise
    and fill small gaps in the binary mask. This helps improve
    well detection accuracy by cleaning up the thresholded image.
    
    Args:
        mask (numpy.ndarray): Binary mask to be cleaned.
        
    Returns:
        numpy.ndarray: Cleaned binary mask.
    """
    # Apply morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def detect_wells_from_image(image, area_threshold=2000):
    """
    Detect and classify positive wells in a Colilert Quanti-Tray image.
    
    This function processes the image using LAB color space conversion,
    thresholding, and contour detection to identify positive wells that
    have turned yellow due to bacterial presence. Wells are
    classified as either 'small' or 'large' based on their area and 
    position for proper MPN calculation according to IDEXX methodology.
    
    Args:
        image (numpy.ndarray): Input BGR image of the Quanti-Tray.
        area_threshold (int, optional): Area threshold in pixels to distinguish
            between small and large wells. Defaults to 2000.
            
    Returns:
        tuple: A tuple containing:
            - small_positives (list): List of contours for positive small wells.
            - large_positives (list): List of contours for positive large wells.
            
    Note:
        - Contours in the top 10% of the image are merged together. This is due to the largest well, sometimes being incomplete or registered as multiple small wells.
        - Small wells are only counted if they're in the bottom 30% of the image.
        - Contours smaller than 100 pixels are filtered out as noise.
        - Uses LAB color space B channel thresholding to detect yellow wells.
    """
    # Convert to LAB color space and threshold on the B channel to detect yellow wells
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, mask = cv2.threshold(lab[:, :, 2], 160, 255, cv2.THRESH_BINARY)
    mask_clean = clean_mask(mask)

    # Find contours in the cleaned mask
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define image regions for well classification (based on Quanti-Tray layout)
    h = image.shape[0]
    top_limit = h * 0.1      # Top 10% of image
    bottom_limit = h * 0.7   # Bottom 30% of image

    top_contours = []
    usable_contours = []

    # Separate contours by vertical position and filter by area
    for c in contours:
        if cv2.contourArea(c) < 100:  # Filter out noise
            continue
        _, y, _, _ = cv2.boundingRect(c)
        if y < top_limit:
            top_contours.append(c)
        else:
            usable_contours.append(c)

    # Merge contours from the top region
    if top_contours:
        merged_top = np.vstack(top_contours)
        usable_contours.append(merged_top)

    # Classify contours as small or large positive wells
    small_positives = []
    large_positives = []

    for c in usable_contours:
        area = cv2.contourArea(c)
        _, y, _, _ = cv2.boundingRect(c)

        if area < area_threshold:
            if y >= bottom_limit:  # Small wells only in bottom region
                small_positives.append(c)
        else:
            large_positives.append(c)

    return small_positives, large_positives


def draw_well_groups(img, small_contours, large_contours, output_path):
    """
    Draw detected positive wells on the image and save the result.
    
    This function visualizes the detection results by drawing contours
    around identified positive wells using different colors for small and 
    large wells. This helps in manual verification of the automated detection.
    
    Args:
        img (numpy.ndarray): Input image to draw on.
        small_contours (list): List of contours for positive small wells.
        large_contours (list): List of contours for positive large wells.
        output_path (str): Path where the annotated image will be saved.
        
    Note:
        - Small positive wells are drawn in green color.
        - Large positive wells are drawn in red color.
        - Contour thickness is set to 2 pixels for visibility.
    """
    result = img.copy()
    
    # Draw small positive wells in green
    for c in small_contours:
        cv2.drawContours(result, [c], -1, (0, 255, 0), 2)  
    
    # Draw large positive wells in red
    for c in large_contours:
        cv2.drawContours(result, [c], -1, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, result)


if __name__ == "__main__":
    # Example usage for testing the detection functions
    image_path = "data/processed/segmented_image.jpg"
    output_path = "data/processed/image1_debug_contours.jpg"

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    small, large = detect_wells_from_image(img)

    print(f"Colilert Quanti-Tray analysis completed successfully!")
    print(f"Total positive wells detected: {len(small) + len(large)}")
    print(f"Small positive wells: {len(small)}")
    print(f"Large positive wells: {len(large)}")

    draw_well_groups(img, small, large, output_path)
    print(f"Results saved to: {output_path}")

    cv2.imshow("Detected Positive Wells", cv2.imread(output_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

