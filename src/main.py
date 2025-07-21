"""
Main application module for Colilert Quanti-Tray analysis system.

This module orchestrates the complete pipeline for processing Colilert Quanti-Tray images:
1. Image segmentation to extract the tray area
2. Positive well detection and classification by size
3. MPN (Most Probable Number)/100 mL calculation according to IDEXX methodology
4. Results compilation and CSV export

The system processes all images in the raw data directory and generates
annotated images with detection results and a CSV report with MPN values.
"""

import os
import csv
from src.cropper import segment_plate
from src.detect_wells import detect_wells_from_image, draw_well_groups
from utils.mpn_table import mpn_dict

# Directory and file path configurations
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
CSV_OUTPUT_PATH = os.path.join(RESULTS_DIR, "results.csv")


def get_mpn_result(num_large, num_small):
    """
    Retrieve the MPN value based on positive well counts.
    
    Looks up the Most Probable Number (MPN) value from the standardized
    IDEXX Colilert MPN table using the counts of large and small positive wells.
    This follows the official IDEXX Colilert methodology for water quality testing.
    
    Args:
        num_large (int): Number of large positive wells detected.
        num_small (int): Number of small positive wells detected.
        
    Returns:
        str or float: The corresponding MPN value per 100mL, or "Not defined" if
                     the combination is not found in the lookup table.
    """
    return mpn_dict.get((num_large, num_small), "Not defined")


def process_all_images():
    """
    Process all Colilert Quanti-Tray images in the raw data directory.
    
    This function implements the complete image processing pipeline for
    water quality analysis using the IDEXX Colilert method:
    1. Creates necessary output directories
    2. Iterates through all image files in the raw directory
    3. For each Quanti-Tray image:
       - Segments the tray from the background
       - Detects and classifies positive wells (yellow/fluorescent)
       - Calculates MPN value according to IDEXX methodology
       - Generates annotated output image showing detection results
       - Records results in CSV file for further analysis
    4. Handles and reports processing errors
    
    The function processes common image formats (JPG, JPEG, PNG) and
    saves results to a CSV file with columns: image_name, large_wells,
    small_wells, mpn_value.
    
    Raises:
        Exception: Individual image processing errors are caught and logged
                  but do not stop the overall processing.
    """
    # Ensure output directories exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Initialize CSV output file with headers
    with open(CSV_OUTPUT_PATH, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_name", "large_wells", "small_wells", "mpn_value"])

        # Process each Quanti-Tray image file in the raw directory
        for filename in os.listdir(RAW_DIR):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            raw_path = os.path.join(RAW_DIR, filename)
            print(f"Processing Quanti-Tray image: {filename}...")

            try:
                # Step 1: Segment the Quanti-Tray from the image
                segmented = segment_plate(raw_path)
                
                # Step 2: Detect and classify positive wells
                small_positives, large_positives = detect_wells_from_image(segmented)

                # Step 3: Count positive wells and calculate MPN
                num_large = len(large_positives)
                num_small = len(small_positives)
                mpn_result = get_mpn_result(num_large, num_small)

                # Step 4: Generate annotated output image
                name, _ = os.path.splitext(filename)
                output_path = os.path.join(PROCESSED_DIR, f"{name}_contours.jpg")
                draw_well_groups(segmented, small_positives, large_positives, output_path)

                # Step 5: Record results in CSV
                writer.writerow([filename, num_large, num_small, mpn_result])
                
                print(f"  ✓ Completed: {num_large} large, {num_small} small positive wells (MPN: {mpn_result}/100mL)")

            except Exception as e:
                print(f"  ✗ Failed to process {filename}: {e}")
                
    print(f"\nColilert Quanti-Tray analysis completed! Results saved to: {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    print("Starting Colilert Quanti-Tray analysis system...")
    process_all_images()
