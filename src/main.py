import os
import cv2
from src.cropper import segment_plate
from src.detect_wells import detect_wells_from_image, draw_contours_and_save

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def process_all_images():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for filename in os.listdir(RAW_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        raw_path = os.path.join(RAW_DIR, filename)
        print(f"Processing {filename}...")

        try:
            segmented = segment_plate(raw_path)
            contours = detect_wells_from_image(segmented)

            name, _ = os.path.splitext(filename)
            output_path = os.path.join(PROCESSED_DIR, f"{name}_contours.jpg")
            draw_contours_and_save(segmented, contours, output_path)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    process_all_images()
