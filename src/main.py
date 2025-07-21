import os
import csv
from src.cropper import segment_plate
from src.detect_wells import detect_wells_from_image, draw_well_groups
from src.mpn_table import mpn_dict

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
CSV_OUTPUT_PATH = os.path.join(RESULTS_DIR, "results.csv")

def get_mpn_result(num_large, num_small):
    return mpn_dict.get((num_large, num_small), "Not defined")

def process_all_images():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Abrir o CSV para escrita
    with open(CSV_OUTPUT_PATH, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_name", "large_wells", "small_wells", "mpn_value"])

        for filename in os.listdir(RAW_DIR):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            raw_path = os.path.join(RAW_DIR, filename)
            print(f"Processing {filename}...")

            try:
                segmented = segment_plate(raw_path)
                small_positives, large_positives = detect_wells_from_image(segmented)

                num_large = len(large_positives)
                num_small = len(small_positives)
                mpn_result = get_mpn_result(num_large, num_small)

                # Salvar imagem com contornos desenhados
                name, _ = os.path.splitext(filename)
                output_path = os.path.join(PROCESSED_DIR, f"{name}_contours.jpg")
                draw_well_groups(segmented, small_positives, large_positives, output_path)

                # Escrever linha no CSV
                writer.writerow([filename, num_large, num_small, mpn_result])

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    process_all_images()
