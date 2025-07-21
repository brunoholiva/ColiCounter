import cv2
import numpy as np

def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed

def detect_wells_from_image(image, area_threshold=2000):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, mask = cv2.threshold(lab[:, :, 2], 160, 255, cv2.THRESH_BINARY)
    mask_clean = clean_mask(mask)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h = image.shape[0]
    top_limit = h * 0.1
    bottom_limit = h * 0.7

    top_contours = []
    usable_contours = []

    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        _, y, _, _ = cv2.boundingRect(c)
        if y < top_limit:
            top_contours.append(c)
        else:
            usable_contours.append(c)

    if top_contours:
        merged_top = np.vstack(top_contours)
        usable_contours.append(merged_top)

    small_positives = []
    large_positives = []

    for c in usable_contours:
        area = cv2.contourArea(c)
        _, y, _, _ = cv2.boundingRect(c)

        if area < area_threshold:
            if y >= bottom_limit:  
                small_positives.append(c)
        else:
            large_positives.append(c)

    return small_positives, large_positives

def draw_well_groups(img, small_contours, large_contours, output_path):
    result = img.copy()
    for c in small_contours:
        cv2.drawContours(result, [c], -1, (0, 255, 0), 2)  
    for c in large_contours:
        cv2.drawContours(result, [c], -1, (0, 0, 255), 2)
    cv2.imwrite(output_path, result)
    


if __name__ == "__main__":
    image_path = "data/processed/segmented_image.jpg"
    output_path = "data/processed/image1_debug_contours.jpg"

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    small, large = detect_wells_from_image(img)

    print(f"Total detected: {len(small) + len(large)}")
    print(f"Small wells: {len(small)}")
    print(f"Big wells: {len(large)}")

    draw_well_groups(img, small, large, output_path)

    cv2.imshow("Detected Wells", cv2.imread(output_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

