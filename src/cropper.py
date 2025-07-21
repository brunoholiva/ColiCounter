import cv2
import numpy as np

def segment_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    rect = (100, 100, img.shape[1] - 100, img.shape[0] - 100)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = img * mask2[:, :, np.newaxis]

    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found!")

    largest_contour = max(contours, key=cv2.contourArea)

    new_mask = np.zeros_like(gray)
    cv2.drawContours(new_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    result = cv2.bitwise_and(img, img, mask=new_mask)

    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = result[y:y+h, x:x+w]

    return cropped

if __name__ == "__main__":
    file_path = "data/raw/image1.jpg"
    segmented_image = segment_plate(file_path)
    cv2.imwrite("data/processed/segmented_image.jpg", segmented_image)