import cv2
import numpy as np
from utils.visu import show_resized

def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed

def detect_wells(mask, original_img, top_fraction=0.2, min_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h = original_img.shape[0]
    top_limit = h * top_fraction

    top_contours = []
    final_contours = []

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        _, y, _, _ = cv2.boundingRect(c)
        (top_contours if y < top_limit else final_contours).append(c)

    if top_contours:
        combined = np.vstack(top_contours)
        final_contours.append(combined)

    img_with_contours = original_img.copy()
    for c in final_contours:
        cv2.drawContours(img_with_contours, [c], -1, (0, 255, 0), 2)

    show_resized("Detected Wells", img_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final_contours
