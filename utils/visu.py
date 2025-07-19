import cv2

def show_resized(title, img, max_width=800, max_height=600):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(title, resized)
