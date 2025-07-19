import cv2
import numpy as np


def grabcut(image_path):
    img = cv2.imread(image_path)
    img_copy = img.copy()

    # Adjust if necessary!
    rect = (100, 100, img.shape[1] - 100, img.shape[0] - 100)

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    img_segmented = img * mask2[:, :, np.newaxis]
    return img_segmented


if __name__ == "__main__":
    def show_resized(title, img, scale=0.4):
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        resized_img = cv2.resize(img, (width, height))
        cv2.imshow(title, resized_img)
    
    
    img = grabcut("data/raw/image1.jpg")
    show_resized("Segmented Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
