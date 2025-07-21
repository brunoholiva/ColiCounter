# ColiCounter
**ColiCounter** is a project developed as part of an Introduction to Image Processing course. Its main goal is to detect and count positive wells in Colilert Quanti-Trays through classical computer vision techniques.
This project is **NOT intended for clinical or official use**, but as a learning and experimentation tool. Feel free to use, modify, and contribute!

## Features
- Automatic segmentation of the Colilert Quanti-Tray from raw images using GrabCut algorithm.

- Detection and classification of positive wells as "small positives" and "large positives" based on contour area and position within the tray layout.

- Integration of Most Probable Number (MPN) table to estimate bacterial concentration following IDEXX Colilert methodology.

- Batch processing of multiple Quanti-Tray images.

- Visual output with contours drawn around detected positive wells for easy verification.

## Usage

1. Place the raw images in: `data/raw/`
2. Run ``main.py`` with ```python -m src.main```

## Limitations
As this is a tool built using classic computer vision algorithms (thresholding, contour detection, morphological operations), it is heavily limited by factors such as:

- Variations in lighting conditions

- Changes in camera angle or Quanti-Tray positioning

- Image quality, resolution, and noise

- Color distortions and artifacts that affect yellow well detection

A more robust solution would likely involve deep learning models trained on a large dataset of labeled Colilert Quanti-Tray images. That said, this project intentionally avoids those techniques to focus on foundational image processing concepts, making it more transparent and easier to understand for learning purposes.

Also, this tool still misses sometimes, even when everything is perfect.
*Escherichia coli* detection with blue light fluorescence is not yet implemented.

## Data
All images were generously made available by the folks at [LABAC-UEL](https://share.google/xrxR72UWHsQ8GAbIq).