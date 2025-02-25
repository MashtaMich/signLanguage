import cv2
import numpy as np
import os

def resize_to_match(img, target_height):
    """Resize image to match target height while keeping aspect ratio."""
    if img is None:
        print("Error: Image is None, cannot resize.")
        return None
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height))

# Load image (ensure correct path)
image_path = r"gestures\0\0.jpg"  # Change this to an existing image
if not os.path.exists(image_path):
    print(f"Error: {image_path} does not exist.")
else:
    img1 = cv2.imread(image_path, 0)
    if img1 is None:
        print(f"Error: Failed to load {image_path}")
    else:
        target_height = 480
        img1 = resize_to_match(img1, target_height)
