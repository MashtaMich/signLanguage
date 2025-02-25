import cv2
import os
import numpy as np

def rotate_image(image, angle):
    """
    Rotates an image by the given angle (in degrees) around its center.
    Fills border areas with black.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=0)
    return rotated

def augment_images():
    gest_folder = "gestures"
    
    # Loop through each gesture folder (each gesture is in its own subfolder)
    for g_id in os.listdir(gest_folder):
        folder_path = os.path.join(gest_folder, g_id)
        if not os.path.isdir(folder_path):
            continue
        
        # Get list of all .jpg files (assuming originals are named as numbers)
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
        
        # Determine a starting counter for new augmented images.
        # We assume the original images are numbered sequentially.
        if image_files:
            # Extract numeric parts from filenames and take the max.
            try:
                numbers = [int(os.path.splitext(f)[0]) for f in image_files]
                counter = max(numbers)
            except ValueError:
                counter = len(image_files)
        else:
            counter = 0
        
        # For each original image, apply augmentations.
        for img_file in image_files:
            image_path = os.path.join(folder_path, img_file)
            img = cv2.imread(image_path, 0)  # read in grayscale
            if img is None:
                print(f"Could not read {image_path}")
                continue
            
            # Horizontal flip augmentation
            flipped = cv2.flip(img, 1)
            counter += 1
            new_filename = os.path.join(folder_path, f"{counter}.jpg")
            cv2.imwrite(new_filename, flipped)
            
            # Small rotations: -10 and +10 degrees
            for angle in [-10, 10]:
                rotated = rotate_image(img, angle)
                counter += 1
                new_filename = os.path.join(folder_path, f"{counter}.jpg")
                cv2.imwrite(new_filename, rotated)
            
            print(f"Augmented {img_file} in folder {g_id}")

if __name__ == "__main__":
    augment_images()