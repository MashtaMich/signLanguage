import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os

def pickle_images_labels():
    images_labels = []
    # Get all jpg images in subfolders of "gestures"
    image_paths = glob(os.path.join("gestures", "*", "*.jpg"))
    image_paths.sort()  # sort for consistency
    for image_path in image_paths:
        print("Processing:", image_path)
        # Extract label from folder name
        # Assuming structure "gestures/<label>/<image_filename>.jpg"
        label = os.path.basename(os.path.dirname(image_path))
        # Read image in grayscale mode
        img = cv2.imread(image_path, 0)
        if img is None:
            continue
        images_labels.append((np.array(img, dtype=np.uint8), int(label)))
    return images_labels

# Load the images and labels
images_labels = pickle_images_labels()
# Shuffle the dataset (for reproducibility, you can set a seed)
images_labels = shuffle(images_labels, random_state=42)

# Unzip images and labels
images, labels = zip(*images_labels)
total = len(images)
print("Total images:", total)

# Split dataset into train, validation and test sets (70% / 15% / 15%)
train_count = int(0.7 * total)
val_count = int(0.15 * total)
test_count = total - train_count - val_count

train_data = (images[:train_count], labels[:train_count])
val_data = (images[train_count:train_count + val_count], labels[train_count:train_count + val_count])
test_data = (images[train_count + val_count:], labels[train_count + val_count:])

print("Train samples:", len(train_data[0]))
print("Validation samples:", len(val_data[0]))
print("Test samples:", len(test_data[0]))

# Save each set into its own pickle file
with open("train2.pickle", "wb") as f:
    pickle.dump(train_data, f)
with open("val2.pickle", "wb") as f:
    pickle.dump(val_data, f)
with open("test2.pickle", "wb") as f:
    pickle.dump(test_data, f)