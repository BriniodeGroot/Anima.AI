import os
import shutil
import numpy as np

# Path to the downloaded dataset
dataset_dir = './images'

# Directories for the training and validation splits
train_dir = './train_dir'
validation_dir = './validation_dir'

# Split ratio
train_ratio = 0.8

# Create the train and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Loop over the directories within the dataset
for breed_dir in os.listdir(dataset_dir):
    breed_path = os.path.join(dataset_dir, breed_dir)

    # Only process directories
    if os.path.isdir(breed_path):
        # Create subdirectories in train and validation directories
        os.makedirs(os.path.join(train_dir, breed_dir), exist_ok=True)
        os.makedirs(os.path.join(validation_dir, breed_dir), exist_ok=True)

        # Get a list of images and shuffle it
        images = os.listdir(breed_path)
        np.random.shuffle(images)

        # Split the images
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Copy images to their respective directories
        for img in train_images:
            shutil.copy(os.path.join(breed_path, img), os.path.join(train_dir, breed_dir))
        for img in val_images:
            shutil.copy(os.path.join(breed_path, img), os.path.join(validation_dir, breed_dir))
