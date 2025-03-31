########################################
# Author: Utkarsh P Srivastava
# Description: Augmenting the HAM10000 dataset
# for skin lesion classification
# for balancing the dataset.
#License: MIT
# Date: 2023-10-01
########################################

# Import necessary libraries
from numpy.random import seed
seed(101)  # Set seed for reproducibility
import tensorflow
tensorflow.random.set_seed(101)  # Set seed for TensorFlow operations

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
from sklearn.model_selection import train_test_split

# Create base directory for dataset
base_dir = 'base_dir'
os.mkdir(base_dir)

# Create directories for training and validation sets
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)

# Define class labels (skin conditions)
classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# Create subdirectories for each class in train and val directories
for c in classes:
    os.mkdir(os.path.join(train_dir, c))
    os.mkdir(os.path.join(val_dir, c))

# Load metadata CSV containing image details
df_data = pd.read_csv('archive/HAM10000_metadata.csv')

# Identify unique lesion IDs with only one associated image
df = df_data.groupby('lesion_id').count()
df = df[df['image_id'] == 1]
df.reset_index(inplace=True)

# Function to mark images that have duplicate lesion IDs
def identify_duplicates(x):
    unique_list = list(df['lesion_id'])
    return 'no_duplicates' if x in unique_list else 'has_duplicates'

df_data['duplicates'] = df_data['lesion_id'].apply(identify_duplicates)

# Filter out images with unique lesion IDs for validation set
df_val = df_data[df_data['duplicates'] == 'no_duplicates']

# Split validation set from unique images (17% of the dataset)
y = df_val['dx']
_, df_val = train_test_split(df_val, test_size=0.17, random_state=101, stratify=y)

# Mark images as belonging to train or validation set
def identify_val_rows(x):
    val_list = list(df_val['image_id'])
    return 'val' if str(x) in val_list else 'train'

df_data['train_or_val'] = df_data['image_id'].apply(identify_val_rows)

# Extract training images
df_train = df_data[df_data['train_or_val'] == 'train']

# Set image_id as index for easy lookup
df_data.set_index('image_id', inplace=True)

# Define paths to original image folders
folder_1 = os.listdir('archive/ham10000_images_part_1')
folder_2 = os.listdir('archive/ham10000_images_part_2')

# Lists of train and validation images
train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

# Function to move images to appropriate directories
def move_images(image_list, source_folders, dest_dir):
    for image in image_list:
        fname = image + '.jpg'
        label = df_data.loc[image, 'dx']  # Get class label
        
        for folder in source_folders:
            if fname in folder:
                src = os.path.join(f'archive/{folder}', fname)
                dst = os.path.join(dest_dir, label, fname)
                shutil.copyfile(src, dst)

# Transfer training and validation images
move_images(train_list, [folder_1, folder_2], train_dir)
move_images(val_list, [folder_1, folder_2], val_dir)

# Data Augmentation (not applied to 'nv' class)
class_list = ['mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

for item in class_list:
    aug_dir = 'aug_dir'
    os.mkdir(aug_dir)
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)

    # Copy images from the original class directory to a temporary directory
    img_list = os.listdir(os.path.join(train_dir, item))
    for fname in img_list:
        shutil.copyfile(os.path.join(train_dir, item, fname), os.path.join(img_dir, fname))

    # Set up image augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    batch_size = 50
    save_path = os.path.join(train_dir, item)

    aug_datagen = datagen.flow_from_directory(
        aug_dir,
        save_to_dir=save_path,
        save_format='jpg',
        target_size=(224, 224),
        batch_size=batch_size)

    # Generate augmented images until desired count is reached
    num_aug_images_wanted = 6000
    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted - num_files) / batch_size))
    
    for _ in range(num_batches):
        next(aug_datagen)
    
    # Remove temporary augmentation directory
    shutil.rmtree(aug_dir)

print("Dataset preparation complete.")
