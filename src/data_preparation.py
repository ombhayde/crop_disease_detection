# src/data_preparation.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm.auto import tqdm  # Changed this import

def process_dataset(raw_data_dir='data/raw/PlantVillage/color', 
                   processed_dir='data/processed',
                   img_size=(224, 224)):
    """
    Process raw dataset: resize images and split into train/val/test sets
    """
    print(f"Starting to process dataset from {raw_data_dir}")
    
    # Create processed directories
    os.makedirs(f'{processed_dir}/train', exist_ok=True)
    os.makedirs(f'{processed_dir}/val', exist_ok=True)
    os.makedirs(f'{processed_dir}/test', exist_ok=True)
    
    # Get all class folders
    class_folders = [f for f in os.listdir(raw_data_dir) 
                     if os.path.isdir(os.path.join(raw_data_dir, f))]
    
    print(f"Found {len(class_folders)} class folders")
    
    for class_folder in tqdm(class_folders, desc="Processing classes"):
        print(f"Processing class: {class_folder}")
        
        # Create class directories in train/val/test
        os.makedirs(f'{processed_dir}/train/{class_folder}', exist_ok=True)
        os.makedirs(f'{processed_dir}/val/{class_folder}', exist_ok=True)
        os.makedirs(f'{processed_dir}/test/{class_folder}', exist_ok=True)
        
        # Get all images in the class folder
        img_paths = [os.path.join(raw_data_dir, class_folder, img) 
                    for img in os.listdir(os.path.join(raw_data_dir, class_folder))
                    if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  - Found {len(img_paths)} images")
        
        # Split into train (70%), validation (15%), and test (15%)
        train_paths, test_val_paths = train_test_split(img_paths, test_size=0.3, random_state=42)
        val_paths, test_paths = train_test_split(test_val_paths, test_size=0.5, random_state=42)
        
        print(f"  - Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
        
        # Process and save images
        for subset, paths in [
            ('train', train_paths),
            ('val', val_paths),
            ('test', test_paths)
        ]:
            for i, path in enumerate(paths):
                try:
                    if i % 10 == 0:  # Show progress every 10 images
                        print(f"  - Processing {subset}: {i}/{len(paths)}")
                        
                    img = cv2.imread(path)
                    if img is None:
                        print(f"Warning: Could not read image {path}")
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                    
                    # Save the processed image
                    filename = os.path.basename(path)
                    output_path = f'{processed_dir}/{subset}/{class_folder}/{filename}'
                    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")
    
    print("Dataset processing completed.")
def create_data_generators(batch_size=32):
    """
    Create data generators for training, validation, and testing
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescale validation and test data
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        'data/processed/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    valid_generator = valid_datagen.flow_from_directory(
        'data/processed/val',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        'data/processed/test',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, valid_generator, test_generator

if __name__ == "__main__":
    process_dataset()