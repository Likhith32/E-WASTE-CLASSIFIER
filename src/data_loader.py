from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from collections import Counter

def load_data(train_dir, test_dir):
    """Load and preprocess training, validation, and test data"""
    datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Test data generator (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        class_mode='categorical',
        batch_size=32,
        subset='training',
        shuffle=True
    )
    
    val_data = datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        class_mode='categorical',
        batch_size=32,
        subset='validation',
        shuffle=False
    )
    
    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(64, 64),
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    
    # Calculate class weights for imbalanced datasets
    class_counts = Counter(train_data.classes)
    total_samples = sum(class_counts.values())
    class_weights = {
        class_id: total_samples / (len(class_counts) * count)
        for class_id, count in class_counts.items()
    }
    
    print(f"Classes found: {train_data.class_indices}")
    print(f"Class weights: {class_weights}")
    
    return train_data, val_data, test_data, class_weights