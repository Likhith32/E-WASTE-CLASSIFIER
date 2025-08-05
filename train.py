import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.data_loader import load_data
from src.model_builder import build_e_waste_model

def train_model():
    """Train the e-waste classification model"""
    # Set up directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    train_dir = os.path.join(base_dir, "data", "modified-dataset", "train")
    test_dir = os.path.join(base_dir, "data", "modified-dataset", "test")
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    print("Loading data...")
    train_data, val_data, test_data, class_weights = load_data(train_dir, test_dir)
    
    print("Building model...")
    model = build_e_waste_model(num_classes=train_data.num_classes)
    
    print("Model summary:")
    model.summary()
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_e_waste_model.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    print("Training model...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=50,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(model_dir, "e_waste_model.h5"))
    
    print("✅ Model trained and saved successfully!")
    
    return history, model

if __name__ == "__main__":
    try:
        history, model = train_model()
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")