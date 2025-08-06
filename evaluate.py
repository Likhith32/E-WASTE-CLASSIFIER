import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from src.data_loader import load_data
from src.predictor import EWastePredictor
import tensorflow as tf

def retrain_and_save_model(train_dir, model_path):
    """Train a simple CNN model and save it to .keras format"""
    print("📦 Training new model since none was found...")
    train_data, val_data, _, class_names = load_data(train_dir, val_dir=None)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, epochs=5, validation_data=val_data)
    model.save(model_path)
    print(f"✅ New model trained and saved at {model_path}")

def evaluate_model():
    """Evaluate the trained model on test data"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "e_waste_model.keras")
    
    train_dir = os.path.join(base_dir, "data", "modified-dataset", "train")
    test_dir = os.path.join(base_dir, "data", "modified-dataset", "test")
    
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        retrain_and_save_model(train_dir, model_path)
    
    # Load test data
    _, _, test_data, _ = load_data(train_dir, test_dir)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    print("📊 Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
    
    print(f"\n✅ Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    y_pred = model.predict(test_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_data.classes
    class_names = list(test_data.class_indices.keys())
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'))
    plt.show()
    
    return test_accuracy

def test_single_image(image_path):
    """Test the model on a single image"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "e_waste_model.keras")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found at {image_path}")
        return
    
    predictor = EWastePredictor(model_path)
    result = predictor.predict_image(image_path)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\n🔍 Prediction Results for: {os.path.basename(image_path)}")
    print(f"E-Waste Type: {result['ewaste_type']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nTop 3 Predictions:")
    for i, pred in enumerate(result['top_3_predictions'], 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']:.2%}")
    print(f"\nAll Probabilities: {result['all_probabilities']}")

if __name__ == "__main__":
    try:
        accuracy = evaluate_model()
        print(f"\n✅ Evaluation completed. Final accuracy: {accuracy:.2%}")
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")

    # To test a single image:
    # test_single_image("path/to/image.jpg")
