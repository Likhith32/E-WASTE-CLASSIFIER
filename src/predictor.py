import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

class EWastePredictor:
    def __init__(self, model_path):
        """Initialize the predictor with a trained model"""
        self.model = tf.keras.models.load_model(model_path)
        # E-waste categories
        self.class_names = ['battery', 'keyboards', 'microwave', 'mobiles', 'mouse', 'pcb']
        
    def predict_image(self, image_path, confidence_threshold=0.5):
        """Predict the type of e-waste in the image"""
        try:
            # Load and preprocess image
            img = image.load_img(image_path, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            predicted_class = self.class_names[predicted_class_index]
            
            # All e-waste items are considered e-waste
            is_ewaste = True
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                {
                    'class': self.class_names[i],
                    'confidence': float(predictions[0][i])
                }
                for i in top_3_indices
            ]
            
            return {
                'is_ewaste': is_ewaste,
                'ewaste_type': predicted_class,
                'confidence': float(confidence),
                'predicted_class': predicted_class,
                'top_3_predictions': top_3_predictions,
                'all_probabilities': {
                    name: float(prob) for name, prob in zip(self.class_names, predictions[0])
                }
            }
            
        except Exception as e:
            return {'error': str(e)}