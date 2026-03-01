import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class PredictionPipelineConfig:
    model_registry_path: str = os.path.join('artifacts', 'model_registry', 'registry_metadata.json')

class PredictionPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig()
        self.model = None
        self.preprocessor = None
        self.model_version = None
        self.load_latest_model()

    def load_latest_model(self):
        """
        Automatically load the newest trained model from registry.
        """
        try:
            logging.info("Loading latest model from registry for prediction...")
            
            if not os.path.exists(self.config.model_registry_path):
                raise FileNotFoundError(f"Registry metadata not found at {self.config.model_registry_path}")
                
            with open(self.config.model_registry_path, 'r') as f:
                metadata = json.load(f)
                
            if not metadata:
                raise ValueError("No models found in registry")
                
            # Sort by timestamp (or just take last appended if logic holds)
            latest_entry = metadata[-1]
            self.model_version = latest_entry['version_id']
            model_path = latest_entry['paths']['model']
            preprocessor_path = latest_entry['paths']['preprocessor']
            
            logging.info(f"Loading Model Version: {self.model_version}")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
                
            logging.info("Model and Preprocessor loaded successfully.")
            
        except Exception as e:
            raise CustomException(e, sys)

    def validate_input_features(self, features):
        """
        Ensure incoming data matches expected format.
        Expected: [delta, theta, alpha, beta, gamma]
        """
        try:
            if not isinstance(features, (list, np.ndarray)):
                 raise TypeError("Input must be a list or numpy array")
            
            features_list = list(features)
            
            if len(features_list) != 5:
                raise ValueError(f"Expected 5 features (Delta, Theta, Alpha, Beta, Gamma), got {len(features_list)}")
            
            # Check for numeric types
            if not all(isinstance(x, (int, float)) for x in features_list):
                 raise ValueError("All features must be numeric")
                 
            return np.array(features_list).reshape(1, -1)
            
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_input_data(self, input_array):
        """
        Apply saved preprocessing pipeline.
        """
        try:
            # logging.info("Preprocessing input data...")
            transformed_data = self.preprocessor.transform(input_array)
            return transformed_data
        except Exception as e:
            raise CustomException(e, sys)

    def predict_emotion(self, processed_input):
        """
        Generate emotion prediction.
        """
        try:
            prediction = self.model.predict(processed_input)
            
            # Get probability if supported
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(processed_input)
                confidence = np.max(probabilities)
            else:
                confidence = 1.0 # Fallback
                
            return prediction[0], confidence
        except Exception as e:
            raise CustomException(e, sys)

    def decode_prediction_label(self, label_idx):
        """
        Convert numeric label into human-readable emotion.
        0: Angry, 1: Calm, 2: Happy, 3: Sad
        (Based on label encoder mapping from Feature Engineering step: 
        Angry: 0, Calm: 1, Happy: 2, Sad: 3 - Alphabetical order usually if LabelEncoder fits strings)
        """
        try:
            # Hardcoded mapping based on typical LabelEncoder alphabet sort of ['Angry', 'Calm', 'Happy', 'Sad']
            # If changed, need to load encoder or save mapping in metadata.
            mapping = {0: "Angry", 1: "Calm", 2: "Happy", 3: "Sad"}
            return mapping.get(label_idx, "Unknown")
        except Exception as e:
             raise CustomException(e, sys)

    def return_prediction_output(self, emotion, confidence):
        """
        Return structured prediction result.
        """
        try:
            return {
                "predicted_emotion": emotion,
                "confidence_score": round(float(confidence), 4),
                "model_version": self.model_version
            }
        except Exception as e:
             raise CustomException(e, sys)

    def predict_eeg_emotion(self, features):
        """
        Master prediction function.
        """
        try:
            # 1. Validate
            input_array = self.validate_input_features(features)
            
            # 2. Preprocess
            processed_input = self.preprocess_input_data(input_array)
            
            # 3. Predict
            label_idx, confidence = self.predict_emotion(processed_input)
            
            # 4. Decode
            emotion = self.decode_prediction_label(label_idx)
            
            # 5. Return
            result = self.return_prediction_output(emotion, confidence)
            
            # logging.info(f"Prediction: {result}")
            return result
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Test with dummy data
        pipeline = PredictionPipeline()
        dummy_input = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = pipeline.predict_eeg_emotion(dummy_input)
        print(f"\nTest Prediction Result: {json.dumps(result, indent=4)}")
        
    except Exception as e:
        print(e)
