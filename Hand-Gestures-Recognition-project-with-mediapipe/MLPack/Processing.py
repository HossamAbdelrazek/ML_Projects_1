import numpy as np
import pandas as pd
from scipy import stats
def normalize_hand_landmarks(landmarks):
    """Normalize hand landmarks while preserving the exact feature structure expected by the model."""
    landmarks = np.array(landmarks)

    # Center around (x1, y1)
    x1, y1 = landmarks[0, 0], landmarks[0, 1]
    
    # Subtract x1, y1 from all x and y coordinates
    landmarks[:, 0] -= x1
    landmarks[:, 1] -= y1

    # Normalize by distance to y13
    y13 = landmarks[12, 1]
    landmarks[:, 0] /= y13
    landmarks[:, 1] /= y13
    
    # Convert to a flat array but keep track of the original structure
    flattened_data = {}
    
    # Create named features in the format expected by the model
    for i in range(21):  # 21 landmarks
        flattened_data[f'x{i+1}'] = landmarks[i, 0]
        flattened_data[f'y{i+1}'] = landmarks[i, 1]
        flattened_data[f'z{i+1}'] = landmarks[i, 2]
    
    # The error suggests we might need to exclude 'x1' but keep 'y1'
    # Remove 'x1' from the dictionary
    flattened_data.pop('x1', None)
    flattened_data.pop('x2', None)

    
    return flattened_data

def predict_gesture(model, frames, encoder):
    """Predict gesture based on k frames and return the gesture and confidence."""
    
    # Apply normalization to each frame and get named features
    processed_frames = [normalize_hand_landmarks(frame) for frame in frames]
    
    # Average the features across frames
    averaged_features = {}
    for feature_name in processed_frames[0].keys():
        averaged_features[feature_name] = np.mean([frame[feature_name] for frame in processed_frames])
    
    # Convert to DataFrame with the exact feature names expected by the model
    input_data = pd.DataFrame([averaged_features])
    
    # Get predictions and probabilities
    predictions = model.predict(input_data)
    class_probabilities = model.predict_proba(input_data)
    
    # Get the predicted class and confidence
    prediction = predictions[0]
    prediction_index = np.argmax(class_probabilities)
    confidence = class_probabilities[0, prediction_index]
    
    # Decode the prediction
    decoded_prediction = encoder.inverse_transform([prediction])[0]

    return decoded_prediction, confidence