# Hand Gesture Recognition using MediaPipe and Machine Learning

## Overview

This project focuses on real-time hand gesture recognition by combining MediaPipe's hand tracking technology with machine learning classifiers trained on a prerecorded dataset of hand landmark positions.

The goal was to build an accurate, responsive system capable of detecting and classifying hand gestures from live webcam input.

## Project Overview

- Hand Landmark Detection: Used MediaPipe's Hands solution to extract 21 3D landmarks per detected hand from a live video stream.

- Data Source: Worked with a prerecorded dataset of hand landmark positions labeled with corresponding gesture classes.

- Model Training: Trained several classical machine learning models on the extracted landmark features.

- Model Selection: Selected the top-performing models based on evaluation metrics and combined them into a Voting Classifier for optimal performance.

- Real-time Inference: Deployed the final model for real-time hand gesture classification through webcam input.

## Technologies and Libraries

- Python 3.8+
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- OpenCV
- Lightgbm

![alt text](image.png)

Final Model: An ensemble Voting Classifier combining the best models achieved 98.36% accuracy.

Hand-Gestures-Recognition-project-with-mediapipe/
│
├── hand_landmarks_data.csv                     
├── models/                                  
├── MLPack/
│   ├── Processing.py  
│   ├── model.joblib      
│   └── encoder.joblib 
├── README.md                  
├── requirements.txt           
└── Main.py                    