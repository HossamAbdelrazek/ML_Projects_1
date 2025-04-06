import cv2
import mediapipe as mp
import numpy as np
import joblib
from MLPack.Processing import predict_gesture
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Load model
model = joblib.load("/media/hossam/01DB4F3124760DF0/Users/Hossam Abdelrazek/Desktop/ITI AI and ML/AI ITI Material/Machine Learning I Supervised/Supervised_ML_Project/Hand-Gestures-Recognition-project-with-mediapipe/MLPack/model.joblib")
encoder = joblib.load("Hand-Gestures-Recognition-project-with-mediapipe/MLPack/encoder.joblib")# Constants

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    # Constants
    K = 5  # Number of frames to accumulate for prediction
    fps = cv2.CAP_PROP_FPS
    cap = cv2.VideoCapture(0)

    # Video Writer setup to record video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video
    out = cv2.VideoWriter('/media/hossam/01DB4F3124760DF0/Users/Hossam Abdelrazek/Desktop/ITI AI and ML/AI ITI Material/Machine Learning I Supervised/Supervised_ML_Project/Hand-Gestures-Recognition-project-with-mediapipe/Personal/video/hand_gesture_output.avi', fourcc, 10, (640, 480))  # Save video to 'hand_gesture_output.avi'

    # Initialize variables
    frame_buffer = []  # To store k frames (landmarks)
    current_prediction = None
    current_confidence = 0.0
    cv2.namedWindow("Hand Gesture Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Gesture Recognition", 1280, 720)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as hands:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame (mirror view)
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks
            results = hands.process(image)

            landmarks = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])  # Collect landmarks as a list of [x, y, z]

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

                # Append the current frame's landmarks to the buffer
                frame_buffer.append(landmarks)

                # If we have enough frames, process them for prediction
                if len(frame_buffer) == K:
                    current_prediction, current_confidence = predict_gesture(model, frame_buffer, encoder)
                    # Display the prediction and confidence
                    text = f"Gesture: {current_prediction} ({current_confidence*100:.1f}%)"
                    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    frame_buffer.pop(0)
                
                else:
                    text = " "
                    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Show frame
            h, w = frame.shape[0:2]
            c = h/w
            wid = 1920*c
            frame2 = cv2.resize(frame, (1920, int(wid)))
            cv2.imshow("Hand Gesture Recognition", frame2)

            # Save frame to video
            out.write(frame)

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
