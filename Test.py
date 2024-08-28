import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import time

# Load the trained model and label encoder classes
model = tf.keras.models.load_model('gesture_classifier.h5')
gesture_classes = np.load('gesture_classes.npy', allow_pickle=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Timer variables to ignore gestures for a specific duration
last_heart_time = 0
last_finished_time = 0
last_thumbsup_time = 0
last_thumbsdown_time = 0
ignore_duration = 3  # seconds

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collecting the landmarks for classification
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # Flatten the landmarks list and make prediction
            landmarks = np.array(landmarks).flatten().reshape(1, -1)
            prediction = model.predict(landmarks)
            class_idx = np.argmax(prediction)
            gesture_name = gesture_classes[class_idx]

            # Display the predicted gesture
            cv2.putText(frame, f'Gesture: {gesture_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            current_time = time.time()

            # Check if "heart" gesture is detected
            if gesture_name.lower() == 'heart':
                if current_time - last_heart_time > ignore_duration:
                    engine.say("Hi")
                    engine.runAndWait()
                    last_heart_time = current_time  # Update the last "heart" gesture time

            # Check if "finished" gesture is detected
            elif gesture_name.lower() == 'finished':
                if current_time - last_finished_time > ignore_duration:
                    engine.say("Bye, see you next time")
                    engine.runAndWait()
                    last_finished_time = current_time  # Update the last "finished" gesture time

            # Check if "thumbs-up" gesture is detected
            elif gesture_name.lower() == 'thumbs-up':
                if current_time - last_thumbsup_time > ignore_duration:
                    engine.say("Well, what's next?")
                    engine.runAndWait()
                    last_thumbsup_time = current_time  # Update the last "thumbs-up" gesture time

            # Check if "thumbs-down" gesture is detected
            elif gesture_name.lower() == 'thumbs-down':
                if current_time - last_thumbsdown_time > ignore_duration:
                    engine.say("Tell me, where to improve")
                    engine.runAndWait()
                    last_thumbsdown_time = current_time  # Update the last "thumbs-down" gesture time

    # Display the frame
    cv2.imshow('Hand Gesture Tracking', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
