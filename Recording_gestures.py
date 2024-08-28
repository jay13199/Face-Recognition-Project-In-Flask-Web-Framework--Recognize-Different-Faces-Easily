import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to save landmarks with a gesture label
def save_landmarks(landmarks, gesture_name):
    with open('gesture_data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([gesture_name] + [coord for point in landmarks for coord in point])

gesture_name = ""   

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
            
            # Save the landmarks when a gesture is labeled
            if gesture_name:
                save_landmarks(landmarks, gesture_name)
                gesture_name = ""  # Reset after saving

    # Display the frame
    cv2.imshow('Hand Gesture Tracking', frame)

    # Capture keyboard input
    key = cv2.waitKey(5) & 0xFF

    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('1'):  # Press '1' for finished
        gesture_name = "finished"
    elif key == ord('2'):  # Press '2' for open
        gesture_name = "heart"
    elif key == ord('3'):  # Press '2' for open
        gesture_name = "thumbs-up"
    elif key == ord('4'):  # Press '2' for open
        gesture_name = "thumbs-down"
    # Add more elif blocks for other gestures

cap.release()
cv2.destroyAllWindows()
