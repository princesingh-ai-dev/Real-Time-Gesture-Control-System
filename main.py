import cv2
import mediapipe as mp
import numpy as np
import pickle
import json
import pyautogui
import time

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

CONFIDENCE_THRESHOLD = config['confidence_threshold']

# Load trained model
try:
    with open('gesture_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Model file 'gesture_model.pkl' not found. Please run train_model.py first.")
    exit()

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Gesture smoothing
last_gesture = None
last_execution_time = 0
EXECUTION_COOLDOWN = 1.0  # seconds

def execute_command(gesture):
    """Execute the command associated with the gesture."""
    global last_execution_time
    current_time = time.time()

    if current_time - last_execution_time < EXECUTION_COOLDOWN:
        return

    if gesture in config['gestures']:
        command_key = config['gestures'][gesture]
        if command_key in config['commands']:
            command = config['commands'][command_key]
            if command['type'] == 'key':
                pyautogui.press(command['key'])
            elif command['type'] == 'hotkey':
                pyautogui.hotkey(*command['keys'])
            print(f"Executed command: {command_key}")
            last_execution_time = current_time

def main():
    global last_gesture

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time gesture control. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip and convert frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = hands.process(rgb_frame)

        current_gesture = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]

            # Extract landmark coordinates
            landmark_list = []
            for lm in landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            # Predict gesture
            try:
                probabilities = model.predict_proba([landmark_list])[0]
                confidence = max(probabilities)
                gesture_index = np.argmax(probabilities)
                current_gesture = model.classes_[gesture_index]

                if confidence >= CONFIDENCE_THRESHOLD:
                    if current_gesture != last_gesture:
                        execute_command(current_gesture)
                        last_gesture = current_gesture
                else:
                    last_gesture = None
            except:
                pass

            # Draw landmarks
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Display gesture and confidence
        status_text = f"Gesture: {current_gesture or 'None'} ({confidence:.2f})"
        cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Real-time Hand Gesture Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()