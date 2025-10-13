import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

GESTURES = list(config['gestures'].keys())
DATASET_SIZE = 100  # Number of samples per gesture

def collect_landmarks(gesture, num_samples):
    """Collect landmark data for a specific gesture."""
    print(f"Collecting data for {gesture}. Press 'c' to capture, 'q' to quit.")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)

    landmarks_data = []
    samples_collected = 0

    while samples_collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            landmark_list = []
            for lm in landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            landmarks_data.append(landmark_list)
            samples_collected += 1
            cv2.putText(frame, f'Samples: {samples_collected}/{num_samples}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f'Gesture: {gesture}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Data Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return landmarks_data

def main():
    X = []
    y = []

    for gesture in GESTURES:
        print(f"Collecting data for {gesture}...")
        landmarks = collect_landmarks(gesture, DATASET_SIZE)
        X.extend(landmarks)
        y.extend([gesture] * len(landmarks))

    X = np.array(X)
    y = np.array(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM model
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(".2%")

    if accuracy >= 0.9:
        # Save model
        with open('gesture_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("Model saved as gesture_model.pkl")
    else:
        print("Model accuracy below 90%. Consider collecting more data or adjusting parameters.")

if __name__ == "__main__":
    main()