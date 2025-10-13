# Real-time Hand Gesture Control and Command System: Project Plan and Technical Specification

## 1. Project Goal

The primary objective of this project is to develop a hands-free, real-time interaction system for operating a laptop, desktop, or PC using distinct, user-defined hand gestures captured via a standard webcam. The system will translate these gestures into predefined keyboard and mouse commands, enabling intuitive control of the operating system and applications without physical input devices.

## 2. Core Technology Stack

The following technologies are recommended for their robustness, ease of integration, and alignment with the project's requirements:

- **Programming Language: Python**  
  Chosen for its extensive ecosystem of libraries for computer vision, machine learning, and system automation. Python's simplicity and cross-platform compatibility make it ideal for rapid prototyping and deployment on various operating systems.

- **Computer Vision/Gesture Recognition Library: OpenCV and MediaPipe**  
  - **OpenCV**: Utilized for capturing and processing the webcam feed, providing efficient frame handling and image manipulation capabilities.  
  - **MediaPipe (Hand Tracking module)**: Employed for real-time hand detection and landmark extraction. MediaPipe offers fast, accurate skeleton-point detection with 21 key landmarks (x, y, z coordinates), reducing development time compared to custom implementations.

- **Machine Learning/Classification: scikit-learn or Keras/TensorFlow**  
  - For simplicity and speed, **scikit-learn** can be used with algorithms like Support Vector Machines (SVM) or Random Forests for gesture classification.  
  - For more advanced needs, **Keras/TensorFlow** enables building a lightweight Feed-Forward Neural Network to classify gestures based on normalized landmark data, offering scalability if the system expands to more gestures.

- **Operating System Interaction: pyautogui or pynput**  
  - **pyautogui**: Selected for its straightforward API to simulate keyboard and mouse events across platforms, facilitating command execution based on gesture mappings.

## 3. Key System Features

The system must incorporate the following capabilities to ensure reliable, user-friendly operation:

- **Real-time Hand Detection**: Continuously locate and track a single hand in the live webcam feed with latency under 100 ms, leveraging MediaPipe's optimized pipeline for smooth performance.

- **Gesture Classification**: Accurately classify at least five distinct, predefined static hand gestures using the trained ML model. Target gestures include:
  - Peace sign
  - Open palm
  - Closed fist
  - Thumbs up
  - Pointing index finger

- **Dynamic Command Mapping**: Support a configuration file (e.g., JSON) for easy customization, allowing users to map each gesture to specific commands without code changes.

- **Example Commands**:
  1. **System Control**: 'Open Palm' → Volume Up; 'Closed Fist' → Volume Down; 'Thumbs Up' → Mute/Unmute.
  2. **Web Browsing**: 'Pointing Index' → Scroll Down; 'Peace Sign' → Go Back.
  3. **Application Control**: 'OK Sign' (if added) → Play/Pause Media.

- **Visual Feedback**: Provide an overlay on the camera window or screen displaying the detected gesture, confidence level (e.g., percentage), and the command being executed, enhancing user awareness and system transparency.

## 4. Detailed Development Phases

The project will follow a structured, four-phase development lifecycle to ensure systematic progress:

1. **Phase 1: Setup & Data Acquisition**  
   - Set up a Python virtual environment and install required libraries (OpenCV, MediaPipe, scikit-learn/Keras, pyautogui).  
   - Implement webcam feed capture using OpenCV to display and process frames.  
   - Integrate MediaPipe Hand Tracking to extract 21 landmark coordinates (x, y, z) from detected hands in real-time.

2. **Phase 2: Gesture Training**  
   - Collect a dataset of landmark coordinates for each of the five target gestures by recording multiple samples per gesture.  
   - Normalize landmark data and design/train a classification model (e.g., SVM via scikit-learn or a simple neural network via Keras).  
   - Validate the model to achieve ≥90% accuracy on test data, tuning hyperparameters as needed.

3. **Phase 3: Integration & Command Engine**  
   - Integrate the trained model into the OpenCV real-time stream for live gesture prediction.  
   - Develop the Command Mapping Engine to load configurations from the JSON file and use pyautogui to execute mapped commands.  
   - Implement gesture smoothing to reduce false positives and ensure stable command triggering.

4. **Phase 4: User Interface & Refinement**  
   - Add visual feedback overlay using OpenCV drawing functions.  
   - Optimize the pipeline for performance (e.g., frame rate, memory usage) and stability.  
   - Package the application, create documentation, and test across different environments.

## 5. Final Deliverables

The project will produce the following outputs:

- **Python Script (`main.py`)**: Contains the main execution logic for real-time gesture detection, classification, and command execution.
- **Jupyter Notebook or Training Script (`train_model.py` or `gesture_training.ipynb`)**: Demonstrates model training, evaluation, and validation for gesture classification.
- **Configuration File (`config.json`)**: Defines gesture-to-command mappings in a user-editable format.
- **README.md**: Provides installation instructions, usage guide, and troubleshooting information.
- **Requirements File (`requirements.txt`)**: Lists all Python dependencies for easy environment setup.

## Minimal Pseudocode Example for Real-time Gesture Loop (Phase 3)

```python
import cv2
import mediapipe as mp
import numpy as np
from sklearn.svm import SVC  # or equivalent model
import pyautogui
import json

# Load trained model and config
model = SVC()  # Assume pre-trained and loaded
config = json.load(open('config.json'))

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        # Extract and normalize landmark coordinates (21 points)
        landmark_list = []
        for lm in landmarks.landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])  # Normalize if needed

        # Predict gesture
        gesture = model.predict([landmark_list])[0]
        confidence = model.predict_proba([landmark_list])[0].max()  # If available

        # Execute command if confidence > threshold
        if confidence > 0.8 and gesture in config:
            pyautogui.press(config[gesture])  # or appropriate action

        # Draw landmarks and feedback
        mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f'Gesture: {gesture} ({confidence:.2f})', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()