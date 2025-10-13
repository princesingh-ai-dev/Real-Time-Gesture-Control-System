# Real-time Hand Gesture Control and Command System

A hands-free computer control system that uses hand gestures captured via webcam to execute predefined keyboard and mouse commands.

## Features

- Real-time hand detection and gesture recognition
- Configurable gesture-to-command mappings
- Support for 5 predefined gestures: open_palm, closed_fist, thumbs_up, pointing_index, peace_sign
- Visual feedback with gesture confidence display
- Low latency (< 100ms) performance

## Requirements

- Python 3.8+
- Webcam
- Windows/Linux/macOS

## Installation

1. Clone or download this repository.

2. Create a virtual environment:
   ```bash
   python -m venv gesture_env
   ```

3. Activate the virtual environment:
   - Windows: `gesture_env\Scripts\activate`
   - Linux/macOS: `source gesture_env/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Train the Gesture Model

Before running the main application, you need to train the gesture recognition model:

```bash
python train_model.py
```

This will:
- Open your webcam
- Prompt you to collect samples for each gesture
- Train an SVM classifier on the collected data
- Save the trained model as `gesture_model.pkl`

Follow the on-screen instructions to collect approximately 100 samples per gesture.

### Step 2: Configure Gestures (Optional)

Edit `config.json` to customize gesture-to-command mappings. The default mappings are:

- `open_palm`: Volume Up
- `closed_fist`: Volume Down
- `thumbs_up`: Volume Mute
- `pointing_index`: Page Down
- `peace_sign`: Browser Back

You can modify the `gestures` object to change mappings and add new commands in the `commands` section.

### Step 3: Run the Gesture Control System

```bash
python main.py
```

This will:
- Open your webcam
- Display real-time gesture recognition
- Execute commands based on detected gestures
- Show confidence levels and current gesture

Press 'q' to quit the application.

## Configuration

The `config.json` file contains:

- `gestures`: Maps gesture names to command keys
- `commands`: Defines available commands (key presses or hotkeys)
- `confidence_threshold`: Minimum confidence required for gesture recognition (default: 0.8)

## Troubleshooting

### Model Training Issues
- Ensure good lighting and clear hand visibility
- Collect diverse samples for each gesture
- If accuracy is below 90%, collect more samples

### Runtime Issues
- Check that your webcam is not used by other applications
- Ensure all dependencies are installed correctly
- Adjust confidence threshold in config.json if gestures are not detected reliably

### Performance Issues
- Close other resource-intensive applications
- Use a better webcam if available
- Adjust MediaPipe detection confidence if needed

## Dependencies

- opencv-python: Computer vision and webcam handling
- mediapipe: Hand tracking and landmark detection
- scikit-learn: Machine learning classification
- pyautogui: System command simulation
- numpy: Numerical computations

## License

This project is open-source. Feel free to modify and distribute.