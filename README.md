# ✋ Real-Time Gesture Control System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-4285F4?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Control your computer hands-free using gesture recognition — built with MediaPipe, SVM & OpenCV.**

A real-time hand gesture recognition system that maps gestures to keyboard/mouse commands for zero-touch computer control.

</div>

---

## ✨ Features

- 🖐️ **5 Predefined Gestures** — Open palm, closed fist, thumbs up, pointing index, peace sign
- ⚡ **Low Latency** — Sub-100ms gesture recognition for seamless interaction
- 🎯 **High Accuracy** — SVM classifier with configurable confidence threshold (default: 80%)
- 🔧 **Customizable Mappings** — JSON-based gesture-to-command configuration
- 📊 **Visual Feedback** — Real-time confidence display and gesture indicators
- 🎓 **Built-in Training** — Collect your own gesture samples and train the model

## 🎬 How It Works

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐     ┌──────────────┐
│   Webcam     │────▶│ MediaPipe Hands  │────▶│ SVM Classifier│────▶│   Execute    │
│   Feed       │     │ (21 Landmarks)   │     │ (scikit-learn)│     │   Command    │
└──────────────┘     └──────────────────┘     └───────────────┘     └──────────────┘
                              │                        │
                              ▼                        ▼
                     Feature Extraction        Confidence > 80%?
                     (x, y coordinates)        ├── Yes → Execute
                                               └── No  → Ignore
```

## 🎮 Default Gesture Mappings

| Gesture | Command | Description |
|---------|---------|-------------|
| 🖐️ Open Palm | Volume Up | Raise hand to increase volume |
| ✊ Closed Fist | Volume Down | Close fist to decrease volume |
| 👍 Thumbs Up | Mute/Unmute | Toggle audio mute |
| 👆 Pointing Index | Page Down | Scroll down through content |
| ✌️ Peace Sign | Browser Back | Navigate to previous page |

## 📋 Requirements

- Python 3.8+
- Webcam
- Windows / Linux / macOS

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/princesingh1702/Real-Time-Gesture-Control-System.git
cd Real-Time-Gesture-Control-System

# 2. Create virtual environment
python -m venv gesture_env
gesture_env\Scripts\activate      # Windows
# source gesture_env/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the gesture model (first time only)
python train_model.py
# Follow on-screen instructions — collect ~100 samples per gesture

# 5. Run gesture control
python main.py
# Press 'q' to quit
```

## ⚙️ Configuration

Edit `config.json` to customize gesture mappings:

```json
{
  "gestures": {
    "open_palm": "volume_up",
    "closed_fist": "volume_down",
    "thumbs_up": "mute"
  },
  "confidence_threshold": 0.8
}
```

## 📁 Project Structure

```
Real-Time-Gesture-Control-System/
├── main.py            # Main application — webcam + gesture detection
├── train_model.py     # Gesture data collection & SVM training
├── config.json        # Gesture-to-command mappings
├── requirements.txt   # Python dependencies
└── gesture_model.pkl  # Trained model (generated after training)
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Hand Tracking | MediaPipe Hands (21 landmarks) |
| Classification | scikit-learn SVM |
| Computer Vision | OpenCV |
| System Control | PyAutoGUI |
| Math | NumPy |

## 📊 Performance

| Metric | Value |
|--------|-------|
| Recognition Latency | < 100ms |
| Training Accuracy | 95%+ (with 100+ samples/gesture) |
| Supported Gestures | 5 (expandable) |
| Frame Rate | 30 FPS |

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Low accuracy | Collect more diverse samples, ensure good lighting |
| Webcam not detected | Close other apps using the camera, check device index |
| Commands not executing | Run as administrator (Windows) for PyAutoGUI permissions |
| Gesture not recognized | Adjust `confidence_threshold` in config.json |

## 📄 License

MIT License — Feel free to use and modify!