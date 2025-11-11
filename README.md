# AI Emotion Detection with Real-Time Overlay

Developed by Aravind

A real-time emotion detection application that uses webcam feed to identify facial emotions and overlays a futuristic cyberpunk-style heads-up display (HUD) with emotion analysis and persona mapping.

## Features

- **Real-time Face Detection**: Uses MediaPipe Face Mesh for accurate face and landmark detection
- **Multiple Emotion Detection Modes**:
  - Fast heuristic-based detection (default)
  - Optional deep learning inference via ONNX
  - Optional DeepFace integration
- **Cyberpunk Neon HUD**: Beautiful animated overlay with:
  - Glowing rounded face tracking box
  - Animated scanline effect
  - Glitch-style header text
  - Smooth fade animations
  - FPS counter
- **Persona Mapping**: Maps detected emotions to cyberpunk-themed personas
- **Screenshot Capture**: Save frames with overlays (press 'S')
- **Runtime Controls**: Adjust smoothing and detection parameters on-the-fly

## Requirements

- **OS**: Windows (tested), Linux/macOS (should work, audio features may vary)
- **Python**: 3.10 or higher (3.11 recommended)
- **Webcam**: Any OpenCV-compatible camera

## Quick Start (Lightweight Mode)

This mode uses fast heuristic-based emotion detection with optional ONNX support:

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.venv\Scripts\activate.bat

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements_lite.txt

# Run the application
cd emotion_detector
python app.py
```

## Full Installation (with Deep Learning)

For full DeepFace and transformer model support:

```bash
pip install -r requirements_full.txt
```

**Note**: Full installation includes TensorFlow which is large. If you encounter issues, stick to the lightweight mode or use ONNX.

## Project Structure

```
AI-Emotion-Detection/
├── emotion_detector/
│   ├── app.py                 # Main application entry point
│   ├── detector.py            # Face detection using MediaPipe
│   ├── emotion_analyzer.py    # Emotion classification engine
│   ├── hud_renderer.py        # HUD overlay rendering
│   └── assets/                # Asset storage (screenshots, models)
├── requirements_lite.txt      # Lightweight dependencies
├── requirements_full.txt      # Full dependencies with DL
├── LICENSE
└── README.md
```

## How It Works

### Architecture

1. **detector.py**: Wraps MediaPipe Face Mesh to detect faces and extract facial landmarks in real-time
2. **emotion_analyzer.py**: 
   - Implements fast heuristic emotion detection based on facial geometry
   - Optional deep learning inference via ONNX or DeepFace
   - Temporal smoothing for stable emotion labels
   - Persona mapping system
3. **hud_renderer.py**: 
   - Renders cyberpunk-style UI elements
   - Animated effects (scanlines, glows, glitch text)
   - Status panels and FPS display
4. **app.py**: Main loop that orchestrates detection, analysis, and rendering

### Emotion Detection

The system supports three detection modes:

- **Heuristic Mode** (default): Fast geometric analysis of facial features
  - Analyzes eye aspect ratio, mouth curvature, and facial proportions
  - Low latency, runs smoothly on most hardware
  - No external model downloads required
  
- **ONNX Mode**: Deep learning using ONNX runtime
  - Automatically downloads FER+ model on first run
  - Better accuracy than heuristics
  - Good performance without TensorFlow
  
- **DeepFace Mode**: Full deep learning pipeline
  - Highest accuracy
  - Requires full installation
  - May have compatibility issues on some systems

## Keyboard Controls

During runtime, you can control the application using these keys:

| Key | Action |
|-----|--------|
| `S` | Save screenshot with overlays |
| `+` or `=` | Increase emotion label smoothing (less jitter) |
| `-` | Decrease emotion label smoothing (more responsive) |
| `]` | Increase bounding box smoothing (slower tracking) |
| `[` | Decrease bounding box smoothing (faster tracking) |
| `D` | Toggle deep learning mode ON/OFF |
| `M` | Switch DL backend (ONNX ↔ DeepFace) |
| `ESC` | Exit application |

## Using ONNX Models

For deep learning without heavy TensorFlow installation:

1. The app will automatically attempt to download an ONNX emotion model on first DL-mode use
2. Model is saved to `emotion_detector/assets/emotion_model.onnx`
3. You can also manually place your own ONNX FER model at this location

**Expected format**: Grayscale 64x64 input, FER/FER+ style output

## Persona Mapping

Detected emotions are mapped to cyberpunk-themed personas:

| Emotion | Persona |
|---------|--------|
| Joy | Circuit Dancer |
| Surprise | Neon Explorer |
| Anger | Cyber Warrior |
| Sadness | Digital Wanderer |
| Confused | Quantum Thinker |
| Happy | Code Dreamer |
| Excited | Energy Surger |
| Fear | Shadow Watcher |
| Disgust | System Skeptic |
| Neutral | Data Observer |

## Troubleshooting

### Camera Not Opening
- Ensure no other application is using the webcam
- Check Windows Privacy settings (Settings > Privacy > Camera)
- Try a different camera index in `app.py` if you have multiple cameras

### Module Import Errors
- Always run from within the `emotion_detector` directory
- Or use: `python -m emotion_detector.app` from project root

### ONNX Download Fails
- Check internet connection
- Manually download and place in `assets/emotion_model.onnx`
- Continue using heuristic mode (DL toggle OFF)

### TensorFlow/DeepFace Crashes
- Try `pip install tensorflow-cpu` for better compatibility
- Use ONNX mode instead (`M` key to switch)
- Stay in heuristic mode for stability

## Performance Tips

- Start with heuristic mode (default) for best performance
- Only enable DL mode when needed (press `D`)
- Prefer ONNX over DeepFace for better speed
- Adjust smoothing parameters for your use case

## Screenshots

Press `S` during runtime to capture screenshots. They will be saved to `emotion_detector/assets/screenshots/` with timestamps.

## Contributing

Contributions are welcome! Areas for improvement:

- Additional emotion detection models
- More HUD themes and styles
- Enhanced persona mapping system
- Cross-platform audio support
- Performance optimizations

## License

MIT License - see LICENSE file for details

## Credits

Developed by Aravind

**Technologies used**:
- MediaPipe (Face detection)
- OpenCV (Computer vision)
- ONNX Runtime (Deep learning inference)
- NumPy (Numerical computing)

---

**Note**: First run may take longer as models are downloaded and initialized.
