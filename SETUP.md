# Setup Guide - AI Emotion Detection

Developed by Aravind

Complete installation and setup instructions for the AI Emotion Detection system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Options](#installation-options)
3. [Quick Start](#quick-start)
4. [Detailed Setup](#detailed-setup)
5. [Troubleshooting](#troubleshooting)
6. [Platform-Specific Notes](#platform-specific-notes)

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.10 or higher (3.11 recommended)
- **Webcam**: Any USB or built-in webcam supported by OpenCV
- **RAM**: Minimum 4GB (8GB recommended for deep learning modes)
- **Disk Space**: 
  - Lightweight: ~500MB
  - Full installation: ~3GB (includes deep learning models)

### Python Installation

If you don't have Python installed:

**Windows:**
```powershell
# Download from python.org or use winget
winget install Python.Python.3.11
```

**macOS:**
```bash
# Using Homebrew
brew install python@3.11
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Fedora
sudo dnf install python3.11
```

Verify installation:
```bash
python --version
# or
python3 --version
```

## Installation Options

### Option 1: Lightweight Installation (Recommended)

Fast heuristic-based emotion detection with optional ONNX support.
Best for quick setup and good performance on most systems.

### Option 2: Full Installation

Includes all deep learning frameworks (TensorFlow, PyTorch, DeepFace).
Best for maximum accuracy but requires more disk space and setup time.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/aravinditte/AI-Emotion-Detection.git
cd AI-Emotion-Detection
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

**Lightweight:**
```bash
pip install --upgrade pip
pip install -r requirements_lite.txt
```

**Full:**
```bash
pip install --upgrade pip
pip install -r requirements_full.txt
```

### 4. Run the Application

```bash
cd emotion_detector
python app.py
```

Or from the project root:
```bash
python run.py
```

## Detailed Setup

### Step-by-Step Installation

#### 1. Clone and Navigate

```bash
git clone https://github.com/aravinditte/AI-Emotion-Detection.git
cd AI-Emotion-Detection
```

#### 2. Virtual Environment Setup

Creating a virtual environment isolates project dependencies:

```bash
# Create virtual environment
python -m venv .venv

# Activate it (see platform-specific commands above)
# Your prompt should now show (.venv)
```

#### 3. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

#### 4. Install Project Dependencies

Choose based on your needs:

**For Lightweight Mode:**
```bash
pip install -r requirements_lite.txt
```

This installs:
- OpenCV (computer vision)
- MediaPipe (face detection)
- NumPy (numerical computing)
- ONNX Runtime (optional deep learning)
- Requests (model downloading)

**For Full Mode:**
```bash
pip install -r requirements_full.txt
```

Additionally installs:
- PyTorch (deep learning framework)
- Transformers (NLP models)
- DeepFace (emotion recognition)
- TensorFlow (via DeepFace)

#### 5. Verify Installation

Test that all modules can be imported:

```bash
python -c "import cv2, mediapipe, numpy; print('Core dependencies OK')"
```

#### 6. Run Application

```bash
# From emotion_detector directory
cd emotion_detector
python app.py

# Or from project root
python run.py

# Or as a module
python -m emotion_detector.app
```

## Troubleshooting

### Common Issues

#### Camera Not Opening

**Problem**: `ERROR: Could not open webcam!`

**Solutions**:
1. Check if another application is using the camera
2. Verify camera permissions:
   - **Windows**: Settings → Privacy → Camera → Allow desktop apps
   - **macOS**: System Preferences → Security & Privacy → Camera
   - **Linux**: Check `/dev/video*` permissions
3. Try a different camera index in `app.py`:
   ```python
   cap = cv2.VideoCapture(1)  # Try 1, 2, etc.
   ```

#### Module Import Errors

**Problem**: `ModuleNotFoundError: No module named 'detector'`

**Solutions**:
1. Ensure you're running from the correct directory:
   ```bash
   cd emotion_detector
   python app.py
   ```
2. Or use the convenience script:
   ```bash
   python run.py
   ```
3. Or run as a module:
   ```bash
   python -m emotion_detector.app
   ```

#### TensorFlow Installation Issues

**Problem**: TensorFlow fails to install or crashes

**Solutions**:
1. Try CPU-only version:
   ```bash
   pip install tensorflow-cpu
   ```
2. Use lightweight mode instead (no TensorFlow required)
3. Use ONNX backend for deep learning (press M to switch)

#### ONNX Model Download Fails

**Problem**: Model download error when enabling DL mode

**Solutions**:
1. Check internet connection
2. Manually download model:
   ```bash
   cd emotion_detector/assets
   curl -L -o emotion_model.onnx \
     https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx
   ```
3. Continue using heuristic mode (DL toggle OFF)

#### Low FPS Performance

**Problem**: Application runs slowly

**Solutions**:
1. Keep deep learning mode OFF (default)
2. Reduce video resolution in `app.py`:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```
3. Adjust smoothing parameters (lower = faster)

#### Virtual Environment Not Activating

**Problem**: Virtual environment commands not working

**Solutions**:

**Windows PowerShell execution policy**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Alternative activation methods**:
```bash
# Windows
.venv\Scripts\python.exe app.py

# macOS/Linux
.venv/bin/python app.py
```

## Platform-Specific Notes

### Windows

- Use PowerShell (recommended) or Command Prompt
- Ensure Python is in PATH
- Audio features (shutter sound) work out of the box
- If camera issues, check Windows Defender Firewall

### macOS

- May need to grant Terminal camera permissions
- Audio features require system permissions
- Use `python3` command instead of `python`
- M1/M2 Macs: TensorFlow may require Rosetta or ARM-specific builds

### Linux

- Ensure camera device has proper permissions:
  ```bash
  sudo usermod -a -G video $USER
  # Log out and back in
  ```
- Install system dependencies:
  ```bash
  # Ubuntu/Debian
  sudo apt install libgl1-mesa-glx libglib2.0-0
  
  # Fedora
  sudo dnf install mesa-libGL glib2
  ```
- Audio features may not work (platform-specific)

## Advanced Configuration

### Custom Model Paths

Edit `emotion_analyzer.py` to change model locations:

```python
self._model_path = 'custom/path/to/model.onnx'
```

### Video Source Selection

Modify `app.py` to use different video sources:

```python
# Different camera
cap = cv2.VideoCapture(1)

# Video file
cap = cv2.VideoCapture('path/to/video.mp4')

# IP camera
cap = cv2.VideoCapture('rtsp://camera_ip:port/stream')
```

### Performance Tuning

In `app.py`, adjust these parameters:

```python
# Face detector sensitivity
detector = FaceDetector(
    max_faces=1,  # Increase for multiple faces
    refine_landmarks=True  # Set False for speed
)

# Analyzer smoothing
analyzer.smoothing_factor = 0.85  # Lower = faster response
analyzer.bbox_lerp = 0.20  # Higher = faster tracking
```

## Uninstallation

To completely remove the project:

```bash
# Deactivate virtual environment
deactivate

# Remove project directory
cd ..
rm -rf AI-Emotion-Detection

# Or on Windows
rmdir /s AI-Emotion-Detection
```

## Getting Help

If you encounter issues not covered here:

1. Check the main [README.md](README.md) for troubleshooting tips
2. Review error messages carefully
3. Ensure all dependencies are correctly installed
4. Try the lightweight mode first
5. Open an issue on GitHub with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce

## Next Steps

Once installed successfully:

1. Read the [README.md](README.md) for usage instructions
2. Explore keyboard controls
3. Try different detection modes
4. Experiment with parameter tuning
5. Capture and save your results

Enjoy using AI Emotion Detection!

---

Developed by Aravind
