# Troubleshooting Guide

Developed by Aravind

## Common Issues and Solutions

### 1. Protobuf Compatibility Error

**Error Message:**
```
AttributeError: 'SymbolDatabase' object has no attribute 'GetPrototype'
Face detection error: 'SymbolDatabase' object has no attribute 'GetPrototype'
```

**Cause:** Version conflict between protobuf, MediaPipe, and TensorFlow.

**Solution A - Use Lightweight Mode (Recommended):**
```powershell
# Remove current environment
deactivate
cd ..
Remove-Item -Recurse -Force .venv

# Create fresh environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install lightweight dependencies (no TensorFlow)
pip install --upgrade pip
pip install -r requirements_lite.txt

# Run application
cd emotion_detector
python app.py
```

**Solution B - Fix Protobuf in Full Installation:**
```powershell
# Stay in current environment
pip uninstall protobuf -y
pip install protobuf==3.20.3

# If that doesn't work, reinstall everything
pip uninstall mediapipe tensorflow deepface -y
pip install protobuf==3.20.3
pip install mediapipe>=0.10.0
pip install tensorflow>=2.13.0,<2.16.0
pip install deepface

# Run application
python app.py
```

**Solution C - Clean Reinstall:**
```powershell
# Remove virtual environment
deactivate
Remove-Item -Recurse -Force .venv

# Create new environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install in specific order
pip install --upgrade pip setuptools wheel
pip install protobuf==3.20.3
pip install numpy>=1.24.0
pip install opencv-python>=4.8.0
pip install mediapipe>=0.10.0
pip install tensorflow>=2.13.0,<2.16.0
pip install deepface
pip install onnxruntime
pip install requests

# Test
cd emotion_detector
python app.py
```

---

### 2. Camera Not Opening

**Error Message:**
```
ERROR: Could not open webcam!
```

**Solutions:**

1. **Check if another app is using the camera:**
   - Close Zoom, Teams, Skype, or other video apps
   - Close browser tabs with camera access

2. **Check Windows camera permissions:**
   ```
   Settings > Privacy > Camera > Allow desktop apps to access camera
   ```

3. **Try different camera index:**
   Edit `app.py` line ~25:
   ```python
   cap = cv2.VideoCapture(1)  # Try 0, 1, 2, etc.
   ```

4. **Test camera with Windows Camera app:**
   - If it doesn't work there, it's a hardware/driver issue

---

### 3. Import Errors

**Error Message:**
```
ModuleNotFoundError: No module named 'detector'
```

**Solutions:**

1. **Run from correct directory:**
   ```powershell
   cd emotion_detector
   python app.py
   ```

2. **Or use the run script:**
   ```powershell
   # From project root
   python run.py
   ```

3. **Or run as module:**
   ```powershell
   python -m emotion_detector.app
   ```

---

### 4. TensorFlow Installation Fails

**Error Message:**
```
ERROR: Could not find a version that satisfies the requirement tensorflow
```

**Solutions:**

1. **Use CPU-only version:**
   ```powershell
   pip install tensorflow-cpu>=2.13.0
   ```

2. **Check Python version:**
   ```powershell
   python --version
   ```
   TensorFlow 2.13+ requires Python 3.9-3.11

3. **Use lightweight mode instead:**
   ```powershell
   pip install -r requirements_lite.txt
   ```

---

### 5. Low FPS / Slow Performance

**Issue:** Application runs at <10 FPS

**Solutions:**

1. **Ensure DL mode is OFF** (default):
   - Don't press 'D' to enable deep learning
   - Check status panel shows "DL Mode: OFF"

2. **Reduce video resolution:**
   Edit `app.py` after line 26:
   ```python
   cap = cv2.VideoCapture(0)
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

3. **Adjust smoothing (lower = faster):**
   - Press `-` multiple times during runtime
   - Or edit `emotion_analyzer.py` line 41:
   ```python
   self.smoothing_factor = 0.70  # Lower value
   ```

4. **Disable landmark refinement:**
   Edit `app.py` line 38:
   ```python
   detector = FaceDetector(max_faces=1, refine_landmarks=False)
   ```

---

### 6. ONNX Model Download Fails

**Error Message:**
```
Failed to download ONNX model: ...
```

**Solutions:**

1. **Check internet connection**

2. **Manual download:**
   ```powershell
   cd emotion_detector\assets
   curl -L -o emotion_model.onnx https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx
   ```

3. **Use browser to download:**
   - Visit: https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus
   - Download `emotion-ferplus-8.onnx`
   - Place in `emotion_detector/assets/emotion_model.onnx`

4. **Continue without ONNX:**
   - Keep DL mode OFF
   - Use heuristic detection (default, works great)

---

### 7. Virtual Environment Issues

**Issue:** Can't activate virtual environment

**Windows PowerShell:**
```powershell
# Set execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
.venv\Scripts\activate.bat
```

**Alternative - Run directly:**
```powershell
.venv\Scripts\python.exe app.py
```

---

### 8. OpenCV Window Issues

**Issue:** Window doesn't appear or freezes

**Solutions:**

1. **Check if window is behind other windows**

2. **Try different OpenCV backend:**
   Add before `cv2.imshow()` in `app.py`:
   ```python
   cv2.namedWindow('AI Emotion Detection', cv2.WINDOW_NORMAL)
   ```

3. **Update graphics drivers**

4. **Try without OpenGL:**
   ```powershell
   set OPENCV_VIDEOIO_PRIORITY_MSMF=0
   python app.py
   ```

---

### 9. DeepFace Crashes

**Error:** Application crashes when pressing 'D'

**Solutions:**

1. **Switch to ONNX backend:**
   - Press 'M' to switch to ONNX
   - Then press 'D' to enable DL

2. **Use CPU-only TensorFlow:**
   ```powershell
   pip uninstall tensorflow
   pip install tensorflow-cpu
   ```

3. **Stay in heuristic mode:**
   - Don't enable DL mode
   - Heuristic mode is fast and accurate enough for most uses

---

### 10. Memory Issues

**Issue:** High memory usage or out of memory errors

**Solutions:**

1. **Use lightweight mode:**
   ```powershell
   pip install -r requirements_lite.txt
   ```

2. **Don't enable DL mode**

3. **Close other applications**

4. **Reduce video resolution** (see solution #5)

---

## Platform-Specific Issues

### Windows

**Antivirus blocking:**
- Add Python and the project folder to antivirus exceptions
- Windows Defender may block camera access

**Permission issues:**
```powershell
Settings > Privacy & Security > Camera > Allow apps to access camera
```

### Linux

**Camera permissions:**
```bash
sudo usermod -a -G video $USER
# Log out and back in
```

**Missing libraries:**
```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx libglib2.0-0 python3-opencv

# Fedora
sudo dnf install mesa-libGL glib2
```

### macOS

**Camera permissions:**
```
System Preferences > Security & Privacy > Camera > Terminal (check)
```

**Python version:**
```bash
# Use python3 explicitly
python3 -m venv .venv
source .venv/bin/activate
python3 app.py
```

---

## Getting More Help

If your issue isn't listed here:

1. **Check the full error message carefully**
2. **Look at the [SETUP.md](SETUP.md) guide**
3. **Try lightweight mode first** (requirements_lite.txt)
4. **Search for the error message online**
5. **Open a GitHub issue** with:
   - Your OS and Python version
   - Full error message
   - Steps you've tried
   - Output of `pip list`

---

## Quick Diagnostic

Run this to check your setup:

```python
import sys
print(f"Python: {sys.version}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except: print("OpenCV: NOT INSTALLED")

try:
    import mediapipe
    print(f"MediaPipe: {mediapipe.__version__}")
except: print("MediaPipe: NOT INSTALLED")

try:
    import numpy
    print(f"NumPy: {numpy.__version__}")
except: print("NumPy: NOT INSTALLED")

try:
    import google.protobuf
    print(f"Protobuf: {google.protobuf.__version__}")
except: print("Protobuf: NOT INSTALLED")

try:
    import tensorflow
    print(f"TensorFlow: {tensorflow.__version__}")
except: print("TensorFlow: NOT INSTALLED")

try:
    import onnxruntime
    print(f"ONNX Runtime: {onnxruntime.__version__}")
except: print("ONNX Runtime: NOT INSTALLED")
```

Save as `check_setup.py` and run: `python check_setup.py`

---

Developed by Aravind
