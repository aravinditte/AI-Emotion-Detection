# AI Emotion Detection — Fast Real-Time Facial Emotion HUD

Developed by Aravind

A lightning-fast real-time webcam emotion detection system with a cyberpunk neon HUD overlay. Built for developers and creators with simple setup and instant results.

---

## How to Run — Quick Start

### 1. Clone & Enter Project
```bash
git clone https://github.com/aravinditte/AI-Emotion-Detection.git
cd AI-Emotion-Detection
```

### 2. Create & Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Choose How to Run

#### [A] **Ultra-Lite/CPU-Only (Recommended for Most Users)**
- Uses only OpenCV, MediaPipe, and NumPy for face detection and fast emotion heuristics
- NO deep learning dependencies
- Best for non-GPU machines, laptops, development

```bash
pip install --upgrade pip
pip install -r requirements_lite.txt
python emotion_detector/app.py
```

#### [B] **Full DL (CPU/GPU, DeepFace & ONNX)**
- Enables ONNX and optional DeepFace backend for higher emotion accuracy
- Works on both CPU and compatible GPU (CUDA required for GPU)

```bash
pip install --upgrade pip
pip install -r requirements_full.txt
python emotion_detector/app.py
```

### GPU Acceleration
- ONNX will use your GPU if installed with compatible drivers
- DeepFace (via TensorFlow) requires a CUDA-capable GPU and CUDA libraries
- For maximum speed: `pip install tensorflow-gpu` (optional)

### Runtime Controls
- `S` — Save screenshot
- `+`/`=` — Increase smoothing
- `-` — Decrease smoothing
- `]` — Increase bbox tracking speed
- `[` — Decrease bbox tracking speed
- `D` — Enable deep learning backend
- `M` — Switch DL backend (ONNX/DeepFace)
- `ESC` — Exit

---

## Troubleshooting (Common Issues)
- **Grey overlay?** — Fixed; latest version overlays only border and visible effects
- **Camera permissions error?** — Make sure webcam is plugged in & accessible
- **Protobuf or MediaPipe error?** — Use requirements_lite.txt if unsure; clean environment fixes 99% of problems
- **TensorFlow/DeepFace error?** — Use requirements_lite.txt, or check GPU/CUDA installation

For advanced fixes, see `TROUBLESHOOTING.md` in repo.

---

## Features
- Real-time webcam & face tracking
- Fast facial emotion heuristics (works everywhere)
- Optional full deep learning (ONNX, DeepFace)
- Cyberpunk neon HUD with only clean overlays
- All overlays/texts drawn directly on camera feed
- Screenshot saving

---

MIT License — Built and maintained by Aravind
