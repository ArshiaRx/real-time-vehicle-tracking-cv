# Real-Time Vehicle Tracking System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

A computer vision system for tracking and counting vehicles in video footage. Built as a course project for CPS843 (Introduction to Computer Vision) at Toronto Metropolitan University.

> ⚠️ **Note**: This is an educational project developed for visualization and learning purposes. It demonstrates core computer vision concepts like optical flow, object detection, and Kalman filtering.

---

## 🎯 What It Does

- Tracks vehicles across video frames using either **optical flow** or **YOLOv8 detection**
- Counts vehicles crossing a user-defined line or entering/exiting a region
- Provides both a **command-line interface** and a **Streamlit web dashboard**
- Outputs annotated videos with tracking visualization and statistics

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **OpenCV** | Video processing, optical flow (Lucas-Kanade), drawing/visualization |
| **YOLOv8** (Ultralytics) | Deep learning-based vehicle detection |
| **FilterPy** | Kalman filter implementation for trajectory smoothing |
| **Streamlit** | Web-based interface for interactive processing |
| **NumPy/SciPy** | Numerical computations |
| **Plotly** | Interactive charts in the web interface |

---

## 📁 Project Structure

```
real-time-vehicle-tracking-cv/
├── Project/                    # Main application code
│   ├── app.py                  # Streamlit web interface
│   ├── main.py                 # Command-line entry point
│   ├── config.py               # Configuration parameters
│   ├── requirements.txt        # Python dependencies
│   ├── test_system.py          # Unit tests for components
│   ├── yolov8n.pt              # YOLO model weights
│   │
│   ├── src/                    # Core modules
│   │   ├── video_processor.py      # Main processing pipeline
│   │   ├── optical_flow_tracker.py # Lucas-Kanade tracking
│   │   ├── yolo_detector.py        # YOLOv8 detection wrapper
│   │   ├── kalman_filter.py        # Trajectory smoothing
│   │   ├── vehicle_counter.py      # ROI-based counting logic
│   │   ├── auto_roi_detector.py    # Automatic ROI detection
│   │   └── utils.py                # Visualization helpers
│   │
│   ├── data/                   # Sample input videos
│   └── output/                 # Processed video outputs
│
├── Presentation and Progress Report/
│   └── progress_report.pdf
│
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Setup

```bash
cd Project
pip install -r requirements.txt
```

### 2. Run

**Web Interface** (easier to use):
```bash
streamlit run app.py
# or: python -m streamlit run app.py
```

**Command Line** (more control):
```bash
# Basic usage
python main.py --video data/sample_traffic_test2.mp4

# With YOLO detection (recommended for accuracy)
python main.py --video data/sample_traffic_test2.mp4 --yolo

# Webcam
python main.py --webcam --yolo
```

### 3. Draw ROI

When prompted, click to define your counting region:
- **Line**: Click 2 points → counts vehicles crossing
- **Polygon**: Click 3+ points → counts vehicles entering/exiting

Press `q` to confirm, `r` to reset.

---

## ⌨️ Keyboard Controls (CLI Mode)

| Key | Action |
|-----|--------|
| `Space` | Pause/Resume |
| `s` | Step one frame (when paused) |
| `t` | Toggle track trails |
| `v` | Verbose mode |
| `r` | Reset counters |
| `q` / `Esc` | Quit |

---

## 📖 More Details

See [`Project/README.md`](Project/README.md) for:
- Detailed algorithm explanations
- Configuration options
- Troubleshooting
- Implementation notes

---

## 👤 Author

**Arshia Rahim**  
Computer Engineering (Software) @ Toronto Metropolitan University

- GitHub: [@ArshiaRx](https://github.com/ArshiaRx)
- LinkedIn: [in/arshia-rahim](https://www.linkedin.com/in/arshia-rahim)

---

## 📚 Course Info

**CPS843 - Introduction to Computer Vision**  
Fall 2025 • Toronto Metropolitan University

---

## 📄 License

Educational project. Feel free to reference with attribution, but please don't copy directly for coursework.
