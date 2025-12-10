# Vehicle Tracking & Counting System - Technical Documentation

This document covers the implementation details, architecture decisions, and usage specifics for the vehicle tracking system.

---

## 📐 Architecture Overview

The system follows a modular pipeline design:

```
Input Video/Webcam
       │
       ▼
┌──────────────────┐
│  Detection       │  ← Either Optical Flow OR YOLOv8
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Tracking        │  ← Track ID assignment, trajectory history
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Kalman Filter   │  ← Smooths trajectories (optical flow mode only)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Vehicle Counter │  ← ROI crossing detection
└────────┬─────────┘
         │
         ▼
   Annotated Output + Statistics
```

---

## 🔧 Core Components

### 1. Video Processor (`src/video_processor.py`)

The main orchestrator. Initializes all components and runs the frame-by-frame loop.

```python
processor = VideoProcessor(
    use_kalman=True,      # Enable trajectory smoothing
    roi_type='line',      # 'line' or 'polygon'
    roi_points=None,      # User-defined or auto-detected
    use_yolo=False,       # Toggle YOLO vs optical flow
    yolo_confidence=0.4,  # Detection threshold
    min_box_size=20       # Filter tiny detections
)
```

### 2. Optical Flow Tracker (`src/optical_flow_tracker.py`)

Uses **Lucas-Kanade sparse optical flow** to track feature points across frames.

**How it works:**
1. Detect "good features to track" using Shi-Tomasi corner detection
2. Track those points frame-to-frame using Lucas-Kanade
3. Manage track lifecycle (create, update, delete)

**Parameters** (in code):
```python
lk_params = dict(
    winSize=(15, 15),    # Search window size
    maxLevel=2,          # Pyramid levels
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

feature_params = dict(
    maxCorners=500,      # Max features to track
    qualityLevel=0.01,   # Feature quality threshold
    minDistance=10,      # Min distance between features
    blockSize=7
)
```

**Limitations:**
- Struggles with vehicles that have similar colors to the road
- Can't detect stationary vehicles
- Features may drift or get lost in low-texture areas

### 3. YOLO Detector (`src/yolo_detector.py`)

Wraps **YOLOv8** for vehicle detection. More robust than optical flow but slower.

**Vehicle classes detected** (COCO IDs):
- 2: car
- 3: motorcycle
- 5: bus
- 7: truck

**Key features:**
- Confidence thresholding
- Minimum box size filtering (removes distant/small detections)
- Track ID assignment using IoU matching between frames

### 4. Kalman Filter (`src/kalman_filter.py`)

Implements a **constant velocity model** for trajectory smoothing.

**State vector:** `[x, y, vx, vy]` (position + velocity)

**Why it helps:**
- Smooths noisy optical flow measurements
- Predicts position during brief occlusions
- Provides velocity estimates

**Note:** Kalman filtering is disabled when using YOLO mode (YOLO's built-in tracking is already smooth).

### 5. Vehicle Counter (`src/vehicle_counter.py`)

Handles ROI-based counting logic.

**Line ROI:**
- Detects when a track crosses from one side to the other
- Determines direction (up/down) based on crossing vector

**Polygon ROI:**
- Tracks whether each vehicle is inside or outside
- Counts entries and exits separately

**Anti-double-counting:**
- Each track ID can only trigger one count
- Cooldown period prevents rapid re-counting

### 6. Auto ROI Detector (`src/auto_roi_detector.py`)

Automatically suggests a counting line by analyzing vehicle movement patterns in the first few seconds of video. Uses YOLO to detect vehicles and estimates the main flow direction.

---

## 🎮 Usage Examples

### Command Line

```bash
# Basic - optical flow tracking
python main.py --video data/sample_traffic_test2.mp4

# YOLO detection (recommended for better accuracy)
python main.py --video data/sample_traffic_test2.mp4 --yolo

# Adjust YOLO sensitivity
python main.py --video data/sample_traffic_test2.mp4 --yolo --confidence 0.5 --min-box-size 30

# Custom direction labels
python main.py --video data/sample_traffic_test2.mp4 --direction-up "Northbound" --direction-down "Southbound"

# Save output without display (batch processing)
python main.py --video data/sample_traffic_test2.mp4 --yolo --no-display --output output/result.mp4

# Polygon ROI instead of line
python main.py --video data/sample_traffic_test2.mp4 --roi-type polygon

# Adjust playback speed
python main.py --video data/sample_traffic_test2.mp4 --speed 2.0
```

### Web Interface

```bash
streamlit run app.py
```

Features:
- Upload videos or use demo files
- Adjust parameters with sliders
- Auto-detect optimal counting line
- Download processed videos and CSV statistics
- Real-time progress tracking

---

## ⚙️ Configuration

Default values are in `config.py`. You can modify these or pass them as CLI arguments.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `YOLO_CONFIDENCE_THRESHOLD` | 0.4 | Min confidence for YOLO detections |
| `YOLO_MIN_BOX_SIZE` | 20 | Filter boxes smaller than this (pixels) |
| `MAX_TRACK_LENGTH` | 30 | Max points in track history |
| `MIN_TRACK_LENGTH` | 5 | Min points before track is valid |

---

## 🐛 Troubleshooting

### No vehicles detected

- **Optical flow mode**: Check if video has enough texture/contrast. Try YOLO mode.
- **YOLO mode**: Lower the confidence threshold (`--confidence 0.3`)

### Vehicles being double-counted

- ROI might be too close to where vehicles slow down/stop
- Try repositioning the counting line
- Increase `min_box_size` to filter jittery small detections

### Tracking IDs keep changing

- Common with optical flow when vehicles have similar colors
- Use YOLO mode for more stable tracking
- This is a known limitation of appearance-agnostic tracking

### Slow performance

- Use `--no-display` for batch processing
- Lower video resolution
- YOLO is slower than optical flow (~10-15 FPS vs ~60 FPS on CPU)

### Streamlit won't start

```bash
# Try this instead of 'streamlit run app.py'
python -m streamlit run app.py
```

---

## 📊 Output Files

Processed videos are saved to `output/` with the suffix `_tracked.mp4`.

The web interface also provides:
- CSV export of frame-by-frame counts
- Interactive charts (count over time, active tracks)

---

## 🔮 Future Work

- [ ] Frontend web application (React/TypeScript) for better UX
- [ ] RTSP stream support for live camera feeds
- [ ] Vehicle type classification (separate counts by car/truck/bus)
- [ ] Speed estimation using perspective calibration
- [ ] Database integration for historical analytics
- [ ] GPU acceleration for real-time YOLO inference

---

## 🧪 Testing

Run the test suite to verify components work:

```bash
python test_system.py
```

This tests:
- Kalman filter prediction/update
- Vehicle counter crossing detection
- Optical flow tracker initialization
- Video processor integration

---

## 📚 Algorithm References

- **Lucas-Kanade Optical Flow**: Lucas, B. D., & Kanade, T. (1981). "An iterative image registration technique with an application to stereo vision."
- **Kalman Filter**: Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems."
- **Shi-Tomasi Corners**: Shi, J., & Tomasi, C. (1994). "Good features to track."
- **YOLO**: Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection."

---

## 👤 Author

**Arshia Rahim**  
Computer Engineering (Software) @ Toronto Metropolitan University

- GitHub: [@ArshiaRx](https://github.com/ArshiaRx)
- LinkedIn: [in/arshia-rahim](https://www.linkedin.com/in/arshia-rahim)

---

*Built for CPS843 - Introduction to Computer Vision (Fall 2025)*
