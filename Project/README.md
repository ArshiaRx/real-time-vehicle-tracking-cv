# Vehicle Tracking & Counting System - User Guide

This document provides a practical guide for using the vehicle tracking and counting system.

---

## 🚀 Quick Setup

### Installation

```bash
pip install -r requirements.txt
```

### Run the Web Interface

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 💻 Usage

### Web Interface (Recommended)

1. **Start the application**: Run `streamlit run app.py`
2. **Select video source**: 
   - Choose from demo videos in the `data/` folder, or
   - Upload your own MP4 or MOV file
3. **Configure parameters**:
   - **Use YOLOv8 Detection**: Enable for better accuracy (recommended)
   - **YOLO Confidence Threshold**: 0.4 recommended (balances accuracy and false positives)
   - **Minimum Box Size**: Filter out small detections (default: 20 pixels)
4. **Set up ROI**:
   - **Auto-Detect**: Let the system find optimal counting line using YOLOv8
   - **Manual**: Draw your own line or polygon
5. **Process**: Click "PROCESS VIDEO" and wait for results
6. **View results**: Download processed video and CSV statistics

### Command Line Interface

```bash
# Basic usage with optical flow
python main.py --video data/sample_traffic_test2.mp4

# With YOLO detection (recommended)
python main.py --video data/sample_traffic_test2.mp4 --yolo

# Adjust confidence threshold
python main.py --video data/sample_traffic_test2.mp4 --yolo --confidence 0.4

# Custom minimum box size
python main.py --video data/sample_traffic_test2.mp4 --yolo --min-box-size 30

# Polygon ROI instead of line
python main.py --video data/sample_traffic_test2.mp4 --roi-type polygon
```

---

## ⚙️ Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `YOLO_CONFIDENCE_THRESHOLD` | 0.4 | Minimum confidence for YOLO detections (0.1-0.9) |
| `YOLO_MIN_BOX_SIZE` | 20 | Filter boxes smaller than this (pixels) |
| `MAX_TRACK_LENGTH` | 30 | Maximum points in track history |
| `MIN_TRACK_LENGTH` | 5 | Minimum points before track is valid |

### Confidence Threshold Guidelines

- **0.4 (Recommended)**: Best balance between accuracy (90-95%) and false positives
- **0.1**: Higher recall, captures more vehicles but includes more false positives (15-20%)
- **0.5+**: Higher precision, fewer false positives but may miss smaller vehicles

---

## 🎯 Key Features Explained

### Detection Methods

**Optical Flow (Lucas-Kanade)**
- Fast processing (60+ FPS)
- Tracks feature points across frames
- Good for videos with clear vehicle features
- May struggle with similar-colored vehicles

**YOLOv8 Deep Learning**
- High accuracy (90-95% at threshold 0.4)
- Detects complete vehicles as objects
- More robust to lighting and appearance changes
- Slower processing (10-15 FPS on CPU)

### ROI Types

**Line ROI**
- Draw a line across the road
- Counts vehicles crossing from one side to the other
- Determines direction (up/down) based on crossing vector

**Polygon ROI**
- Draw a polygon region
- Counts vehicles entering and exiting separately
- Useful for counting vehicles in specific zones

### Track IDs

Each detected vehicle is assigned a unique track ID (e.g., ID0, ID1, ID2) that persists across frames. This ensures:
- No double-counting (each ID counted only once)
- Consistent tracking throughout the video
- Accurate counting even with occlusions

---

## 📊 Understanding Results

### Processed Video Output

The annotated video shows:
- **Track IDs**: Numbers on each vehicle (ID0, ID1, etc.)
- **Trajectory Paths**: Colored trails showing vehicle movement
- **Bounding Boxes**: Detection boxes around vehicles
- **ROI Lines**: Visual representation of counting boundaries

### Statistics Panel

- **Total Vehicles**: Total count of all vehicles crossing ROI
- **Direction Up**: Vehicles moving in one direction
- **Direction Down**: Vehicles moving in opposite direction
- **Count History Chart**: Frame-by-frame count progression
- **Active Tracks Chart**: Number of active tracks over time

### CSV Export

Download frame-by-frame statistics including:
- Frame number
- Total count
- Up direction count
- Down direction count

---

## 🔧 Troubleshooting

### No vehicles detected

- **Optical flow mode**: Try YOLO mode for better detection
- **YOLO mode**: Lower confidence threshold (try 0.3 or 0.2)
- Check video quality and lighting conditions

### Vehicles being double-counted

- Reposition the counting line away from where vehicles slow down
- Increase minimum box size to filter jittery detections
- Use multiple lines for validation

### Slow processing

- YOLOv8 is slower than optical flow (~10-15 FPS vs 60+ FPS)
- Lower video resolution for faster processing
- Use optical flow mode for real-time applications

### Streamlit won't start

```bash
# Try this instead
python -m streamlit run app.py
```

---

## 📁 Project Structure

```
Project/
├── app.py                  # Streamlit web interface
├── main.py                 # Command-line entry point
├── config.py               # Configuration parameters
├── requirements.txt        # Python dependencies
│
├── src/                    # Core modules
│   ├── video_processor.py      # Main processing pipeline
│   ├── optical_flow_tracker.py # Lucas-Kanade tracking
│   ├── yolo_detector.py        # YOLOv8 detection wrapper
│   ├── kalman_filter.py        # Trajectory smoothing
│   ├── vehicle_counter.py      # ROI-based counting logic
│   ├── auto_roi_detector.py    # Automatic ROI detection
│   └── utils.py                # Visualization helpers
│
├── data/                   # Sample input videos
└── output/                 # Processed video outputs
```

---

## 📚 Algorithm References

- **Lucas-Kanade Optical Flow**: Lucas, B. D., & Kanade, T. (1981)
- **Kalman Filter**: Kalman, R. E. (1960)
- **Shi-Tomasi Corners**: Shi, J., & Tomasi, C. (1994)
- **YOLO**: Redmon, J., et al. (2016)
- **YOLOv8**: Jocher, G., Chaurasia, A., & Qiu, J. (2023)

---

## 👥 Author

**Arshia Rahim** - System architecture, optical flow, Kalman filter, web application

**Collaborators**: Ansugan Subramaniam (YOLOv8 integration, vehicle detection optimization), Wajeehul Hassan (ROI detection, multi-line configuration, statistics visualization)

---

*Built for CPS843 - Introduction to Computer Vision (Fall 2025) at Toronto Metropolitan University*
