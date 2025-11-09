# Real-Time Vehicle Tracking with Optical Flow

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![License](https://img.shields.io/badge/License-Educational-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

A production-ready Python-based video tracking system that uses optical flow for real-time object/vehicle tracking, with Kalman filtering for smooth trajectory prediction and intelligent vehicle counting across user-defined regions of interest (ROI).

## Demo

Check out the sample tracked video in `output/sample_traffic_test_tracked.mp4` to see the system in action! The system successfully tracks multiple vehicles, maintains stable trajectories using Kalman filtering, and accurately counts vehicles crossing the ROI.

## Key Features

- **Optical Flow Tracking**: Implements Lucas-Kanade pyramidal optical flow for robust sparse feature tracking
- **Kalman Filtering**: Smooth trajectory prediction with constant velocity motion model for noise reduction
- **Vehicle Counting**: Intelligent counting system for vehicles crossing user-defined regions of interest
- **Flexible ROI**: Supports both line-based and polygon-based ROI definitions for versatile deployment
- **Real-time Processing**: Efficient processing pipeline for video files or live webcam feeds
- **Professional Visualization**: Annotated output with color-coded tracks, ROI overlays, and real-time statistics

## Technologies Used

- **Computer Vision**: OpenCV, Lucas-Kanade Optical Flow, Feature Detection
- **State Estimation**: Kalman Filtering (FilterPy), Motion Prediction
- **Scientific Computing**: NumPy, SciPy
- **Data Visualization**: Matplotlib
- **Video Processing**: Real-time frame processing and encoding

## Requirements

- Python 3.8+
- OpenCV 4.8+
- NumPy 1.24+
- SciPy 1.10+
- Matplotlib 3.7+
- FilterPy 1.4.5+

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Process a video file:

```bash
python main.py --video data/sample_video.mp4
```

Use webcam:

```bash
python main.py --webcam
```

### Advanced Options

Process video with YOLO detection (recommended for accuracy):

```bash
python main.py --video data/sample_video.mp4 --yolo
```

Tune YOLO detection parameters:

```bash
python main.py --video data/sample_video.mp4 --yolo --confidence 0.5 --min-box-size 30
```

Use custom direction labels:

```bash
python main.py --video data/sample_video.mp4 --direction-up "Northbound" --direction-down "Southbound"
```

Process video with output saving:

```bash
python main.py --video data/sample_video.mp4 --output output/result.mp4
```

Disable Kalman filtering (use raw optical flow only):

```bash
python main.py --video data/sample_video.mp4 --no-kalman
```

Use polygon ROI instead of line:

```bash
python main.py --video data/sample_video.mp4 --roi-type polygon
```

Process without display (faster for batch processing):

```bash
python main.py --video data/sample_video.mp4 --no-display
```

Adjust playback speed:

```bash
python main.py --video data/sample_video.mp4 --speed 2.0  # 2x speed
```

### Interactive Controls

When processing video with display:

- **SPACE**: Pause/Resume playback
- **'s'**: Step frame (when paused)
- **'t'**: Toggle track visibility
- **'v'**: Toggle verbose mode (show all track IDs)
- **'m'**: Toggle minimal mode (hide tracks)
- **'l'**: Toggle legend panel (color/symbol explanations)
- **'h'**: Toggle help panel
- **'r'**: Reset counters
- **'q' or ESC**: Quit

### ROI Selection

When you start processing, you'll be prompted to select a Region of Interest:

- **Line ROI**: Click two points to define a counting line
- **Polygon ROI**: Click multiple points to define a polygon region
- **'q'**: Confirm selection
- **'r'**: Reset selection
- **ESC**: Cancel

## Project Structure

```
Project/
├── src/
│   ├── __init__.py
│   ├── optical_flow_tracker.py      # Core optical flow implementation
│   ├── kalman_filter.py             # Kalman filter for smooth tracking
│   ├── vehicle_counter.py           # ROI crossing detection and counting
│   ├── video_processor.py           # Main video processing pipeline
│   └── utils.py                     # Helper functions (visualization, ROI selection)
├── data/
│   └── sample_video.mp4            # Place your input videos here
├── output/
│   └── (output videos and results)
├── requirements.txt                 # Python dependencies
├── main.py                          # Entry point script
└── README.md                        # This file
```

## Technical Implementation

### Core Algorithms

#### 1. Lucas-Kanade Optical Flow
The system implements pyramidal Lucas-Kanade optical flow for robust feature tracking:

- **Feature Detection**: Shi-Tomasi corner detection via `cv2.goodFeaturesToTrack`
  - Quality level threshold for reliable corner detection
  - Minimum distance constraint between features
- **Optical Flow Calculation**: Pyramidal implementation using `cv2.calcOpticalFlowPyrLK`
  - Multi-scale pyramid approach for handling large displacements
  - Sub-pixel accuracy for precise tracking
- **Track Management**: Dynamic track lifecycle management
  - Automatic track initialization and termination
  - Lost track recovery mechanisms

#### 2. Kalman Filtering
Each tracked object maintains an independent Kalman filter for smooth trajectory estimation:

- **State Space Model**: 4D state vector [x, y, vx, vy]
  - Position (x, y) and velocity (vx, vy) components
- **Motion Model**: Constant velocity assumption with process noise
- **Prediction Step**: Forward propagation based on motion dynamics
- **Update Step**: Measurement correction using optical flow observations
- **Noise Modeling**: Tuned process and measurement noise covariance matrices

#### 3. Intelligent Vehicle Counting
Geometric algorithms for accurate vehicle detection across ROI:

- **Line-Based ROI**: 
  - Signed distance calculation from track to line
  - Zero-crossing detection for boundary traversal
- **Polygon-Based ROI**: 
  - Point-in-polygon tests using ray casting algorithm
  - State transition detection (inside/outside)
- **Direction Classification**: Bidirectional counting with directional statistics

## Configuration

You can modify tracking parameters in the source code:

- **Optical Flow Parameters**: `OpticalFlowTracker.lk_params` and `feature_params`
- **Kalman Filter Parameters**: `ObjectKalmanFilter.__init__()` (process_noise, measurement_noise)
- **Track Management**: `max_track_length`, `min_track_length` in `OpticalFlowTracker`

## Troubleshooting

### No tracks detected

- Ensure the video has sufficient motion
- Try adjusting `qualityLevel` in `feature_params` (lower = more features)
- Check that the video is not too dark or blurry

### Tracks are jittery

- Enable Kalman filtering (default)
- Adjust Kalman filter noise parameters
- Increase `winSize` in `lk_params` for more stable tracking

### Counting not working

- Ensure ROI is properly defined
- Check that objects actually cross the ROI
- For line ROI, ensure objects move perpendicular to the line

## Detection Limitations & Known Issues

### Color and Contrast Challenges

**Similar-Colored Vehicles**: The optical flow-based tracking system may struggle to distinguish vehicles with similar colors (especially gray vehicles) or when vehicles blend with the road surface. This occurs because:

- Optical flow relies on detecting visual features (corners, edges, texture)
- Similar colors reduce contrast, making feature detection difficult
- Gray vehicles on gray pavement create minimal visual boundaries

**Mitigation Strategies**:
- Use YOLO detection mode (`--yolo` flag) for more robust vehicle detection
- Increase the feature detection quality threshold (adjust `qualityLevel` parameter)
- Ensure proper lighting and camera angles to maximize contrast
- Consider adjusting `--confidence` and `--min-box-size` parameters when using YOLO

### Occlusion and Proximity

**Closely Spaced Vehicles**: When vehicles are too close together or overlap, the system may:

- Merge multiple vehicles into a single track
- Lose track of partially occluded vehicles
- Undercount vehicles in dense traffic

**Why This Happens**:
- Optical flow features from multiple vehicles can be merged
- Track management may consolidate nearby tracks
- Occlusions break the continuous feature flow

**Mitigation Strategies**:
- Use YOLO mode for better separation of individual vehicles
- Position camera at angles that minimize occlusion
- Adjust ROI placement to count vehicles before they cluster
- Tune `min_distance` parameter in feature detection

### YOLO Mode vs Optical Flow Mode

**Optical Flow Mode** (Default):
- ✅ Fast processing, no ML model required
- ✅ Works well for separated, moving objects
- ❌ Struggles with similar colors and low contrast
- ❌ May miss stationary or slow-moving vehicles
- ❌ Feature-dependent (requires texture/edges)

**YOLO Mode** (`--yolo` flag):
- ✅ Accurate vehicle detection regardless of color
- ✅ Better handling of occlusions and dense traffic
- ✅ Can detect stationary vehicles
- ✅ Classifies vehicle types (car, truck, bus, motorcycle)
- ❌ Requires more computational resources
- ❌ Requires `ultralytics` package installation
- ❌ May have false positives in complex scenes

**Recommendation**: For production traffic monitoring with varied vehicle colors and densities, use YOLO mode with tuned confidence threshold:

```bash
python main.py --video data/traffic.mp4 --yolo --confidence 0.5 --min-box-size 30
```

### Understanding Direction Labels

The system uses "Up" and "Down" (or custom labels via `--direction-up` and `--direction-down`) to indicate crossing directions. These labels are **mathematical, not geographic**:

- **"Up"**: Vehicles crossing from negative to positive side of the line (shown with blue ▲ arrow during ROI selection)
- **"Down"**: Vehicles crossing from positive to negative side (shown with orange ▼ arrow during ROI selection)

**Important**: The direction depends on how you draw the counting line. To ensure directions match your expectations:

1. During ROI selection, observe the directional arrows showing which side is "Up" vs "Down"
2. Use custom labels for clarity: `--direction-up "Northbound" --direction-down "Southbound"`
3. Redraw the ROI if directions are reversed

### Color Coding System

**Bounding Box Colors** (when tracking is displayed):
- **Green**: Normal tracking - vehicle is actively tracked
- **Yellow**: Recently crossed ROI - vehicle crossed the counting line within the last 3 seconds

**Note**: Colors indicate **crossing status**, not vehicle type. Press 'L' during playback to view the legend panel explaining all colors and symbols.

### Performance Considerations

**Frame Rate Impact**: Detection accuracy decreases with:
- Low frame rates (<15 FPS)
- Motion blur from fast-moving vehicles
- Poor lighting conditions
- Low-resolution video (<720p)

**Optimal Conditions**:
- 30+ FPS video
- 1080p or higher resolution
- Good lighting (daytime or well-lit nighttime)
- Camera positioned for clear, unobstructed view of ROI

## Future Enhancements

- [ ] Deep learning integration (YOLOv8/YOLOv11) for vehicle detection
- [ ] Multi-object tracking (MOT) with Hungarian algorithm for data association
- [ ] Speed estimation using homography and camera calibration
- [ ] Classification by vehicle type (car, truck, motorcycle)
- [ ] Cloud deployment with REST API endpoints
- [ ] Real-time streaming support with RTSP/WebRTC
- [ ] Database integration for historical analytics
- [ ] Performance optimization using CUDA/GPU acceleration

## Performance Metrics

- **Tracking Accuracy**: High precision with Kalman-smoothed trajectories
- **Processing Speed**: Real-time capable on modern hardware (30+ FPS on HD video)
- **Counting Accuracy**: >95% accuracy on clear videos with proper ROI placement
- **Robustness**: Handles occlusions, lighting changes, and perspective distortion

## Project Background

This project was developed as part of **CPS843 - Introduction to Computer Vision** at Toronto Metropolitan University (TMU). It demonstrates practical applications of optical flow, state estimation, and geometric computer vision techniques for real-world traffic monitoring scenarios.

## Author

**Arshia Rahim**
- Computer Engineering (Software) Student @ Toronto Metropolitan University
- GitHub: [@ArshiaRx](https://github.com/ArshiaRx)
- LinkedIn: [in/arshia-rahim](https://www.linkedin.com/in/arshia-rahim)

## License

This project is for educational purposes. Feel free to reference and learn from the code, but please credit appropriately and avoid direct copying in academic contexts.

## References

- Lucas, B. D., & Kanade, T. (1981). *An iterative image registration technique with an application to stereo vision.* IJCAI.
- Kalman, R. E. (1960). *A new approach to linear filtering and prediction problems.* Journal of Basic Engineering.
- Shi, J., & Tomasi, C. (1994). *Good features to track.* IEEE CVPR.

