# Real-Time Vehicle Tracking with Optical Flow

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)

A video tracking system that uses optical flow to track vehicles in real-time. It includes Kalman filtering for smoother tracking and can count vehicles crossing a line or area you define in the video.

## Demo

Check out the sample tracked video in `output/sample_traffic_test_tracked.mp4` to see how it works.

## Features

- **Optical Flow Tracking**: Uses Lucas-Kanade optical flow to track features in the video
- **Kalman Filtering**: Smooths out the tracking using Kalman filters to reduce jitter
- **Vehicle Counting**: Counts vehicles that cross your defined region of interest
- **Flexible ROI**: Draw either a line or a polygon to define where to count vehicles
- **Real-time Processing**: Works with video files or live webcam feeds
- **Visualization**: Shows tracks with different colors, the ROI overlay, and live statistics

## Technologies

- OpenCV for computer vision and optical flow
- FilterPy for Kalman filtering
- NumPy and SciPy for calculations
- Matplotlib for visualization

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

## How It Works

### Optical Flow Tracking

The system uses Lucas-Kanade optical flow to track movement in the video. It detects good features to track (corners and edges) using Shi-Tomasi corner detection, then follows those features frame by frame. The tracks are managed automatically - new ones start when features are detected, and old ones end when features are lost.

### Kalman Filtering

Each tracked object gets its own Kalman filter that predicts where it will be in the next frame based on position and velocity. This smooths out the tracking and handles noise from the optical flow. The filter uses a constant velocity model and gets updated with each new observation.

### Vehicle Counting

For counting, you can draw either a line or a polygon on the video. The system checks when tracked objects cross this boundary:

- **Line ROI**: Detects when a track crosses from one side of the line to the other
- **Polygon ROI**: Detects when a track enters or exits the polygon area
- Direction is tracked too, so you can count vehicles going in different directions separately

## Tweaking Parameters

If you want to adjust how the tracking works, you can modify these in the source code:

- Optical flow settings: `lk_params` and `feature_params` in `OpticalFlowTracker`
- Kalman filter noise parameters: in `ObjectKalmanFilter.__init__()`
- Track length settings: `max_track_length` and `min_track_length` in `OpticalFlowTracker`

## Troubleshooting

**No tracks showing up?**

- Make sure there's actually movement in the video
- Try lowering `qualityLevel` in `feature_params` to detect more features
- Check if the video is too dark or blurry

**Tracks look jittery?**

- Kalman filtering should be on by default - if you disabled it, turn it back on
- Try increasing `winSize` in `lk_params` for smoother tracking

**Counting not working?**

- Make sure you drew the ROI correctly
- Check that vehicles actually cross through your line/polygon
- For line counting, works best when vehicles move across the line (not parallel to it)

## Known Issues and Limitations

### Similar Colors

The optical flow tracker can struggle when vehicles have similar colors to each other or to the road (especially gray vehicles on gray pavement). This happens because optical flow needs to detect corners and edges, which are harder to find when there's low contrast. If you're having this issue, try using YOLO mode with `--yolo` flag - it's better at detecting vehicles regardless of color.

### Crowded Traffic

When vehicles are too close together or overlapping, the system might:

- Merge multiple vehicles into one track
- Lose track of vehicles that get blocked
- Undercount in heavy traffic

This happens because the optical flow features can blend together. YOLO mode helps with this too, or you can try positioning the counting line where vehicles are more spread out.

### YOLO Mode vs Optical Flow Mode

**Optical Flow Mode** (default):

- Pros: Fast, no ML model needed, works well for separated moving objects
- Cons: Struggles with similar colors, may miss slow vehicles, needs visible texture/edges

**YOLO Mode** (`--yolo` flag):

- Pros: Better at detecting all vehicles regardless of color, handles crowded scenes better, can detect stopped vehicles
- Cons: Slower, needs the ultralytics package installed

For videos with lots of traffic or vehicles that are similar colors, YOLO mode works better:

```bash
python main.py --video data/traffic.mp4 --yolo --confidence 0.5 --min-box-size 30
```

### Direction Labels

The system labels crossing directions as "Up" and "Down" based on which side of the line the vehicle crosses. These are relative to how you draw the line, not geographic directions:

- **"Up"**: Crossing from one side to the other (shown with blue ▲ arrow when you draw the ROI)
- **"Down"**: Crossing the opposite way (shown with orange ▼ arrow)

If the directions seem backwards, just redraw the line the other way. You can also use custom labels like `--direction-up "Northbound"` to make it clearer.

### Colors

When you're watching the tracking:

- **Green boxes**: Vehicle is being tracked normally
- **Yellow boxes**: Vehicle just crossed the counting line (within last 3 seconds)

Press 'L' while the video is playing to see the legend with all the colors and symbols.

### Performance Tips

The tracking works best with:

- 30+ FPS video (lower frame rates make tracking harder)
- 1080p or higher resolution
- Good lighting conditions
- Clear view of the area you're monitoring

If your video is blurry, low resolution, or has poor lighting, you might see lower accuracy.

## Future Ideas

- Add speed estimation using camera calibration
- Improve vehicle type classification (car, truck, motorcycle, etc.)
- Support for real-time video streams (RTSP)
- Database integration for storing historical data
- GPU acceleration for better performance

## About This Project

This was created for CPS843 - Introduction to Computer Vision at Toronto Metropolitan University. It's a practical application of optical flow, Kalman filtering, and computer vision techniques for tracking and counting vehicles.

## Author

**Arshia Rahim**

- Computer Engineering (Software) Student @ Toronto Metropolitan University
- GitHub: [@ArshiaRx](https://github.com/ArshiaRx)
- LinkedIn: [in/arshia-rahim](https://www.linkedin.com/in/arshia-rahim)

## License

Educational project - feel free to learn from it and reference it, but please don't copy it directly for your own coursework.

## References

I used these papers as references for the algorithms:

- Lucas & Kanade (1981) - Lucas-Kanade optical flow
- Kalman (1960) - Kalman filtering
- Shi & Tomasi (1994) - Good features to track
