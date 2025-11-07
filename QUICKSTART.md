# Quick Start Guide

## Installation

1. Navigate to the Project directory:
```bash
cd Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Process a Video File

Place your video file in the `data/` directory, then run:

```bash
python main.py --video data/your_video.mp4
```

The output will be automatically saved to `output/your_video_tracked.mp4`

### Use Webcam

```bash
python main.py --webcam
```

## ROI Selection

When you start the application:

1. **For Line ROI**: Click two points to define a counting line
   - Vehicles crossing from one side to the other will be counted
   - Press 'q' to confirm, 'r' to reset, ESC to cancel

2. **For Polygon ROI**: Click multiple points (at least 3) to define a region
   - Vehicles entering/exiting the region will be counted
   - Press 'q' to confirm, 'r' to reset, ESC to cancel

## Tips

- **Better tracking**: Ensure good lighting and clear video
- **ROI placement**: Place the ROI where vehicles clearly cross
- **Kalman filter**: Enabled by default for smoother tracking (use `--no-kalman` to disable)
- **Performance**: Processing speed depends on video resolution and frame rate

## Troubleshooting

If you get import errors, make sure all packages are installed:
```bash
pip install opencv-python numpy scipy matplotlib filterpy
```

If tracks are not detected, try:
- Using a video with more motion
- Adjusting the ROI position
- Checking video quality (not too dark/blurry)

