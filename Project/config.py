"""
Configuration parameters for Vehicle Tracking and Counting System.
Centralized configuration for easy adjustment and maintenance.
"""

# YOLO Detection Configuration
YOLO_MODEL_SIZE = 'n'  # Options: 'n', 's', 'm', 'l', 'x' (nano, small, medium, large, xlarge)
YOLO_CONFIDENCE_THRESHOLD = 0.4  # Default confidence threshold (0.0-1.0)
YOLO_MIN_BOX_SIZE = 20  # Minimum bounding box size in pixels

# Vehicle Classes (COCO dataset class IDs)
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# Optical Flow Configuration
OPTICAL_FLOW_WIN_SIZE = (15, 15)  # Window size for Lucas-Kanade
OPTICAL_FLOW_MAX_LEVEL = 2  # Pyramid levels
OPTICAL_FLOW_MAX_CORNERS = 500  # Maximum features to track
OPTICAL_FLOW_QUALITY_LEVEL = 0.01  # Feature quality threshold
OPTICAL_FLOW_MIN_DISTANCE = 10  # Minimum distance between features

# Kalman Filter Configuration
KALMAN_DT = 1.0  # Time step (frame-to-frame)
KALMAN_PROCESS_NOISE = 0.03  # Process noise covariance
KALMAN_MEASUREMENT_NOISE = 0.1  # Measurement noise covariance

# Tracking Configuration
MAX_TRACK_LENGTH = 30  # Maximum points in track history
MIN_TRACK_LENGTH = 5  # Minimum track length to consider valid
MAX_LOST_FRAMES = 5  # Frames before marking track as inactive

# ROI Configuration
DEFAULT_ROI_TYPE = 'line'  # Options: 'line', 'polygon'
DEFAULT_DIRECTION_LABELS = ('Up', 'Down')  # Direction labels for counting

# Display Configuration
DEFAULT_DISPLAY_MODE = 'clean'  # Options: 'clean', 'verbose', 'minimal'
SHOW_TRACKS_DEFAULT = True
SHOW_LEGEND_DEFAULT = False
SHOW_HELP_DEFAULT = True

# Video Processing Configuration
DEFAULT_PLAYBACK_SPEED = 1.0  # Playback speed multiplier
AUTO_SAVE_OUTPUT = True  # Automatically save output videos
OUTPUT_DIR = 'output'  # Default output directory

# Performance Configuration
TARGET_FPS = 15  # Target frames per second for real-time processing
MULTI_SCALE_DETECTION = True  # Enable multi-scale detection for YOLO
MULTI_SCALE_FACTORS = [0.8, 1.0, 1.2]  # Scale factors for multi-scale detection

# Counting Configuration
CROSSING_ANIMATION_DURATION = 1.0  # Duration of crossing animation in seconds
RECENT_CROSSINGS_MAX_AGE = 3.0  # Maximum age for recent crossings display (seconds)
RECENT_CROSSINGS_DISPLAY_COUNT = 5  # Number of recent crossings to display

# File Paths
DATA_DIR = 'data'  # Directory for input videos
MODEL_PATH = 'yolov8n.pt'  # Path to YOLO model weights

