"""
Automatic ROI line detection using YOLO vehicle movement analysis.

This module provides functionality to automatically detect optimal counting lines
by analyzing vehicle movement patterns across video frames.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def auto_detect_roi_line(video_path: str, yolo_detector, num_sample_frames: int = 50) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Automatically detect optimal counting line by analyzing vehicle movement patterns.
    
    This function samples frames from the video, uses YOLO to track vehicles,
    and finds the Y-coordinate where most vehicle activity occurs. Returns a
    horizontal line spanning the full width at that optimal Y-coordinate.
    
    Args:
        video_path: Path to the input video file
        yolo_detector: YOLODetector instance for vehicle detection and tracking
        num_sample_frames: Number of frames to sample for analysis (default: 50)
        
    Returns:
        Tuple of two points representing the counting line: [(x1, y1), (x2, y2)]
        Returns None if detection fails (video error, YOLO unavailable, insufficient detections)
    """
    # Check if YOLO is available
    if not yolo_detector or not yolo_detector.available:
        return None
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames == 0 or width == 0 or height == 0:
            return None
        
        # Calculate frame sampling interval
        # Sample evenly across the video, but skip first 10% and last 10% to avoid edge effects
        start_frame = int(total_frames * 0.1)
        end_frame = int(total_frames * 0.9)
        frames_to_sample = min(num_sample_frames, end_frame - start_frame)
        
        if frames_to_sample <= 0:
            return None
        
        frame_interval = max(1, (end_frame - start_frame) // frames_to_sample)
        
        # Collect vehicle positions from sampled frames
        vehicle_y_positions = []
        frames_processed = 0
        
        # Reset YOLO detector to clear any previous tracking state
        yolo_detector.reset()
        
        for frame_idx in range(start_frame, end_frame, frame_interval):
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Track vehicles in this frame
            tracks = yolo_detector.track(frame)
            
            # Collect Y-coordinates of vehicle centers
            for track in tracks:
                position = track.get('position')
                if position is not None:
                    # position is a numpy array [x, y] or tuple-like
                    y_pos = int(position[1])
                    
                    # Only consider positions within frame bounds
                    if 0 <= y_pos < height:
                        vehicle_y_positions.append(y_pos)
            
            frames_processed += 1
            
            # Early exit if we've processed enough frames
            if frames_processed >= num_sample_frames:
                break
        
        # Check if we have enough vehicle detections
        if len(vehicle_y_positions) < 10:  # Need at least 10 detections
            return None
        
        # Find optimal Y-coordinate using histogram analysis
        # Create histogram of Y-positions with bins
        hist, bins = np.histogram(vehicle_y_positions, bins=min(50, height // 20))
        
        # Find bin with maximum count (most vehicle activity)
        max_bin_idx = np.argmax(hist)
        optimal_y = int((bins[max_bin_idx] + bins[max_bin_idx + 1]) / 2)
        
        # Ensure optimal_y is within frame bounds
        optimal_y = max(10, min(optimal_y, height - 10))
        
        # Return horizontal line spanning full width (with small margins)
        margin = int(width * 0.05)  # 5% margin on each side
        x1 = margin
        x2 = width - margin
        
        return ((x1, optimal_y), (x2, optimal_y))
        
    except Exception as e:
        # Return None on any error
        return None
    finally:
        cap.release()

