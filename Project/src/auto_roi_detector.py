"""
Automatic ROI line detection using YOLO vehicle movement analysis.

This module provides functionality to automatically detect optimal counting lines
by analyzing vehicle movement patterns across video frames.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


def auto_detect_roi_line(video_path: str, yolo_detector, num_sample_frames: int = 100) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Automatically detect optimal counting line by analyzing vehicle movement patterns.
    
    This function samples frames from the video, uses YOLO to track vehicles,
    and finds the Y-coordinate where most vehicle activity occurs. Returns a
    horizontal line spanning the full width at that optimal Y-coordinate.
    Enhanced to better capture all vehicles by analyzing density and movement patterns.
    
    Args:
        video_path: Path to the input video file
        yolo_detector: YOLODetector instance for vehicle detection and tracking
        num_sample_frames: Number of frames to sample for analysis (default: 100, increased for better coverage)
        
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
        
        # Collect vehicle positions and create density heatmap
        vehicle_y_positions = []
        vehicle_x_positions = []
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
            
            # Collect X,Y-coordinates of vehicle centers for better analysis
            for track in tracks:
                position = track.get('position')
                if position is not None:
                    # position is a numpy array [x, y] or tuple-like
                    x_pos = int(position[0])
                    y_pos = int(position[1])
                    
                    # Only consider positions within frame bounds
                    if 0 <= y_pos < height and 0 <= x_pos < width:
                        vehicle_y_positions.append(y_pos)
                        vehicle_x_positions.append(x_pos)
            
            frames_processed += 1
            
            # Early exit if we've processed enough frames
            if frames_processed >= num_sample_frames:
                break
        
        # Check if we have enough vehicle detections
        if len(vehicle_y_positions) < 10:  # Need at least 10 detections
            return None
        
        # Find optimal Y-coordinate using enhanced histogram analysis
        # Use more bins for finer granularity
        num_bins = min(80, height // 10)
        hist, bins = np.histogram(vehicle_y_positions, bins=num_bins)
        
        # Apply smoothing to histogram to reduce noise
        if len(hist) >= 5:
            from scipy.ndimage import gaussian_filter1d
            try:
                hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)
            except ImportError:
                # Fallback: simple moving average if scipy not available
                kernel_size = 5
                hist_smooth = np.convolve(hist, np.ones(kernel_size)/kernel_size, mode='same')
        else:
            hist_smooth = hist
        
        # Find bin with maximum count (most vehicle activity)
        max_bin_idx = np.argmax(hist_smooth)
        optimal_y = int((bins[max_bin_idx] + bins[max_bin_idx + 1]) / 2)
        
        # Validate that optimal line covers a good distribution of vehicles
        # Check if vehicles are spread across X-axis at this Y position
        y_tolerance = height // 20  # Within 5% of frame height
        nearby_x_positions = [x for x, y in zip(vehicle_x_positions, vehicle_y_positions) 
                             if abs(y - optimal_y) <= y_tolerance]
        
        # If not enough X-spread, find the median Y instead for better coverage
        if len(nearby_x_positions) < 5:
            optimal_y = int(np.median(vehicle_y_positions))
        
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


def estimate_line_spacing(video_path: str, base_line: List[Tuple[int, int]], 
                          yolo_detector, num_sample_frames: int = 50) -> int:
    """
    Estimate optimal spacing between two parallel counting lines based on vehicle size.
    
    This function samples frames from the video, detects vehicles near the base line,
    and calculates a representative vehicle size to determine appropriate line spacing.
    
    Args:
        video_path: Path to the input video file
        base_line: Base ROI line as [(x1, y1), (x2, y2)]
        yolo_detector: YOLODetector instance for vehicle detection
        num_sample_frames: Number of frames to sample for analysis (default: 50)
        
    Returns:
        Estimated spacing in pixels (clamped between 40-220), or fallback value if detection fails
    """
    # Check if YOLO is available
    if not yolo_detector or not yolo_detector.available:
        return 80  # Fallback spacing
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 80  # Fallback spacing
    
    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames == 0 or height == 0:
            return max(50, int(height * 0.04))  # Fallback to 4% of frame height
        
        # Calculate frame sampling interval
        start_frame = int(total_frames * 0.1)
        end_frame = int(total_frames * 0.9)
        frames_to_sample = min(num_sample_frames, end_frame - start_frame)
        
        if frames_to_sample <= 0:
            return max(50, int(height * 0.04))
        
        frame_interval = max(1, (end_frame - start_frame) // frames_to_sample)
        
        # Extract base line points
        if len(base_line) != 2:
            return max(50, int(height * 0.04))
        
        p1, p2 = base_line[0], base_line[1]
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        
        # Calculate line equation: distance from point to line
        # For a line from (x1, y1) to (x2, y2), perpendicular distance formula
        dx = x2 - x1
        dy = y2 - y1
        line_length = np.sqrt(dx*dx + dy*dy)
        
        if line_length == 0:
            return max(50, int(height * 0.04))
        
        # Collect vehicle bounding box sizes near the line
        vehicle_heights = []
        vehicle_widths = []
        max_distance_to_line = height * 0.1  # Consider vehicles within 10% of frame height from line
        
        # Reset YOLO detector
        yolo_detector.reset()
        
        frames_processed = 0
        for frame_idx in range(start_frame, end_frame, frame_interval):
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Detect vehicles in this frame
            tracks = yolo_detector.track(frame)
            
            for track in tracks:
                bbox = track.get('bbox')
                position = track.get('position')
                
                if bbox is None or position is None:
                    continue
                
                x1_bbox, y1_bbox, x2_bbox, y2_bbox = bbox
                center_x, center_y = position[0], position[1]
                
                # Calculate distance from vehicle center to the base line
                # Using point-to-line distance formula
                # Distance = |(y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1| / sqrt((y2-y1)^2 + (x2-x1)^2)
                numerator = abs((y2 - y1) * center_x - (x2 - x1) * center_y + x2 * y1 - y2 * x1)
                distance_to_line = numerator / line_length if line_length > 0 else float('inf')
                
                # Only consider vehicles near the line
                if distance_to_line <= max_distance_to_line:
                    bbox_height = abs(y2_bbox - y1_bbox)
                    bbox_width = abs(x2_bbox - x1_bbox)
                    
                    # Use the larger dimension as representative size (accounts for different orientations)
                    vehicle_size = max(bbox_height, bbox_width)
                    
                    if vehicle_size > 10:  # Filter out tiny detections
                        vehicle_heights.append(bbox_height)
                        vehicle_widths.append(bbox_width)
            
            frames_processed += 1
            
            # Early exit if we've processed enough frames
            if frames_processed >= num_sample_frames:
                break
        
        # Calculate spacing based on vehicle sizes
        if len(vehicle_heights) >= 3:
            # Use median to be robust to outliers
            median_height = np.median(vehicle_heights)
            median_width = np.median(vehicle_widths)
            
            # Use average of height and width for more robust estimate
            representative_size = (median_height + median_width) / 2
            
            # Spacing should be 1.2-1.5x the representative vehicle size
            # This ensures vehicles can be clearly distinguished between lines
            spacing = int(1.3 * representative_size)
            
            # Clamp between reasonable bounds
            spacing = max(40, min(spacing, 220))
            
            return spacing
        else:
            # Not enough detections, use fallback based on frame height
            return max(50, int(height * 0.04))
            
    except Exception as e:
        # On error, return fallback spacing
        try:
            cap_temp = cv2.VideoCapture(video_path)
            if cap_temp.isOpened():
                h = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_temp.release()
                return max(50, int(h * 0.04))
        except:
            pass
        return 80  # Final fallback
    finally:
        cap.release()

