"""
Utility functions for visualization, ROI selection, and configuration.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


class ROISelector:
    """Interactive ROI selector using mouse callbacks."""
    
    def __init__(self, window_name='Select ROI'):
        self.window_name = window_name
        self.roi_points = []
        self.roi_type = 'line'  # 'line' or 'polygon'
        self.drawing = False
        self.current_point = None
        self.image = None
        self.image_copy = None
        
    def select_line_roi(self, image):
        """
        Select a line ROI by clicking two points.
        
        Args:
            image: Input image
            
        Returns:
            list: [(x1, y1), (x2, y2)] or None if cancelled
        """
        self.image = image.copy()
        self.image_copy = image.copy()
        self.roi_points = []
        self.roi_type = 'line'
        self.drawing = False
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback_line)
        
        print("Click two points to define the counting line. Press 'q' to confirm, 'r' to reset, 'ESC' to cancel.")
        
        while True:
            display_img = self.image_copy.copy()
            
            # Draw current line if we have points
            if len(self.roi_points) == 1:
                cv2.circle(display_img, self.roi_points[0], 5, (0, 255, 0), -1)
                if self.current_point:
                    cv2.line(display_img, self.roi_points[0], self.current_point, (0, 255, 0), 2)
            elif len(self.roi_points) == 2:
                # Use the draw_roi function to show directional arrows
                display_img = draw_roi(display_img, self.roi_points, roi_type='line', 
                                      highlight=False, show_direction=True)
            
            cv2.imshow(self.window_name, display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') and len(self.roi_points) == 2:
                cv2.destroyWindow(self.window_name)
                return self.roi_points
            elif key == ord('r'):
                self.roi_points = []
                self.image_copy = self.image.copy()
            elif key == 27:  # ESC
                cv2.destroyWindow(self.window_name)
                return None
    
    def select_polygon_roi(self, image):
        """
        Select a polygon ROI by clicking multiple points.
        
        Args:
            image: Input image
            
        Returns:
            list: [(x1, y1), (x2, y2), ...] or None if cancelled
        """
        self.image = image.copy()
        self.image_copy = image.copy()
        self.roi_points = []
        self.roi_type = 'polygon'
        self.drawing = False
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback_polygon)
        
        print("Click points to define polygon. Press 'q' to confirm, 'r' to reset, 'ESC' to cancel.")
        
        while True:
            display_img = self.image_copy.copy()
            
            # Draw current polygon
            if len(self.roi_points) > 1:
                pts = np.array(self.roi_points, np.int32)
                cv2.polylines(display_img, [pts], True, (0, 255, 0), 2)
                for pt in self.roi_points:
                    cv2.circle(display_img, pt, 5, (0, 255, 0), -1)
            
            cv2.imshow(self.window_name, display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') and len(self.roi_points) >= 3:
                cv2.destroyWindow(self.window_name)
                return self.roi_points
            elif key == ord('r'):
                self.roi_points = []
                self.image_copy = self.image.copy()
            elif key == 27:  # ESC
                cv2.destroyWindow(self.window_name)
                return None
    
    def _mouse_callback_line(self, event, x, y, flags, param):
        """Mouse callback for line selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.roi_points) < 2:
                self.roi_points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if len(self.roi_points) == 1:
                self.current_point = (x, y)
    
    def _mouse_callback_polygon(self, event, x, y, flags, param):
        """Mouse callback for polygon selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_points.append((x, y))


def draw_tracks(frame, tracks, max_history=10, roi_points=None, display_mode='clean', 
                max_distance=100, recent_crossing_ids=None):
    """
    Draw tracking trails on frame.
    
    Args:
        frame: Input frame
        tracks: List of track dictionaries with 'id', 'position', 'history'
        max_history: Maximum number of history points to draw
        roi_points: ROI points for distance filtering
        display_mode: 'clean', 'verbose', or 'minimal'
        max_distance: Maximum distance from ROI to show tracks (in clean mode)
        recent_crossing_ids: Set of track IDs that recently crossed
        
    Returns:
        frame: Frame with tracks drawn
    """
    if display_mode == 'minimal':
        return frame
    
    if recent_crossing_ids is None:
        recent_crossing_ids = set()
    
    for track in tracks:
        track_id = track.get('id', -1)
        history = track.get('history', [])
        current_pos = track.get('position', None)
        
        if current_pos is None:
            continue
        
        if isinstance(current_pos, np.ndarray):
            x, y = int(current_pos[0]), int(current_pos[1])
        else:
            x, y = int(current_pos[0]), int(current_pos[1])
        
        # Distance filtering in clean mode
        if display_mode == 'clean' and roi_points:
            from .vehicle_counter import VehicleCounter
            counter_temp = VehicleCounter(roi_type='line', roi_points=roi_points)
            dist = counter_temp.get_distance_to_roi((x, y))
            if dist > max_distance and track_id not in recent_crossing_ids:
                continue
        
        # Draw track history (trail)
        if len(history) > 1:
            points = np.array(history[-max_history:], dtype=np.int32)
            line_thickness = 2 if display_mode == 'verbose' else 1
            color = (0, 255, 255) if track_id in recent_crossing_ids else (0, 255, 0)
            for i in range(len(points) - 1):
                cv2.line(frame, tuple(points[i]), tuple(points[i+1]), color, line_thickness)
        
        # Draw current position
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
        
        # Show ID only in verbose mode or for recent crossings
        show_id = (display_mode == 'verbose') or (track_id in recent_crossing_ids)
        
        if show_id:
            text = f"ID{track_id}"
            class_name = track.get('class_name', '')
            if class_name:
                text = f"{text} {class_name}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle for better visibility
            text_x = x + 10
            text_y = y - 8
            cv2.rectangle(frame, 
                         (text_x - 3, text_y - text_height - 3),
                         (text_x + text_width + 3, text_y + baseline + 3),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, text, (text_x, text_y), 
                       font, font_scale, (255, 255, 255), thickness)
    
    return frame


def draw_roi(frame, roi_points=None, roi_lines=None, roi_type='line', highlight=False, show_direction=True, use_double_line=False):
    """
    Draw ROI on frame with optional highlighting and directional arrows.
    Supports multiple lines for line ROI type and double-line counting visualization.
    
    Args:
        frame: Input frame
        roi_points: ROI points (legacy support for single ROI)
        roi_lines: List of ROI lines (for multiple lines support)
        roi_type: 'line' or 'polygon'
        highlight: Whether to draw highlighted (thicker, brighter)
        show_direction: Whether to show directional arrows (for line ROI)
        use_double_line: Whether this is double-line counting mode
        
    Returns:
        frame: Frame with ROI drawn
    """
    # Determine which lines to draw
    lines_to_draw = []
    if roi_type == 'line':
        if roi_lines is not None:
            # Multiple lines support
            if isinstance(roi_lines, list) and len(roi_lines) > 0:
                if isinstance(roi_lines[0][0], (list, tuple)):
                    # List of lines: [[(x1,y1), (x2,y2)], ...]
                    lines_to_draw = roi_lines
                else:
                    # Single line as flat list: [(x1,y1), (x2,y2)]
                    lines_to_draw = [roi_lines]
        elif roi_points is not None and len(roi_points) == 2:
            # Legacy single line support
            lines_to_draw = [roi_points]
    
    if roi_type == 'line' and len(lines_to_draw) == 0:
        return frame
    
    # Adjust appearance based on highlight - make lines thicker and more visible
    if highlight:
        base_color = (0, 255, 128)  # Brighter green
        thickness = 6
        circle_radius = 12
    else:
        base_color = (0, 200, 0)  # Darker green
        thickness = 5  # Increased from 3 to 5 for better visibility
        circle_radius = 10  # Increased from 8 to 10
    
    # Define colors for multiple lines - special colors for double-line mode
    if use_double_line and len(lines_to_draw) == 2:
        line_colors = [
            (0, 255, 255),  # Bright Cyan for Line 1
            (255, 215, 0),  # Bright Gold/Yellow for Line 2
        ]
    else:
        line_colors = [
            (0, 255, 0),    # Bright Green for single line
            (0, 165, 255),   # Orange
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Yellow
            (255, 0, 0),    # Blue
        ]
    
    if roi_type == 'line':
        # Draw all lines
        for line_index, line in enumerate(lines_to_draw):
            if len(line) != 2:
                continue
            
            # Use different color for each line
            color = line_colors[line_index % len(line_colors)]
            if highlight:
                color = tuple(min(255, c + 50) for c in color)  # Brighten for highlight
            
            x1, y1 = line[0]
            x2, y2 = line[1]
            
            # Draw line with thicker stroke for better visibility
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
            # Draw line again with slightly offset to create glow effect
            cv2.line(frame, (x1, y1), (x2, y2), tuple(min(255, c + 30) for c in color), max(1, thickness - 2))
            cv2.circle(frame, (x1, y1), circle_radius, color, -1)
            cv2.circle(frame, (x2, y2), circle_radius, color, -1)
            
            # Draw line label - always show for better visibility
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            if use_double_line and len(lines_to_draw) == 2:
                label = f"Line {line_index + 1}"
                # Make labels more prominent with background
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)
                cv2.rectangle(frame, (mid_x - text_width // 2 - 5, mid_y - text_height - 5), 
                             (mid_x + text_width // 2 + 5, mid_y + baseline + 5), (0, 0, 0), -1)
                cv2.putText(frame, label, (mid_x - text_width // 2, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
            elif len(lines_to_draw) > 1:
                label = f"Line {line_index + 1}"
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (mid_x - text_width // 2 - 5, mid_y - text_height - 5), 
                             (mid_x + text_width // 2 + 5, mid_y + baseline + 5), (0, 0, 0), -1)
                cv2.putText(frame, label, (mid_x - text_width // 2, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw directional arrows to show which side is "up" vs "down"
            if show_direction:
                # Calculate perpendicular vector to the line
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    # Normalize
                    dx /= length
                    dy /= length
                    
                    # Perpendicular vectors (both sides)
                    perp_x = -dy
                    perp_y = dx
                    
                    # Midpoint of line
                    mid_x = (x1 + x2) // 2
                    mid_y = (y1 + y2) // 2
                    
                    # Offset distance for arrows (adjust for multiple lines)
                    offset = 50 + (line_index * 20)  # Stagger arrows for multiple lines
                    
                    # Define consistent colors for North/South directions per line
                    if use_double_line and len(lines_to_draw) == 2:
                        # Line 1: North = Cyan, South = Orange
                        # Line 2: North = Green, South = Magenta
                        if line_index == 0:
                            north_color = (255, 255, 0)  # Cyan (BGR)
                            south_color = (0, 165, 255)  # Orange (BGR)
                            north_label = "N"
                            south_label = "S"
                        else:  # line_index == 1
                            north_color = (0, 255, 0)  # Green (BGR)
                            south_color = (255, 0, 255)  # Magenta (BGR)
                            north_label = "N"
                            south_label = "S"
                    else:
                        # Single line: Use default colors
                        north_color = (150, 220, 255)  # Light blue
                        south_color = (100, 180, 255)  # Darker blue
                        north_label = "UP"
                        south_label = "DOWN"
                    
                    # Draw arrows on both sides
                    # Side 1 (positive side - "North" direction)
                    arrow_start_1 = (int(mid_x + perp_x * offset), int(mid_y + perp_y * offset))
                    arrow_end_1 = (int(mid_x + perp_x * (offset + 30)), int(mid_y + perp_y * (offset + 30)))
                    cv2.arrowedLine(frame, arrow_start_1, arrow_end_1, north_color, 3, tipLength=0.4)
                    cv2.putText(frame, north_label, (int(mid_x + perp_x * (offset + 40)), int(mid_y + perp_y * (offset + 40))),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, north_color, 2)
                    
                    # Side 2 (negative side - "South" direction)
                    arrow_start_2 = (int(mid_x - perp_x * offset), int(mid_y - perp_y * offset))
                    arrow_end_2 = (int(mid_x - perp_x * (offset + 30)), int(mid_y - perp_y * (offset + 30)))
                    cv2.arrowedLine(frame, arrow_start_2, arrow_end_2, south_color, 3, tipLength=0.4)
                    cv2.putText(frame, south_label, (int(mid_x - perp_x * (offset + 45)), int(mid_y - perp_y * (offset + 40))),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, south_color, 2)
    
    elif roi_type == 'polygon' and roi_points is not None and len(roi_points) >= 3:
        color = base_color
        pts = np.array(roi_points, np.int32)
        cv2.polylines(frame, [pts], True, color, thickness)
        for pt in roi_points:
            cv2.circle(frame, pt, circle_radius, color, -1)
    
    return frame


def draw_counts(frame, counts, position=(15, 40)):
    """
    Draw count statistics on frame.
    
    Args:
        frame: Input frame
        counts: Dictionary with 'total', 'up', 'down' counts
        position: (x, y) position to draw counts
        
    Returns:
        frame: Frame with counts drawn
    """
    if not counts:
        return frame
    
    total = counts.get('total', 0)
    up = counts.get('up', 0)
    down = counts.get('down', 0)
    
    x, y = position
    
    # Background rectangle for better visibility
    cv2.rectangle(frame, (x - 10, y - 30), (x + 180, y + 80), (0, 0, 0), -1)
    cv2.rectangle(frame, (x - 10, y - 30), (x + 180, y + 80), (0, 255, 0), 2)
    
    # Draw counts
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    cv2.putText(frame, f"Total: {total}", (x, y), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(frame, f"Up: {up}", (x, y + 25), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(frame, f"Down: {down}", (x, y + 50), font, font_scale, (0, 165, 255), thickness)
    
    return frame


def draw_kalman_predictions(frame, kalman_tracks, roi_points=None, display_mode='clean', max_distance=150):
    """
    Draw Kalman filter predictions on frame.
    
    Args:
        frame: Input frame
        kalman_tracks: Dictionary of track_id -> KalmanFilter objects
        roi_points: ROI points for distance filtering
        display_mode: 'clean', 'verbose', or 'minimal'
        max_distance: Maximum distance from ROI to show (in clean mode)
        
    Returns:
        frame: Frame with Kalman predictions drawn
    """
    if display_mode == 'minimal':
        return frame
    
    for track_id, kf in kalman_tracks.items():
        # Get predicted position
        x = int(kf.x[0])
        y = int(kf.x[1])
        
        # Distance filtering in clean mode
        if display_mode == 'clean' and roi_points:
            from .vehicle_counter import VehicleCounter
            counter_temp = VehicleCounter(roi_type='line', roi_points=roi_points)
            dist = counter_temp.get_distance_to_roi((x, y))
            if dist > max_distance:
                continue
        
        # Draw predicted position
        cv2.circle(frame, (x, y), 6, (255, 0, 255), -1)  # Magenta for predictions
        
        # Draw uncertainty ellipse if available
        if hasattr(kf, 'P') and kf.P is not None:
            # Extract position covariance
            cov = kf.P[:2, :2]
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            
            # Draw ellipse
            axes_lengths = (int(np.sqrt(eigenvals[0]) * 3), int(np.sqrt(eigenvals[1]) * 3))
            cv2.ellipse(frame, (x, y), axes_lengths, angle, 0, 360, (255, 0, 255), 1)
        
        # Show ID in verbose mode
        if display_mode == 'verbose':
            text = f"KF{track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            cv2.putText(frame, text, (x + 10, y - 10), font, font_scale, (255, 0, 255), thickness)
    
    return frame


def draw_crossing_event(frame, animations):
    """
    Draw crossing event animations (flash, ripple effects) and +1 indicators.
    
    Args:
        frame: Input frame
        animations: List of active animations with 'position', 'progress', 'direction'
        
    Returns:
        frame: Frame with animations drawn
    """
    for anim in animations:
        position = anim.get('position', None)
        if position is None:
            continue
        
        progress = anim.get('progress', 0.0)
        direction = anim.get('direction', 'unknown')
        
        if isinstance(position, np.ndarray):
            x, y = int(position[0]), int(position[1])
        else:
            x, y = int(position[0]), int(position[1])
        
        # Expanding circle (ripple effect)
        radius = int(20 + progress * 40)  # 20 to 60 pixels
        alpha = 1.0 - progress  # Fade out
        
        # Color based on direction
        if direction in ['up', 'enter']:
            color = (0, 255, 0)  # Green
        else:
            color = (0, 165, 255)  # Orange
        
        # Draw expanding circle
        thickness = max(1, int(4 * (1 - progress)))
        cv2.circle(frame, (x, y), radius, color, thickness)
        
        # Draw "+1" text that floats up and fades
        if progress < 0.7:
            text = "+1"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            
            # Text position floats upward
            text_y = y - int(30 + progress * 50)
            text_x = x - 15
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay,
                         (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + baseline + 5),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw text
            cv2.putText(frame, text, (text_x, text_y),
                       font, font_scale, color, thickness)
    
    return frame


def draw_yolo_detections(frame, detections, roi_points=None, display_mode='clean',
                         max_distance=150, recent_crossing_ids=None):
    """
    Draw YOLO detections with bounding boxes and labels.
    
    Args:
        frame: Input frame
        detections: List of YOLO detections/tracks
        roi_points: ROI points for distance filtering
        display_mode: 'clean', 'verbose', or 'minimal'
        max_distance: Maximum distance from ROI to show (in clean mode)
        recent_crossing_ids: Set of track IDs that recently crossed
        
    Returns:
        frame: Frame with detections drawn
    """
    if display_mode == 'minimal':
        return frame
    
    if recent_crossing_ids is None:
        recent_crossing_ids = set()
    
    for det in detections:
        track_id = det.get('id', -1)
        bbox = det.get('bbox', None)
        if bbox is None:
            continue
        
        x1, y1, x2, y2 = bbox
        center = det.get('position', None)
        class_name = det.get('class_name', 'vehicle')
        confidence = det.get('confidence', 0.0)
        
        # Distance filtering in clean mode
        if display_mode == 'clean' and roi_points and center is not None:
            from .vehicle_counter import VehicleCounter
            counter_temp = VehicleCounter(roi_type='line', roi_points=roi_points)
            dist = counter_temp.get_distance_to_roi((center[0], center[1]))
            if dist > max_distance and track_id not in recent_crossing_ids:
                continue
        
        # Draw bounding box
        color = (0, 255, 0)  # Green for vehicles
        if track_id in recent_crossing_ids:
            color = (0, 255, 255)  # Yellow for recently crossed
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if display_mode == 'verbose' or track_id in recent_crossing_ids:
            label = f"{class_name}"
            if display_mode == 'verbose':
                label = f"ID{track_id} {label} {confidence:.2f}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(frame, (x1, y1 - text_height - 5), 
                         (x1 + text_width + 5, y1), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, label, (x1 + 2, y1 - 5), font, font_scale, color, thickness)
    
    return frame
