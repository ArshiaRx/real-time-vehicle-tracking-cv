"""
Utility functions for visualization, ROI selection, and configuration.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional


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
            list: List of (x, y) points or None if cancelled
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
            
            # Draw polygon
            if len(self.roi_points) > 0:
                pts = np.array(self.roi_points, np.int32)
                if len(self.roi_points) > 2:
                    cv2.fillPoly(display_img, [pts], (0, 255, 0, 100))
                cv2.polylines(display_img, [pts], len(self.roi_points) > 2, (0, 255, 0), 2)
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
        tracks: List of track dictionaries
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
    
    # Only show IDs for tracks with sufficient history (reduces clutter)
    min_track_length_for_display = 3
    
    if recent_crossing_ids is None:
        recent_crossing_ids = set()
    
    for track in tracks:
        track_id = track['id']
        history = track.get('history', [])
        current_pos = track['position']
        track_length = track.get('length', len(history))
        
        x, y = int(current_pos[0]), int(current_pos[1])
        
        # Distance filtering in clean mode
        if display_mode == 'clean' and roi_points:
            from .vehicle_counter import VehicleCounter
            counter_temp = VehicleCounter(roi_type='line', roi_points=roi_points)
            dist = counter_temp.get_distance_to_roi((x, y))
            if dist > max_distance and track_id not in recent_crossing_ids:
                continue
        
        # Draw track history (thinner in clean mode)
        if len(history) > 1:
            points = np.array(history[-max_history:], dtype=np.int32)
            line_thickness = 2 if display_mode == 'verbose' else 1
            for i in range(len(points) - 1):
                alpha = i / len(points)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(frame, tuple(points[i]), tuple(points[i+1]), color, line_thickness)
        
        # Draw current position
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
        
        # Show IDs only for recently crossed tracks or in verbose mode
        show_id = (display_mode == 'verbose' and track_length >= min_track_length_for_display) or \
                  (track_id in recent_crossing_ids)
        
        if show_id:
            # Include class name if available
            class_name = track.get('class_name', '')
            if class_name:
                text = f"ID:{track_id} {class_name}"
            else:
                text = f"ID:{track_id}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle for better visibility
            text_x = x + 10
            text_y = y - 8
            cv2.rectangle(frame, 
                         (text_x - 3, text_y - text_height - 3),
                         (text_x + text_width + 3, text_y + baseline + 3),
                         (0, 0, 0), -1)
            
            # Draw text with outline for better readability
            cv2.putText(frame, text, (text_x, text_y), 
                       font, font_scale, (255, 255, 255), thickness)
    
    return frame


def draw_roi(frame, roi_points, roi_type='line', highlight=False, show_direction=True):
    """
    Draw ROI on frame with optional highlighting and directional arrows.
    
    Args:
        frame: Input frame
        roi_points: ROI points
        roi_type: 'line' or 'polygon'
        highlight: Whether to draw highlighted (thicker, brighter)
        show_direction: Whether to show directional arrows (for line ROI)
        
    Returns:
        frame: Frame with ROI drawn
    """
    if not roi_points:
        return frame
    
    # Adjust appearance based on highlight
    if highlight:
        color = (0, 255, 128)  # Brighter green
        thickness = 5
        circle_radius = 10
    else:
        color = (0, 200, 0)  # Darker green
        thickness = 3
        circle_radius = 8
    
    if roi_type == 'line' and len(roi_points) == 2:
        cv2.line(frame, roi_points[0], roi_points[1], color, thickness)
        cv2.circle(frame, roi_points[0], circle_radius, color, -1)
        cv2.circle(frame, roi_points[1], circle_radius, color, -1)
        
        # Draw directional arrows to show which side is "up" vs "down"
        if show_direction:
            x1, y1 = roi_points[0]
            x2, y2 = roi_points[1]
            
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
                
                # Offset distance for arrows
                offset = 50
                
                # Draw arrows on both sides
                # Side 1 (positive side - "UP" direction)
                arrow_start_1 = (int(mid_x + perp_x * offset), int(mid_y + perp_y * offset))
                arrow_end_1 = (int(mid_x + perp_x * (offset + 30)), int(mid_y + perp_y * (offset + 30)))
                cv2.arrowedLine(frame, arrow_start_1, arrow_end_1, (150, 220, 255), 3, tipLength=0.4)
                cv2.putText(frame, "UP", (int(mid_x + perp_x * (offset + 40)), int(mid_y + perp_y * (offset + 40))),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 220, 255), 2)
                
                # Side 2 (negative side - "DOWN" direction)
                arrow_start_2 = (int(mid_x - perp_x * offset), int(mid_y - perp_y * offset))
                arrow_end_2 = (int(mid_x - perp_x * (offset + 30)), int(mid_y - perp_y * (offset + 30)))
                cv2.arrowedLine(frame, arrow_start_2, arrow_end_2, (100, 180, 255), 3, tipLength=0.4)
                cv2.putText(frame, "DOWN", (int(mid_x - perp_x * (offset + 45)), int(mid_y - perp_y * (offset + 40))),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 180, 255), 2)
                
    elif roi_type == 'polygon' and len(roi_points) >= 3:
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
        counts: Dictionary with count statistics
        position: (x, y) position for text
        
    Returns:
        frame: Frame with counts drawn
    """
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0  # Increased from 0.7
    thickness = 2
    line_spacing = 35  # Increased spacing between lines
    
    # Calculate background size
    text_lines = [
        f"Total: {counts.get('total', 0)}",
        f"Up: {counts.get('up', 0)}",
        f"Down: {counts.get('down', 0)}"
    ]
    
    max_width = 0
    for text in text_lines:
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        max_width = max(max_width, text_width)
    
    # Draw semi-transparent background rectangle
    bg_height = len(text_lines) * line_spacing + 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (x-10, y-30), 
                  (x + max_width + 15, y + bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (x-10, y-30), 
                  (x + max_width + 15, y + bg_height), (0, 255, 0), 2)
    
    # Draw text with better spacing
    cv2.putText(frame, text_lines[0], 
               (x, y), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(frame, text_lines[1], 
               (x, y + line_spacing), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(frame, text_lines[2], 
               (x, y + line_spacing * 2), font, font_scale, (0, 255, 0), thickness)
    
    return frame


def draw_kalman_predictions(frame, kalman_tracks, roi_points=None, display_mode='clean', max_distance=150):
    """
    Draw Kalman filter predictions on frame.
    
    Args:
        frame: Input frame
        kalman_tracks: Dictionary of track_id -> KalmanFilter objects
        roi_points: ROI points for distance filtering
        display_mode: 'clean', 'verbose', or 'minimal'
        max_distance: Maximum distance from ROI to show tracks
        
    Returns:
        frame: Frame with predictions drawn
    """
    if display_mode == 'minimal':
        return frame
        
    for track_id, kf in kalman_tracks.items():
        # Only show recently updated tracks
        if hasattr(kf, 'time_since_update') and kf.time_since_update >= 3:
            continue
            
        x, y = kf.get_position()
        
        # Distance filtering in clean mode
        if display_mode == 'clean' and roi_points:
            from .vehicle_counter import VehicleCounter
            counter_temp = VehicleCounter(roi_type='line', roi_points=roi_points)
            dist = counter_temp.get_distance_to_roi((x, y))
            if dist > max_distance:
                continue
        
        # Draw predicted position (larger, more visible)
        cv2.circle(frame, (x, y), 8, (255, 165, 0), 2)  # Orange circle
        
        # Draw velocity vector only in verbose mode
        if display_mode == 'verbose':
            vx, vy = kf.get_velocity()
            if abs(vx) > 0.1 or abs(vy) > 0.1:
                end_x = int(x + vx * 10)
                end_y = int(y + vy * 10)
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), (255, 165, 0), 2)
    
    return frame


def draw_crossing_event(frame, animations):
    """
    Draw crossing event animations (flash, ripple effects).
    
    Args:
        frame: Input frame
        animations: List of active animations with progress
        
    Returns:
        frame: Frame with animations drawn
    """
    for anim in animations:
        position = anim['position']
        progress = anim['progress']
        direction = anim['direction']
        
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


def draw_recent_crossings_panel(frame, recent_crossings):
    """
    Draw panel showing recent crossing events.
    
    Args:
        frame: Input frame
        recent_crossings: List of recent crossing events
        
    Returns:
        frame: Frame with panel drawn
    """
    if not recent_crossings:
        return frame
    
    # Panel position (top-right) - MUCH LARGER
    frame_height, frame_width = frame.shape[:2]
    panel_width = 380  # Increased from 280
    panel_x = frame_width - panel_width - 15
    panel_y = 15
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9  # Increased from 0.6
    thickness = 2
    line_height = 45  # Increased from 30
    
    # Calculate panel height
    panel_height = 60 + len(recent_crossings) * line_height + 20  # More padding
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Draw border (thicker)
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (0, 255, 128), 3)
    
    # Draw title (larger)
    title = "RECENT CROSSINGS"
    cv2.putText(frame, title, (panel_x + 15, panel_y + 35),
               cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    
    # Draw separator
    cv2.line(frame, (panel_x + 15, panel_y + 48),
             (panel_x + panel_width - 15, panel_y + 48),
             (100, 100, 100), 2)
    
    # Draw each crossing
    current_time = time.time()
    for i, crossing in enumerate(recent_crossings):
        y_pos = panel_y + 80 + i * line_height
        
        # Direction icon and color
        direction = crossing.get('direction', 'unknown')
        if direction in ['up', 'enter']:
            icon = "▲"
            color = (150, 220, 255)  # Brighter light blue
        else:
            icon = "▼"
            color = (100, 180, 255)  # Brighter orange
        
        # Time ago
        elapsed = current_time - crossing.get('timestamp', current_time)
        time_str = f"{elapsed:.1f}s"
        
        # Track ID with validation
        track_id = crossing.get('track_id', None)
        if track_id is None or not isinstance(track_id, (int, str)):
            track_id = "???"
        
        # Vehicle class name (if available)
        class_name = crossing.get('class_name', '')
        
        try:
            # Draw icon (larger)
            cv2.putText(frame, icon, (panel_x + 20, y_pos),
                       font, font_scale * 1.2, color, thickness + 1)
            
            # Draw track ID with vehicle type (larger, bolder)
            if class_name:
                text = f"Track {track_id} {class_name}"
            else:
                text = f"Track {track_id}"
            cv2.putText(frame, text, (panel_x + 65, y_pos),
                       font, font_scale, (255, 255, 255), 2)
            
            # Draw time (larger)
            cv2.putText(frame, time_str, (panel_x + panel_width - 90, y_pos),
                       font, font_scale * 0.9, (220, 220, 220), 2)
        except Exception as e:
            # Fallback to simple text if rendering fails
            simple_text = f"Track {track_id}"
            cv2.putText(frame, simple_text, (panel_x + 65, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame


def draw_status_panel(frame, counts, frame_number=0, total_frames=0, fps=0, direction_labels=None):
    """
    Draw enhanced status panel with vehicle counts.
    
    Args:
        frame: Input frame
        counts: Dictionary with count statistics
        frame_number: Current frame number
        total_frames: Total frames in video
        fps: Frames per second
        direction_labels: Tuple of (up_label, down_label) for custom direction names
        
    Returns:
        frame: Frame with status panel drawn
    """
    if direction_labels is None:
        direction_labels = ('Up', 'Down')
    up_label, down_label = direction_labels
    
    x, y = 15, 15
    panel_width = 280
    panel_height = 180
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = cv2.FONT_HERSHEY_DUPLEX
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y),
                  (x + panel_width, y + panel_height),
                  (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (x, y),
                  (x + panel_width, y + panel_height),
                  (0, 255, 128), 2)
    
    # Title
    title = "VEHICLE COUNT"
    cv2.putText(frame, title, (x + 15, y + 30),
               font_bold, 0.8, (255, 255, 255), 2)
    
    # Separator
    cv2.line(frame, (x + 15, y + 40), (x + panel_width - 15, y + 40),
             (100, 100, 100), 1)
    
    # Total count (large)
    total = counts.get('total', 0)
    total_text = f"Total: {total}"
    cv2.putText(frame, total_text, (x + 20, y + 75),
               font_bold, 1.1, (0, 255, 128), 2)
    
    # Up count
    up = counts.get('up', 0)
    cv2.putText(frame, "▲", (x + 20, y + 110),
               font, 0.8, (100, 200, 255), 2)
    cv2.putText(frame, f"{up_label}: {up}", (x + 50, y + 110),
               font, 0.7, (255, 255, 255), 2)
    
    # Down count
    down = counts.get('down', 0)
    cv2.putText(frame, "▼", (x + 20, y + 140),
               font, 0.8, (100, 165, 255), 2)
    cv2.putText(frame, f"{down_label}: {down}", (x + 50, y + 140),
               font, 0.7, (255, 255, 255), 2)
    
    # Progress bar if total_frames available
    if total_frames > 0:
        progress = frame_number / total_frames
        bar_width = panel_width - 40
        bar_height = 8
        bar_x = x + 20
        bar_y = y + panel_height - 25
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (80, 80, 80), -1)
        
        # Progress bar
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + progress_width, bar_y + bar_height),
                     (0, 255, 128), -1)
        
        # Frame info
        frame_text = f"{frame_number}/{total_frames}"
        cv2.putText(frame, frame_text, (bar_x, bar_y - 5),
                   font, 0.5, (200, 200, 200), 1)
    
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
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        center = det['position']
        class_name = det.get('class_name', 'vehicle')
        confidence = det.get('confidence', 0.0)
        
        # Distance filtering in clean mode
        if display_mode == 'clean' and roi_points:
            from .vehicle_counter import VehicleCounter
            counter_temp = VehicleCounter(roi_type='line', roi_points=roi_points)
            dist = counter_temp.get_distance_to_roi(center)
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
            if track_id >= 0:
                label = f"ID:{track_id} {class_name}"
            if display_mode == 'verbose':
                label += f" {confidence:.2f}"
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background
            cv2.rectangle(frame, (x1, y1 - text_height - 10),
                         (x1 + text_width + 10, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       font, font_scale, (0, 0, 0), thickness)
        
        # Draw track history if available
        if 'history' in det and len(det['history']) > 1:
            points = np.array(det['history'], dtype=np.int32)
            line_thickness = 2 if display_mode == 'verbose' else 1
            for i in range(len(points) - 1):
                cv2.line(frame, tuple(points[i]), tuple(points[i+1]), color, line_thickness)
    
    return frame


def draw_legend_panel(frame):
    """
    Draw legend panel explaining color meanings and symbols.
    
    Args:
        frame: Input frame
        
    Returns:
        frame: Frame with legend drawn
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Panel settings
    panel_width = 320
    panel_height = 200
    panel_x = (frame_width - panel_width) // 2  # Center horizontally
    panel_y = 80  # Below top panels
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (100, 200, 255), 2)
    
    # Title
    title = "LEGEND"
    cv2.putText(frame, title, (panel_x + 120, panel_y + 30),
               cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    # Separator
    cv2.line(frame, (panel_x + 15, panel_y + 40),
             (panel_x + panel_width - 15, panel_y + 40),
             (100, 100, 100), 1)
    
    y_offset = panel_y + 65
    line_spacing = 35
    
    # Green box - Normal tracking
    cv2.rectangle(frame, (panel_x + 20, y_offset - 12),
                 (panel_x + 40, y_offset + 8), (0, 255, 0), 2)
    cv2.putText(frame, "Normal Tracking", (panel_x + 50, y_offset),
               font, font_scale, (255, 255, 255), 1)
    
    # Yellow box - Recently crossed
    y_offset += line_spacing
    cv2.rectangle(frame, (panel_x + 20, y_offset - 12),
                 (panel_x + 40, y_offset + 8), (0, 255, 255), 2)
    cv2.putText(frame, "Recently Crossed ROI", (panel_x + 50, y_offset),
               font, font_scale, (255, 255, 255), 1)
    
    # Up arrow
    y_offset += line_spacing
    cv2.putText(frame, "▲", (panel_x + 25, y_offset),
               font, font_scale * 1.2, (150, 220, 255), 2)
    cv2.putText(frame, "Direction: Up/Enter", (panel_x + 50, y_offset),
               font, font_scale, (255, 255, 255), 1)
    
    # Down arrow
    y_offset += line_spacing
    cv2.putText(frame, "▼", (panel_x + 25, y_offset),
               font, font_scale * 1.2, (100, 180, 255), 2)
    cv2.putText(frame, "Direction: Down/Exit", (panel_x + 50, y_offset),
               font, font_scale, (255, 255, 255), 1)
    
    # Note
    y_offset += line_spacing + 5
    note = "Press 'L' to toggle legend"
    cv2.putText(frame, note, (panel_x + 50, y_offset),
               font, font_scale * 0.7, (180, 180, 180), 1)
    
    return frame


def draw_controls_help(frame, show_help=True):
    """
    Draw control hints panel at bottom of frame.
    
    Args:
        frame: Input frame
        show_help: Whether to show the help panel
        
    Returns:
        frame: Frame with controls drawn
    """
    if not show_help:
        return frame
    
    frame_height, frame_width = frame.shape[:2]
    
    panel_height = 60
    panel_y = frame_height - panel_height - 10
    panel_x = 15
    panel_width = frame_width - 30
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (100, 100, 100), 1)
    
    # Controls text
    controls = [
        ("[SPACE] Pause", panel_x + 20),
        ("[T] Tracks", panel_x + 160),
        ("[V] Verbose", panel_x + 280),
        ("[M] Minimal", panel_x + 420),
        ("[L] Legend", panel_x + 560),
        ("[R] Reset", panel_x + 680),
        ("[H] Help", panel_x + 800),
        ("[Q] Quit", panel_x + 920)
    ]
    
    y_pos = panel_y + 25
    for text, x_pos in controls:
        if x_pos < panel_x + panel_width - 100:
            # Draw key in brackets
            bracket_start = text.find('[')
            bracket_end = text.find(']')
            if bracket_start >= 0 and bracket_end > bracket_start:
                key = text[bracket_start:bracket_end+1]
                desc = text[bracket_end+2:]
                
                # Draw key
                cv2.putText(frame, key, (x_pos, y_pos),
                           font, font_scale, (0, 255, 128), 2)
                
                # Draw description
                cv2.putText(frame, desc, (x_pos, y_pos + 22),
                           font, font_scale * 0.8, (200, 200, 200), 1)
    
    return frame

