"""
Utility functions for visualization, ROI selection, and configuration.
"""

import cv2
import numpy as np
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
                cv2.line(display_img, self.roi_points[0], self.roi_points[1], (0, 255, 0), 2)
                cv2.circle(display_img, self.roi_points[0], 5, (0, 255, 0), -1)
                cv2.circle(display_img, self.roi_points[1], 5, (0, 255, 0), -1)
            
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


def draw_tracks(frame, tracks, max_history=10):
    """
    Draw tracking trails on frame.
    
    Args:
        frame: Input frame
        tracks: List of track dictionaries
        max_history: Maximum number of history points to draw
        
    Returns:
        frame: Frame with tracks drawn
    """
    for track in tracks:
        track_id = track['id']
        history = track.get('history', [])
        current_pos = track['position']
        
        # Draw track history
        if len(history) > 1:
            points = np.array(history[-max_history:], dtype=np.int32)
            for i in range(len(points) - 1):
                alpha = i / len(points)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(frame, tuple(points[i]), tuple(points[i+1]), color, 2)
        
        # Draw current position
        x, y = int(current_pos[0]), int(current_pos[1])
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(frame, f"ID:{track_id}", (x+10, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame


def draw_roi(frame, roi_points, roi_type='line'):
    """
    Draw ROI on frame.
    
    Args:
        frame: Input frame
        roi_points: ROI points
        roi_type: 'line' or 'polygon'
        
    Returns:
        frame: Frame with ROI drawn
    """
    if not roi_points:
        return frame
    
    if roi_type == 'line' and len(roi_points) == 2:
        cv2.line(frame, roi_points[0], roi_points[1], (0, 255, 0), 3)
        cv2.circle(frame, roi_points[0], 8, (0, 255, 0), -1)
        cv2.circle(frame, roi_points[1], 8, (0, 255, 0), -1)
    elif roi_type == 'polygon' and len(roi_points) >= 3:
        pts = np.array(roi_points, np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
        for pt in roi_points:
            cv2.circle(frame, pt, 8, (0, 255, 0), -1)
    
    return frame


def draw_counts(frame, counts, position=(10, 30)):
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
    font_scale = 0.7
    thickness = 2
    
    # Background rectangle
    text = f"Total: {counts.get('total', 0)}"
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x-5, y-text_height-5), 
                  (x+text_width+5, y+baseline+30), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, f"Total: {counts.get('total', 0)}", 
               (x, y), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(frame, f"Up: {counts.get('up', 0)}", 
               (x, y+25), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(frame, f"Down: {counts.get('down', 0)}", 
               (x, y+50), font, font_scale, (0, 255, 0), thickness)
    
    return frame


def draw_kalman_predictions(frame, kalman_tracks):
    """
    Draw Kalman filter predictions on frame.
    
    Args:
        frame: Input frame
        kalman_tracks: Dictionary of track_id -> KalmanFilter objects
        
    Returns:
        frame: Frame with predictions drawn
    """
    for track_id, kf in kalman_tracks.items():
        x, y = kf.get_position()
        vx, vy = kf.get_velocity()
        
        # Draw predicted position
        cv2.circle(frame, (x, y), 8, (255, 0, 0), 2)
        
        # Draw velocity vector
        if abs(vx) > 0.1 or abs(vy) > 0.1:
            end_x = int(x + vx * 10)
            end_y = int(y + vy * 10)
            cv2.arrowedLine(frame, (x, y), (end_x, end_y), (255, 0, 0), 2)
        
        cv2.putText(frame, f"KF:{track_id}", (x+10, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame

