"""
Main video processing pipeline that integrates optical flow, Kalman filtering, and vehicle counting.
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict
from .optical_flow_tracker import OpticalFlowTracker
from .kalman_filter import ObjectKalmanFilter
from .vehicle_counter import VehicleCounter
from .yolo_detector import YOLODetector
from .utils import (draw_tracks, draw_roi, draw_counts, draw_kalman_predictions,
                   draw_crossing_event, draw_recent_crossings_panel, 
                   draw_status_panel, draw_controls_help, draw_yolo_detections,
                   draw_legend_panel)


class VideoProcessor:
    """Main video processing pipeline."""
    
    def __init__(self, use_kalman=True, roi_type='line', roi_points=None, use_yolo=False, 
                 direction_labels=None, yolo_confidence=0.4, min_box_size=20):
        """
        Initialize video processor.
        
        Args:
            use_kalman: Whether to use Kalman filtering for smooth tracking
            roi_type: Type of ROI ('line' or 'polygon')
            roi_points: ROI points
            use_yolo: Whether to use YOLOv8 for detection (more accurate)
            direction_labels: Tuple of (up_label, down_label) for custom direction names
            yolo_confidence: YOLO confidence threshold (0.0-1.0)
            min_box_size: Minimum bounding box size in pixels
        """
        self.use_yolo = use_yolo
        self.direction_labels = direction_labels if direction_labels else ('Up', 'Down')
        self.min_box_size = min_box_size
        
        if use_yolo:
            self.yolo_detector = YOLODetector(model_size='n', confidence_threshold=yolo_confidence,
                                             min_box_size=min_box_size)
            if not self.yolo_detector.available:
                print("⚠️  YOLOv8 not available, falling back to optical flow")
                self.use_yolo = False
        
        self.optical_flow_tracker = OpticalFlowTracker()
        self.use_kalman = use_kalman and not use_yolo  # Don't need Kalman with YOLO tracking
        self.kalman_tracks = {}  # track_id -> KalmanFilter
        self.vehicle_counter = VehicleCounter(roi_type=roi_type, roi_points=roi_points)
        self.frame_count = 0
        
        # Display settings
        self.display_mode = 'clean'  # 'clean', 'verbose', or 'minimal'
        self.show_tracks = True
        self.show_help = True
        self.show_legend = False  # Toggle legend panel
        self.last_count = 0  # Track previous count for ROI highlighting
        self.roi_highlight_until = 0  # Timestamp until which to highlight ROI
        
    def set_roi(self, roi_type, roi_points):
        """Set or update the ROI."""
        self.vehicle_counter.set_roi(roi_type, roi_points)
    
    def process_frame(self, frame, total_frames=0):
        """
        Process a single frame.
        
        Args:
            frame: Input frame (BGR image)
            total_frames: Total number of frames (for progress bar)
            
        Returns:
            dict: Processing results with annotated frame and statistics
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Choose tracking method
        if self.use_yolo:
            # Use YOLO tracking (more accurate)
            tracks = self.yolo_detector.track(frame)
            self.vehicle_counter.update(tracks)
            of_tracks = tracks  # For visualization
        else:
            # Use optical flow tracking (original method)
            of_tracks = self.optical_flow_tracker.update(frame)
            
            # Update Kalman filters
            if self.use_kalman:
                self._update_kalman_filters(of_tracks)
                # Use Kalman predictions for counting
                kalman_tracks_for_counting = self._get_kalman_track_positions()
                self.vehicle_counter.update(kalman_tracks_for_counting)
            else:
                # Use raw optical flow tracks for counting
                self.vehicle_counter.update(of_tracks)
        
        # Get counts and check if ROI should be highlighted
        counts = self.vehicle_counter.get_counts()
        if counts['total'] > self.last_count:
            # New crossing detected, highlight ROI for 0.5 seconds
            self.roi_highlight_until = current_time + 0.5
            self.last_count = counts['total']
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw ROI with highlight if needed
        highlight_roi = current_time < self.roi_highlight_until
        if self.vehicle_counter.roi_points:
            annotated_frame = draw_roi(
                annotated_frame, 
                self.vehicle_counter.roi_points,
                self.vehicle_counter.roi_type,
                highlight=highlight_roi
            )
        
        # Get recent crossing IDs
        recent_crossings = self.vehicle_counter.get_recent_crossings()
        recent_crossing_ids = {c['track_id'] for c in recent_crossings}
        
        # Draw tracks (with filtering and display modes)
        if self.show_tracks and self.display_mode != 'minimal':
            if self.use_yolo:
                # Draw YOLO detections and tracks
                annotated_frame = draw_yolo_detections(
                    annotated_frame,
                    of_tracks,  # These are YOLO tracks
                    roi_points=self.vehicle_counter.roi_points,
                    display_mode=self.display_mode,
                    recent_crossing_ids=recent_crossing_ids
                )
            elif self.use_kalman:
                # Draw Kalman predictions
                annotated_frame = draw_kalman_predictions(
                    annotated_frame, 
                    self.kalman_tracks,
                    roi_points=self.vehicle_counter.roi_points,
                    display_mode=self.display_mode
                )
                # Also draw optical flow tracks
                annotated_frame = draw_tracks(
                    annotated_frame, 
                    of_tracks,
                    roi_points=self.vehicle_counter.roi_points,
                    display_mode=self.display_mode,
                    recent_crossing_ids=recent_crossing_ids
                )
            else:
                # Draw optical flow tracks only
                annotated_frame = draw_tracks(
                    annotated_frame, 
                    of_tracks,
                    roi_points=self.vehicle_counter.roi_points,
                    display_mode=self.display_mode,
                    recent_crossing_ids=recent_crossing_ids
                )
        
        # Draw crossing animations
        animations = self.vehicle_counter.get_active_animations()
        if animations:
            annotated_frame = draw_crossing_event(annotated_frame, animations)
        
        # Draw enhanced status panel instead of simple counts
        annotated_frame = draw_status_panel(
            annotated_frame, 
            counts,
            frame_number=self.frame_count,
            total_frames=total_frames,
            direction_labels=self.direction_labels
        )
        
        # Draw recent crossings panel
        if recent_crossings and self.display_mode != 'minimal':
            annotated_frame = draw_recent_crossings_panel(annotated_frame, recent_crossings)
        
        # Draw legend panel
        if self.show_legend:
            annotated_frame = draw_legend_panel(annotated_frame)
        
        # Draw controls help
        if self.show_help:
            annotated_frame = draw_controls_help(annotated_frame, show_help=True)
        
        return {
            'frame': annotated_frame,
            'counts': counts,
            'tracks': of_tracks,
            'frame_number': self.frame_count,
            'recent_crossings': recent_crossings
        }
    
    def _update_kalman_filters(self, of_tracks):
        """Update Kalman filters with optical flow measurements."""
        active_track_ids = set()
        
        # Update existing filters and create new ones
        for track in of_tracks:
            track_id = track['id']
            active_track_ids.add(track_id)
            position = track['position']
            x, y = position[0], position[1]
            
            if track_id not in self.kalman_tracks:
                # Create new Kalman filter
                self.kalman_tracks[track_id] = ObjectKalmanFilter(x, y)
                self.kalman_tracks[track_id].id = track_id
            else:
                # Update existing filter
                kf = self.kalman_tracks[track_id]
                kf.predict()
                kf.update([x, y])
        
        # Remove filters for tracks that are no longer active
        inactive_ids = set(self.kalman_tracks.keys()) - active_track_ids
        for track_id in inactive_ids:
            # Keep filter for a few frames in case track reappears
            kf = self.kalman_tracks[track_id]
            kf.time_since_update += 1
            if kf.time_since_update > 10:  # Remove after 10 frames
                del self.kalman_tracks[track_id]
            else:
                # Predict without update
                kf.predict()
    
    def _get_kalman_track_positions(self):
        """Get current positions from Kalman filters formatted for counter."""
        tracks = []
        for track_id, kf in self.kalman_tracks.items():
            if kf.time_since_update < 5:  # Only use recently updated tracks
                x, y = kf.get_position()
                tracks.append({
                    'id': track_id,
                    'position': np.array([x, y])
                })
        return tracks
    
    def reset(self):
        """Reset all tracking and counting."""
        self.optical_flow_tracker.reset()
        self.kalman_tracks = {}
        if self.use_yolo:
            self.yolo_detector.reset()
        self.vehicle_counter.reset_counts()
        self.frame_count = 0
        self.last_count = 0

