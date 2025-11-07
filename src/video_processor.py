"""
Main video processing pipeline that integrates optical flow, Kalman filtering, and vehicle counting.
"""

import cv2
import numpy as np
from typing import Optional, Dict
from .optical_flow_tracker import OpticalFlowTracker
from .kalman_filter import ObjectKalmanFilter
from .vehicle_counter import VehicleCounter
from .utils import draw_tracks, draw_roi, draw_counts, draw_kalman_predictions


class VideoProcessor:
    """Main video processing pipeline."""
    
    def __init__(self, use_kalman=True, roi_type='line', roi_points=None):
        """
        Initialize video processor.
        
        Args:
            use_kalman: Whether to use Kalman filtering for smooth tracking
            roi_type: Type of ROI ('line' or 'polygon')
            roi_points: ROI points
        """
        self.optical_flow_tracker = OpticalFlowTracker()
        self.use_kalman = use_kalman
        self.kalman_tracks = {}  # track_id -> KalmanFilter
        self.vehicle_counter = VehicleCounter(roi_type=roi_type, roi_points=roi_points)
        self.frame_count = 0
        
    def set_roi(self, roi_type, roi_points):
        """Set or update the ROI."""
        self.vehicle_counter.set_roi(roi_type, roi_points)
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            dict: Processing results with annotated frame and statistics
        """
        self.frame_count += 1
        
        # Get optical flow tracks
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
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw ROI
        if self.vehicle_counter.roi_points:
            annotated_frame = draw_roi(
                annotated_frame, 
                self.vehicle_counter.roi_points,
                self.vehicle_counter.roi_type
            )
        
        # Draw tracks
        if self.use_kalman:
            # Draw Kalman predictions
            annotated_frame = draw_kalman_predictions(annotated_frame, self.kalman_tracks)
            # Also draw optical flow tracks in different color
            annotated_frame = draw_tracks(annotated_frame, of_tracks)
        else:
            # Draw optical flow tracks only
            annotated_frame = draw_tracks(annotated_frame, of_tracks)
        
        # Draw counts
        counts = self.vehicle_counter.get_counts()
        annotated_frame = draw_counts(annotated_frame, counts)
        
        return {
            'frame': annotated_frame,
            'counts': counts,
            'tracks': of_tracks,
            'frame_number': self.frame_count
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
        self.vehicle_counter.reset_counts()
        self.frame_count = 0

