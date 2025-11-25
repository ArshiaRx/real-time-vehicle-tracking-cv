"""
Main video processing pipeline that integrates optical flow, Kalman filtering, and vehicle counting.
"""

import cv2
import numpy as np
from typing import Optional, Dict
from .optical_flow_tracker import OpticalFlowTracker
from .kalman_filter import ObjectKalmanFilter
from .vehicle_counter import VehicleCounter
from .yolo_detector import YOLODetector
from .utils import draw_tracks, draw_roi, draw_counts, draw_kalman_predictions, draw_crossing_event


class VideoProcessor:
    """Main video processing pipeline."""
    
    def __init__(self, use_kalman=True, roi_type='line', roi_points=None, roi_lines=None,
                 use_yolo=False, yolo_confidence=0.4, min_box_size=20):
        """
        Initialize video processor.
        
        Args:
            use_kalman: Whether to use Kalman filtering for smooth tracking
            roi_type: Type of ROI ('line' or 'polygon')
            roi_points: ROI points (legacy support)
            roi_lines: List of ROI lines for multiple lines support
            use_yolo: Whether to use YOLOv8 for detection (more accurate)
            yolo_confidence: YOLO confidence threshold (0.0-1.0)
            min_box_size: Minimum bounding box size in pixels
        """
        self.use_yolo = use_yolo
        
        # Initialize YOLO detector if requested
        if use_yolo:
            self.yolo_detector = YOLODetector(
                model_size='n', 
                confidence_threshold=yolo_confidence,
                min_box_size=min_box_size
            )
            if not self.yolo_detector.available:
                print("⚠️  YOLOv8 not available, falling back to optical flow")
                self.use_yolo = False
                self.yolo_detector = None
        else:
            self.yolo_detector = None
        
        # Initialize optical flow tracker (used when YOLO is not enabled or as fallback)
        self.optical_flow_tracker = OpticalFlowTracker()
        self.use_kalman = use_kalman and not use_yolo  # Disable Kalman when using YOLO
        self.kalman_tracks = {}  # track_id -> KalmanFilter
        self.vehicle_counter = VehicleCounter(roi_type=roi_type, roi_points=roi_points, roi_lines=roi_lines)
        self.frame_count = 0
        self.last_yolo_tracks = []  # Cache last YOLO tracks for skipped frames
        self.process_yolo_every_n = 2  # Process YOLO every 2nd frame for 2x speed
        
    def set_roi(self, roi_type, roi_points=None, roi_lines=None):
        """Set or update the ROI."""
        self.vehicle_counter.set_roi(roi_type, roi_points, roi_lines)
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            dict: Processing results with annotated frame and statistics
        """
        self.frame_count += 1
        
        # Use YOLO detection if enabled
        if self.use_yolo and self.yolo_detector and self.yolo_detector.available:
            # Process YOLO every 2nd frame for 2x speed (tracking maintains state internally)
            if self.frame_count % self.process_yolo_every_n == 0:
                # Full YOLO detection and tracking
                yolo_tracks = self.yolo_detector.track(frame)
                self.last_yolo_tracks = yolo_tracks
            else:
                # Reuse last tracks (YOLO tracking maintains state, we just reuse positions)
                # This is safe because YOLO tracking persists internally
                yolo_tracks = self.last_yolo_tracks if self.last_yolo_tracks else []
            
            # Update vehicle counter with YOLO tracks (always update for accurate counting)
            self.vehicle_counter.update(yolo_tracks)
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            # Draw ROI
            if self.vehicle_counter.roi_type == 'line' and self.vehicle_counter.roi_lines:
                annotated_frame = draw_roi(
                    annotated_frame, 
                    roi_points=self.vehicle_counter.roi_points,
                    roi_lines=self.vehicle_counter.roi_lines,
                    roi_type=self.vehicle_counter.roi_type
                )
            elif self.vehicle_counter.roi_points:
                annotated_frame = draw_roi(
                    annotated_frame, 
                    roi_points=self.vehicle_counter.roi_points,
                    roi_type=self.vehicle_counter.roi_type
                )
            
            # Draw YOLO tracks (with display_mode if set)
            display_mode = getattr(self, 'display_mode', 'clean')
            annotated_frame = draw_tracks(
                annotated_frame, 
                yolo_tracks,
                display_mode=display_mode,
                roi_points=self.vehicle_counter.roi_points
            )
            
            # Draw +1 indicators on vehicles when they cross ROI
            animations = self.vehicle_counter.get_active_animations()
            if animations:
                # Enhance animations with bounding box info for better positioning above vehicles
                enhanced_animations = []
                for anim in animations:
                    track_id = anim.get('track_id', None)
                    position = anim['position']
                    
                    # Find matching YOLO track to get bbox top position
                    for track in yolo_tracks:
                        if track.get('id') == track_id and 'bbox' in track:
                            bbox = track['bbox']
                            x1, y1, x2, y2 = bbox
                            # Use top-center of bounding box for better visibility above vehicle
                            enhanced_anim = anim.copy()
                            enhanced_anim['position'] = ((x1 + x2) // 2, y1)
                            enhanced_animations.append(enhanced_anim)
                            break
                    else:
                        # No matching track found, use center position
                        enhanced_animations.append(anim)
                
                annotated_frame = draw_crossing_event(annotated_frame, enhanced_animations)
            
            # Draw counts
            counts = self.vehicle_counter.get_counts()
            annotated_frame = draw_counts(annotated_frame, counts)
            
            return {
                'frame': annotated_frame,
                'counts': counts,
                'tracks': yolo_tracks,
                'frame_number': self.frame_count
            }
        else:
            # Fallback to optical flow tracking
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
            if self.vehicle_counter.roi_type == 'line' and self.vehicle_counter.roi_lines:
                annotated_frame = draw_roi(
                    annotated_frame, 
                    roi_points=self.vehicle_counter.roi_points,
                    roi_lines=self.vehicle_counter.roi_lines,
                    roi_type=self.vehicle_counter.roi_type
                )
            elif self.vehicle_counter.roi_points:
                annotated_frame = draw_roi(
                    annotated_frame, 
                    roi_points=self.vehicle_counter.roi_points,
                    roi_type=self.vehicle_counter.roi_type
                )
            
            # Draw tracks (with display_mode if set)
            display_mode = getattr(self, 'display_mode', 'clean')
            if self.use_kalman:
                # Draw Kalman predictions
                annotated_frame = draw_kalman_predictions(
                    annotated_frame, 
                    self.kalman_tracks,
                    display_mode=display_mode,
                    roi_points=self.vehicle_counter.roi_points
                )
                # Also draw optical flow tracks in different color
                annotated_frame = draw_tracks(
                    annotated_frame, 
                    of_tracks,
                    display_mode=display_mode,
                    roi_points=self.vehicle_counter.roi_points
                )
            else:
                # Draw optical flow tracks only
                annotated_frame = draw_tracks(
                    annotated_frame, 
                    of_tracks,
                    display_mode=display_mode,
                    roi_points=self.vehicle_counter.roi_points
                )
            
            # Draw +1 indicators on vehicles when they cross ROI
            animations = self.vehicle_counter.get_active_animations()
            if animations:
                annotated_frame = draw_crossing_event(annotated_frame, animations)
            
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
        if self.yolo_detector and self.yolo_detector.available:
            self.yolo_detector.reset()
        self.kalman_tracks = {}
        self.vehicle_counter.reset_counts()
        self.frame_count = 0

