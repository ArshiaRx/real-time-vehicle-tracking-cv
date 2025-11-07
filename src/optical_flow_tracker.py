"""
Optical Flow Tracker using Lucas-Kanade method for sparse feature tracking.
Detects and tracks features across video frames.
"""

import cv2
import numpy as np
from collections import defaultdict


class OpticalFlowTracker:
    """Tracks objects using Lucas-Kanade optical flow."""
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Parameters for feature detection
    feature_params = dict(
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7
    )
    
    def __init__(self, max_track_length=30, min_track_length=5):
        """
        Initialize optical flow tracker.
        
        Args:
            max_track_length: Maximum number of points to keep in track history
            min_track_length: Minimum track length to consider valid
        """
        self.max_track_length = max_track_length
        self.min_track_length = min_track_length
        self.tracks = []  # List of active tracks
        self.frame_idx = 0
        self.prev_gray = None
        self.next_id = 0
        
    def update(self, frame):
        """
        Update tracker with new frame.
        
        Args:
            frame: Current frame (BGR image)
            
        Returns:
            list: List of active tracks, each containing points and track_id
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize tracks on first frame
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            # Detect initial features
            p0 = cv2.goodFeaturesToTrack(
                frame_gray,
                mask=None,
                **self.feature_params
            )
            if p0 is not None:
                for point in p0:
                    track = {
                        'id': self.next_id,
                        'points': [point.ravel()],
                        'active': True,
                        'lost_frames': 0
                    }
                    self.tracks.append(track)
                    self.next_id += 1
            self.frame_idx += 1
            return self._get_active_tracks()
        
        # Calculate optical flow
        if len(self.tracks) > 0:
            # Get all points from active tracks
            p0 = np.float32([tr['points'][-1] for tr in self.tracks if tr['active']]).reshape(-1, 1, 2)
            
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, frame_gray, p0, None, **self.lk_params
            )
            
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            # Update tracks
            track_idx = 0
            for i, track in enumerate(self.tracks):
                if not track['active']:
                    continue
                    
                if st[track_idx] == 1:
                    # Track is good, update it
                    new_point = good_new[track_idx]
                    track['points'].append(new_point)
                    track['lost_frames'] = 0
                    
                    # Limit track history
                    if len(track['points']) > self.max_track_length:
                        track['points'].pop(0)
                else:
                    # Track lost
                    track['lost_frames'] += 1
                    if track['lost_frames'] > 5:  # Mark as inactive after 5 lost frames
                        track['active'] = False
                
                track_idx += 1
            
            # Add new features periodically
            if self.frame_idx % 5 == 0:  # Every 5 frames
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                
                # Draw circles around existing tracks
                for track in self.tracks:
                    if track['active'] and len(track['points']) > 0:
                        x, y = track['points'][-1].astype(int)
                        cv2.circle(mask, (x, y), 5, 0, -1)
                
                # Detect new features
                p0_new = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
                if p0_new is not None:
                    for point in p0_new:
                        track = {
                            'id': self.next_id,
                            'points': [point.ravel()],
                            'active': True,
                            'lost_frames': 0
                        }
                        self.tracks.append(track)
                        self.next_id += 1
        
        # Update previous frame
        self.prev_gray = frame_gray
        self.frame_idx += 1
        
        # Clean up inactive tracks
        self.tracks = [t for t in self.tracks if t['active'] or len(t['points']) >= self.min_track_length]
        
        return self._get_active_tracks()
    
    def _get_active_tracks(self):
        """
        Get active tracks formatted for use.
        
        Returns:
            list: List of tracks with id, current position, and history
        """
        active_tracks = []
        for track in self.tracks:
            if track['active'] and len(track['points']) >= self.min_track_length:
                current_pos = track['points'][-1]
                active_tracks.append({
                    'id': track['id'],
                    'position': current_pos,
                    'history': track['points'][-self.max_track_length:],
                    'length': len(track['points'])
                })
        return active_tracks
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.prev_gray = None
        self.frame_idx = 0
        self.next_id = 0

