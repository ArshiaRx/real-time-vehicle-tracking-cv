"""
Vehicle counter that detects when tracked objects cross a region of interest (ROI).
Supports line-based and polygon-based ROI definitions.
"""

import numpy as np
import cv2
import time


class VehicleCounter:
    """Counts vehicles crossing a region of interest."""
    
    def __init__(self, roi_type='line', roi_points=None, roi_lines=None):
        """
        Initialize vehicle counter.
        
        Args:
            roi_type: Type of ROI - 'line' or 'polygon'
            roi_points: Points defining the ROI (legacy support for single ROI)
                - For 'line': [(x1, y1), (x2, y2)] - two endpoints
                - For 'polygon': [(x1, y1), (x2, y2), ...] - polygon vertices
            roi_lines: List of ROI lines (for multiple lines support)
                - Each line is [(x1, y1), (x2, y2)] - two endpoints
                - If provided, roi_points is ignored for line type
        """
        self.roi_type = roi_type
        
        # Support multiple lines: roi_lines takes precedence for line type
        if roi_type == 'line' and roi_lines is not None:
            self.roi_lines = roi_lines if isinstance(roi_lines, list) else [roi_lines]
            # Convert to list of lines format
            if len(self.roi_lines) > 0 and not isinstance(self.roi_lines[0][0], (list, tuple)):
                # Single line provided as flat list, convert to list of lines
                self.roi_lines = [self.roi_lines]
            # Legacy support: also set roi_points to first line for backward compatibility
            self.roi_points = self.roi_lines[0] if self.roi_lines else []
        else:
            # Legacy single ROI support
            self.roi_points = roi_points if roi_points is not None else []
            if roi_type == 'line' and len(self.roi_points) == 2:
                self.roi_lines = [self.roi_points]
            else:
                self.roi_lines = []
        
        self.count_up = 0  # Vehicles crossing in positive direction
        self.count_down = 0  # Vehicles crossing in negative direction
        self.total_count = 0
        self.tracked_objects = {}  # track_id -> {position, side, crossed, crossed_line_index}
        self.crossing_history = []  # History of crossings
        self.recent_crossings = []  # Recent crossings with timestamps for display
        self.crossing_animations = []  # Active crossing animations
        self.frame_time = 0  # Current frame timestamp
        self.crossing_threshold = 15.0  # Distance threshold for crossing detection (pixels)
        
    def set_roi(self, roi_type, roi_points=None, roi_lines=None):
        """
        Set or update the ROI.
        
        Args:
            roi_type: 'line' or 'polygon'
            roi_points: Points defining the ROI (legacy support)
            roi_lines: List of ROI lines for multiple lines support
        """
        self.roi_type = roi_type
        
        # Support multiple lines
        if roi_type == 'line' and roi_lines is not None:
            self.roi_lines = roi_lines if isinstance(roi_lines, list) else [roi_lines]
            # Convert to list of lines format
            if len(self.roi_lines) > 0 and not isinstance(self.roi_lines[0][0], (list, tuple)):
                # Single line provided as flat list, convert to list of lines
                self.roi_lines = [self.roi_lines]
            self.roi_points = self.roi_lines[0] if self.roi_lines else []
        elif roi_points is not None:
            # Legacy single ROI support
            self.roi_points = roi_points
            if roi_type == 'line' and len(self.roi_points) == 2:
                self.roi_lines = [self.roi_points]
            else:
                self.roi_lines = []
        
        self.tracked_objects = {}  # Reset tracking when ROI changes
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """
        Calculate signed distance from point to line.
        Positive = on one side, Negative = on other side.
        
        Args:
            point: (x, y) point
            line_start: (x, y) line start point
            line_end: (x, y) line end point
            
        Returns:
            float: Signed distance
        """
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line_start to line_end
        dx = x2 - x1
        dy = y2 - y1
        
        # Vector from line_start to point
        px_dx = px - x1
        py_dy = py - y1
        
        # Cross product to determine side (signed distance)
        cross = dx * py_dy - dy * px_dx
        
        # Normalize by line length
        line_length = np.sqrt(dx*dx + dy*dy)
        if line_length < 1e-6:
            return 0.0
        
        return cross / line_length
    
    def _point_in_polygon(self, point, polygon):
        """
        Check if point is inside polygon using ray casting algorithm.
        
        Args:
            point: (x, y) point
            polygon: List of (x, y) vertices
            
        Returns:
            bool: True if point is inside polygon
        """
        if len(polygon) < 3:
            return False
        
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _line_intersection(self, p1, p2, p3, p4):
        """
        Check if two line segments intersect.
        
        Args:
            p1, p2: Endpoints of first line segment
            p3, p4: Endpoints of second line segment
            
        Returns:
            bool: True if segments intersect
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def update(self, tracks):
        """
        Update counter with new track positions.
        
        Args:
            tracks: List of track dictionaries with 'id' and 'position'
        """
        # Check if we have valid ROI
        if self.roi_type == 'line':
            if not self.roi_lines or len(self.roi_lines) == 0:
                return
            # Check if all lines are valid
            if not all(len(line) == 2 for line in self.roi_lines):
                return
        else:  # polygon
            if not self.roi_points or len(self.roi_points) < 3:
                return
        
        current_positions = {}
        
        # Process each track
        for track in tracks:
            track_id = track['id']
            position = track['position']
            x, y = position[0], position[1]
            current_positions[track_id] = (x, y)
            
            # Store class name if available
            class_name = track.get('class_name', '')
            
            # Initialize tracking for new objects
            if track_id not in self.tracked_objects:
                # Check if vehicle is already near the line when first detected
                # Use current position as both previous and current initially
                self.tracked_objects[track_id] = {
                    'previous_position': (x, y),
                    'current_position': (x, y),
                    'side': None,
                    'crossed': False,
                    'class_name': class_name,
                    'frames_since_crossing': 0,
                    'frames_seen': 0  # Track how many frames this object has been seen
                }
                # Check crossing immediately for new tracks (they might already be crossing)
                # But only if we have history from YOLO tracks
                if 'history' in track and len(track.get('history', [])) > 1:
                    # Use history to determine previous position
                    history = track['history']
                    if len(history) >= 2:
                        prev_x, prev_y = history[-2]
                        self.tracked_objects[track_id]['previous_position'] = (prev_x, prev_y)
                        # Calculate movement for crossing detection
                        move_dx = x - prev_x
                        move_dy = y - prev_y
                        movement = np.sqrt(move_dx*move_dx + move_dy*move_dy)
                        # Check for crossing immediately
                        if not self.tracked_objects[track_id]['crossed']:
                            if self.roi_type == 'line':
                                self._check_line_crossing(self.tracked_objects[track_id], track_id, movement)
                            elif self.roi_type == 'polygon' and len(self.roi_points) >= 3:
                                self._check_polygon_crossing(self.tracked_objects[track_id], track_id)
                continue
            
            # Update position and class name
            obj = self.tracked_objects[track_id]
            
            # Only update previous_position if there's actual movement
            # This helps detect crossings even when vehicle first appears
            dx = x - obj['current_position'][0]
            dy = y - obj['current_position'][1]
            movement = np.sqrt(dx*dx + dy*dy)
            
            # Update previous position (for crossing detection)
            obj['previous_position'] = obj['current_position']
            obj['current_position'] = (x, y)
            obj['class_name'] = class_name  # Update in case it changes
            
            # Track how many frames this object has been seen
            if 'frames_seen' not in obj:
                obj['frames_seen'] = 0
            obj['frames_seen'] += 1
            
            # Increment frames since crossing (for potential reset logic)
            if 'frames_since_crossing' not in obj:
                obj['frames_since_crossing'] = 0
            if obj['crossed']:
                obj['frames_since_crossing'] += 1
                # Reset crossing flag after many frames (handles ID reuse)
                if obj['frames_since_crossing'] > 30:
                    obj['crossed'] = False
                    obj['frames_since_crossing'] = 0
            
            # Check for crossing (always check, not just when not crossed)
            # This ensures we detect crossings even if previous check missed it
            if not obj['crossed']:
                if self.roi_type == 'line':
                    self._check_line_crossing(obj, track_id, movement)
                elif self.roi_type == 'polygon' and len(self.roi_points) >= 3:
                    self._check_polygon_crossing(obj, track_id)
        
        # Remove tracks that are no longer active
        active_ids = {track['id'] for track in tracks}
        self.tracked_objects = {
            k: v for k, v in self.tracked_objects.items() 
            if k in active_ids
        }
    
    def _check_line_crossing(self, obj, track_id, movement=0):
        """Check if object crossed any of the ROI lines. Counts only once (first line crossed)."""
        prev_pos = obj['previous_position']
        curr_pos = obj['current_position']
        
        # Check all lines, but count only once (first line crossed)
        for line_index, line in enumerate(self.roi_lines):
            line_start = line[0]
            line_end = line[1]
            
            # Calculate signed distances
            prev_dist = self._point_to_line_distance(prev_pos, line_start, line_end)
            curr_dist = self._point_to_line_distance(curr_pos, line_start, line_end)
            
            # Check if crossed (sign change) - primary detection method
            crossed = False
            direction = None
            
            if prev_dist * curr_dist < 0:  # Different signs = crossed
                crossed = True
                # Determine direction based on crossing direction
                # Positive to negative = one direction, negative to positive = other
                if prev_dist > 0 and curr_dist < 0:
                    direction = 'down'
                else:
                    direction = 'up'
            # Alternative detection: vehicle very close to line and moving across
            elif abs(curr_dist) < self.crossing_threshold and movement > 0:
                # Check if vehicle is moving perpendicular to the line
                # Calculate line direction
                line_dx = line_end[0] - line_start[0]
                line_dy = line_end[1] - line_start[1]
                line_length = np.sqrt(line_dx*line_dx + line_dy*line_dy)
                
                if line_length > 1e-6:
                    # Normalize line direction
                    line_dx /= line_length
                    line_dy /= line_length
                    
                    # Calculate movement direction
                    move_dx = curr_pos[0] - prev_pos[0]
                    move_dy = curr_pos[1] - prev_pos[1]
                    move_length = np.sqrt(move_dx*move_dx + move_dy*move_dy)
                    
                    if move_length > 1e-6:
                        move_dx /= move_length
                        move_dy /= move_length
                        
                        # Check if movement is roughly perpendicular to line (crossing)
                        # Dot product should be small (perpendicular) or opposite signs
                        dot_product = line_dx * move_dx + line_dy * move_dy
                        
                        # If movement is perpendicular to line (crossing) and close to line
                        if abs(dot_product) < 0.7:  # Less than 45 degrees from perpendicular
                            crossed = True
                            # Determine direction based on which side vehicle is on
                            if curr_dist > 0:
                                direction = 'down'  # Moving from positive side
                            else:
                                direction = 'up'  # Moving from negative side
            
            if crossed and direction:
                # Count only once (first line crossed)
                # Determine which counter to increment
                if direction == 'down':
                    self.count_down += 1
                else:
                    self.count_up += 1
                
                self.total_count += 1
                obj['crossed'] = True
                obj['crossed_line_index'] = line_index  # Track which line was crossed
                
                crossing_event = {
                    'track_id': track_id,
                    'direction': direction,
                    'position': curr_pos,
                    'timestamp': time.time(),
                    'class_name': obj.get('class_name', ''),
                    'line_index': line_index  # Track which line was crossed
                }
                
                self.crossing_history.append(crossing_event)
                self.recent_crossings.append(crossing_event)
                
                # Add crossing animation with track_id for better positioning
                self.crossing_animations.append({
                    'position': curr_pos,
                    'direction': direction,
                    'start_time': time.time(),
                    'duration': 1.0,  # 1 second animation
                    'track_id': track_id,  # Store track_id for matching with bbox
                    'line_index': line_index  # Store which line was crossed
                })
                
                # Only count once, so break after first crossing detected
                break
    
    def _check_polygon_crossing(self, obj, track_id):
        """Check if object entered/exited the polygon ROI."""
        prev_pos = obj['previous_position']
        curr_pos = obj['current_position']
        
        prev_inside = self._point_in_polygon(prev_pos, self.roi_points)
        curr_inside = self._point_in_polygon(curr_pos, self.roi_points)
        
        # Entered polygon
        if not prev_inside and curr_inside:
            self.count_up += 1
            self.total_count += 1
            obj['crossed'] = True
            obj['frames_since_crossing'] = 0
            
            crossing_event = {
                'track_id': track_id,
                'direction': 'enter',
                'position': curr_pos,
                'timestamp': time.time(),
                'class_name': obj.get('class_name', '')
            }
            
            self.crossing_history.append(crossing_event)
            self.recent_crossings.append(crossing_event)
            
            # Add crossing animation with track_id for better positioning
            self.crossing_animations.append({
                'position': curr_pos,
                'direction': 'enter',
                'start_time': time.time(),
                'duration': 1.0,
                'track_id': track_id  # Store track_id for matching with bbox
            })
            
        # Exited polygon
        elif prev_inside and not curr_inside:
            self.count_down += 1
            self.total_count += 1
            obj['crossed'] = True
            obj['frames_since_crossing'] = 0
            
            crossing_event = {
                'track_id': track_id,
                'direction': 'exit',
                'position': curr_pos,
                'timestamp': time.time(),
                'class_name': obj.get('class_name', '')
            }
            
            self.crossing_history.append(crossing_event)
            self.recent_crossings.append(crossing_event)
            
            # Add crossing animation with track_id for better positioning
            self.crossing_animations.append({
                'position': curr_pos,
                'direction': 'exit',
                'start_time': time.time(),
                'duration': 1.0,
                'track_id': track_id  # Store track_id for matching with bbox
            })
    
    def get_counts(self, per_line=False):
        """
        Get current count statistics.
        
        Args:
            per_line: If True, return per-line statistics (only for line ROI type)
        
        Returns:
            dict: Count statistics
        """
        counts = {
            'total': self.total_count,
            'up': self.count_up,
            'down': self.count_down
        }
        
        # Add per-line statistics if requested and we have multiple lines
        if per_line and self.roi_type == 'line' and len(self.roi_lines) > 1:
            line_counts = {}
            for line_index in range(len(self.roi_lines)):
                line_crossings = [c for c in self.crossing_history if c.get('line_index') == line_index]
                line_counts[f'line_{line_index}'] = {
                    'total': len(line_crossings),
                    'up': len([c for c in line_crossings if c['direction'] == 'up']),
                    'down': len([c for c in line_crossings if c['direction'] == 'down'])
                }
            counts['per_line'] = line_counts
        
        return counts
    
    def get_recent_crossings(self, n=5, max_age=3.0):
        """
        Get recent crossing events.
        
        Args:
            n: Maximum number of crossings to return
            max_age: Maximum age in seconds for crossings to include
            
        Returns:
            list: Recent crossing events
        """
        current_time = time.time()
        
        # Filter crossings by age
        recent = [c for c in self.recent_crossings 
                  if current_time - c['timestamp'] <= max_age]
        
        # Update the list to only keep recent ones
        self.recent_crossings = recent
        
        # Return last n crossings
        return recent[-n:] if len(recent) > n else recent
    
    def get_active_animations(self):
        """
        Get active crossing animations.
        
        Returns:
            list: Active animations with progress
        """
        current_time = time.time()
        active = []
        
        for anim in self.crossing_animations[:]:
            elapsed = current_time - anim['start_time']
            progress = elapsed / anim['duration']
            
            if progress < 1.0:
                anim_data = anim.copy()
                anim_data['progress'] = progress
                active.append(anim_data)
            else:
                # Remove completed animations
                self.crossing_animations.remove(anim)
        
        return active
    
    def get_distance_to_roi(self, position):
        """
        Calculate distance from a position to the ROI.
        For multiple lines, returns distance to nearest line.
        
        Args:
            position: (x, y) tuple or array
            
        Returns:
            float: Distance to ROI (nearest line for multiple lines)
        """
        if self.roi_type == 'line':
            if not self.roi_lines or len(self.roi_lines) == 0:
                return float('inf')
            
            # For multiple lines, return distance to nearest line
            min_dist = float('inf')
            for line in self.roi_lines:
                if len(line) == 2:
                    dist = abs(self._point_to_line_distance(position, line[0], line[1]))
                    min_dist = min(min_dist, dist)
            return min_dist
        elif self.roi_type == 'polygon':
            if not self.roi_points or len(self.roi_points) < 3:
                return float('inf')
            
            # Distance to nearest polygon edge
            min_dist = float('inf')
            n = len(self.roi_points)
            for i in range(n):
                p1 = self.roi_points[i]
                p2 = self.roi_points[(i + 1) % n]
                dist = abs(self._point_to_line_distance(position, p1, p2))
                min_dist = min(min_dist, dist)
            return min_dist
        
        return float('inf')
    
    def reset_counts(self):
        """Reset all counters."""
        self.count_up = 0
        self.count_down = 0
        self.total_count = 0
        self.tracked_objects = {}
        self.crossing_history = []
        self.recent_crossings = []
        self.crossing_animations = []

