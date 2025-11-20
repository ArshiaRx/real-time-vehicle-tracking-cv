"""
Vehicle counter that detects when tracked objects cross a region of interest (ROI).
Supports line-based and polygon-based ROI definitions.
"""

import numpy as np
import cv2
import time


class VehicleCounter:
    """Counts vehicles crossing a region of interest."""
    
    def __init__(self, roi_type='line', roi_points=None):
        """
        Initialize vehicle counter.
        
        Args:
            roi_type: Type of ROI - 'line' or 'polygon'
            roi_points: Points defining the ROI
                - For 'line': [(x1, y1), (x2, y2)] - two endpoints
                - For 'polygon': [(x1, y1), (x2, y2), ...] - polygon vertices
        """
        self.roi_type = roi_type
        self.roi_points = roi_points if roi_points is not None else []
        self.count_up = 0  # Vehicles crossing in positive direction
        self.count_down = 0  # Vehicles crossing in negative direction
        self.total_count = 0
        self.tracked_objects = {}  # track_id -> {position, side, crossed, class_name, color}
        self.crossing_history = []  # History of crossings with full details
        self.recent_crossings = []  # Recent crossings with timestamps for display
        self.crossing_animations = []  # Active crossing animations
        self.frame_time = 0  # Current frame timestamp
        
    def set_roi(self, roi_type, roi_points):
        """
        Set or update the ROI.
        
        Args:
            roi_type: 'line' or 'polygon'
            roi_points: Points defining the ROI
        """
        self.roi_type = roi_type
        self.roi_points = roi_points
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
        if not self.roi_points or len(self.roi_points) < 2:
            return
        
        current_positions = {}
        
        # Process each track
        for track in tracks:
            track_id = track['id']
            position = track['position']
            x, y = position[0], position[1]
            current_positions[track_id] = (x, y)
            
            # Store class name and color if available
            class_name = track.get('class_name', '')
            color_info = track.get('color', {})
            
            # Initialize tracking for new objects
            if track_id not in self.tracked_objects:
                self.tracked_objects[track_id] = {
                    'previous_position': (x, y),
                    'current_position': (x, y),
                    'side': None,
                    'crossed': False,
                    'class_name': class_name,
                    'color': color_info,
                    'frames_since_crossing': 0
                }
                continue
            
            # Update position, class name, and color
            obj = self.tracked_objects[track_id]
            obj['previous_position'] = obj['current_position']
            obj['current_position'] = (x, y)
            obj['class_name'] = class_name  # Update in case it changes
            if color_info:  # Update color if available
                obj['color'] = color_info
            
            # Increment frames since crossing (for potential reset logic)
            if 'frames_since_crossing' not in obj:
                obj['frames_since_crossing'] = 0
            if obj['crossed']:
                obj['frames_since_crossing'] += 1
                # Reset crossing flag after many frames (handles ID reuse)
                if obj['frames_since_crossing'] > 30:
                    obj['crossed'] = False
                    obj['frames_since_crossing'] = 0
            
            # Check for crossing
            if not obj['crossed']:
                if self.roi_type == 'line' and len(self.roi_points) == 2:
                    self._check_line_crossing(obj, track_id)
                elif self.roi_type == 'polygon' and len(self.roi_points) >= 3:
                    self._check_polygon_crossing(obj, track_id)
        
        # Remove tracks that are no longer active
        active_ids = {track['id'] for track in tracks}
        self.tracked_objects = {
            k: v for k, v in self.tracked_objects.items() 
            if k in active_ids
        }
    
    def _check_line_crossing(self, obj, track_id):
        """Check if object crossed the line ROI."""
        line_start = self.roi_points[0]
        line_end = self.roi_points[1]
        
        prev_pos = obj['previous_position']
        curr_pos = obj['current_position']
        
        # Calculate signed distances
        prev_dist = self._point_to_line_distance(prev_pos, line_start, line_end)
        curr_dist = self._point_to_line_distance(curr_pos, line_start, line_end)
        
        # Check if crossed (sign change)
        if prev_dist * curr_dist < 0:  # Different signs = crossed
            # Determine direction based on crossing direction
            # Positive to negative = one direction, negative to positive = other
            if prev_dist > 0 and curr_dist < 0:
                self.count_down += 1
                direction = 'down'
            else:
                self.count_up += 1
                direction = 'up'
            
            self.total_count += 1
            obj['crossed'] = True
            
            crossing_event = {
                'track_id': track_id,
                'direction': direction,
                'position': curr_pos,
                'timestamp': time.time(),
                'class_name': obj.get('class_name', ''),
                'color': obj.get('color', {})
            }
            
            self.crossing_history.append(crossing_event)
            self.recent_crossings.append(crossing_event)
            
            # Add crossing animation
            self.crossing_animations.append({
                'position': curr_pos,
                'direction': direction,
                'start_time': time.time(),
                'duration': 1.0  # 1 second animation
            })
    
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
                'class_name': obj.get('class_name', ''),
                'color': obj.get('color', {})
            }
            
            self.crossing_history.append(crossing_event)
            self.recent_crossings.append(crossing_event)
            
            # Add crossing animation
            self.crossing_animations.append({
                'position': curr_pos,
                'direction': 'enter',
                'start_time': time.time(),
                'duration': 1.0
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
                'class_name': obj.get('class_name', ''),
                'color': obj.get('color', {})
            }
            
            self.crossing_history.append(crossing_event)
            self.recent_crossings.append(crossing_event)
            
            # Add crossing animation
            self.crossing_animations.append({
                'position': curr_pos,
                'direction': 'exit',
                'start_time': time.time(),
                'duration': 1.0
            })
    
    def get_counts(self):
        """
        Get current count statistics.
        
        Returns:
            dict: Count statistics
        """
        return {
            'total': self.total_count,
            'up': self.count_up,
            'down': self.count_down
        }
    
    def get_vehicle_statistics(self):
        """
        Get detailed vehicle statistics including types and colors.
        
        Returns:
            dict: Statistics with type and color breakdowns
        """
        type_counts = {}
        color_counts = {}
        direction_type_counts = {'up': {}, 'down': {}}
        direction_color_counts = {'up': {}, 'down': {}}
        
        for crossing in self.crossing_history:
            class_name = crossing.get('class_name', 'unknown')
            color_name = crossing.get('color', {}).get('name', 'unknown')
            direction = crossing.get('direction', 'unknown')
            
            # Count by type
            type_counts[class_name] = type_counts.get(class_name, 0) + 1
            
            # Count by color
            color_counts[color_name] = color_counts.get(color_name, 0) + 1
            
            # Count by direction and type
            if direction in direction_type_counts:
                direction_type_counts[direction][class_name] = direction_type_counts[direction].get(class_name, 0) + 1
            
            # Count by direction and color
            if direction in direction_color_counts:
                direction_color_counts[direction][color_name] = direction_color_counts[direction].get(color_name, 0) + 1
        
        return {
            'type_counts': type_counts,
            'color_counts': color_counts,
            'direction_type_counts': direction_type_counts,
            'direction_color_counts': direction_color_counts,
            'total_crossings': len(self.crossing_history)
        }
    
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
        
        Args:
            position: (x, y) tuple or array
            
        Returns:
            float: Distance to ROI
        """
        if not self.roi_points or len(self.roi_points) < 2:
            return float('inf')
        
        if self.roi_type == 'line' and len(self.roi_points) == 2:
            # Distance to line segment
            return abs(self._point_to_line_distance(position, self.roi_points[0], self.roi_points[1]))
        elif self.roi_type == 'polygon' and len(self.roi_points) >= 3:
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

