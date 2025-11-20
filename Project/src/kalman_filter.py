"""
Kalman Filter implementation for smooth object tracking.
Uses a constant velocity model with 4D state: [x, y, vx, vy]
"""

import numpy as np
from filterpy.kalman import KalmanFilter


class ObjectKalmanFilter:
    """Kalman filter for tracking a single object with position and velocity."""
    
    def __init__(self, initial_x, initial_y, dt=1.0, process_noise=0.03, measurement_noise=0.1):
        """
        Initialize Kalman filter for object tracking.
        
        Args:
            initial_x: Initial x position
            initial_y: Initial y position
            dt: Time step (default 1.0 for frame-to-frame)
            process_noise: Process noise covariance (motion uncertainty)
            measurement_noise: Measurement noise covariance (observation uncertainty)
        """
        self.dt = dt
        
        # Create Kalman filter with 4D state: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        # x' = x + vx*dt
        # y' = y + vy*dt
        # vx' = vx
        # vy' = vy
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we only observe position)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kf.Q = np.eye(4) * process_noise
        
        # Measurement noise covariance
        self.kf.R = np.eye(2) * measurement_noise
        
        # Initial state: [x, y, vx, vy]
        self.kf.x = np.array([[initial_x], [initial_y], [0.0], [0.0]], dtype=np.float32)
        
        # Initial covariance (uncertainty in initial state)
        self.kf.P = np.eye(4) * 1000.0
        
        self.last_update_time = 0
        self.hit_streak = 0  # Number of consecutive successful updates
        self.time_since_update = 0  # Frames since last update
        self.id = None  # Will be set by tracker
        
    def predict(self):
        """Predict next state based on motion model."""
        self.kf.predict()
        self.time_since_update += 1
        
    def update(self, measurement):
        """
        Update filter with new measurement.
        
        Args:
            measurement: [x, y] position measurement
        """
        if measurement is not None:
            self.kf.update(measurement)
            self.hit_streak += 1
            self.time_since_update = 0
        else:
            # No measurement available, only prediction
            self.hit_streak = 0
    
    def get_state(self):
        """
        Get current state estimate.
        
        Returns:
            tuple: (x, y, vx, vy) current state
        """
        state = self.kf.x.flatten()
        return state[0], state[1], state[2], state[3]
    
    def get_position(self):
        """
        Get current position estimate.
        
        Returns:
            tuple: (x, y) current position
        """
        x, y, _, _ = self.get_state()
        return (int(x), int(y))
    
    def get_velocity(self):
        """
        Get current velocity estimate.
        
        Returns:
            tuple: (vx, vy) current velocity
        """
        _, _, vx, vy = self.get_state()
        return (vx, vy)

