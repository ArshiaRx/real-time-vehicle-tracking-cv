"""
Vehicle color detection module.
Extracts dominant color from vehicle bounding boxes and classifies them.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter


class ColorDetector:
    """Detects and classifies vehicle colors from bounding box regions."""
    
    # Color name mappings based on RGB/HSV ranges
    COLOR_NAMES = {
        'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
        'blue': [(100, 50, 50), (130, 255, 255)],
        'green': [(50, 50, 50), (70, 255, 255)],
        'yellow': [(20, 50, 50), (30, 255, 255)],
        'orange': [(10, 50, 50), (20, 255, 255)],
        'purple': [(130, 50, 50), (160, 255, 255)],
        'pink': [(160, 50, 50), (170, 255, 255)],
        'white': [(0, 0, 200), (180, 30, 255)],
        'gray': [(0, 0, 50), (180, 30, 200)],
        'silver': [(0, 0, 150), (180, 20, 220)],
        'black': [(0, 0, 0), (180, 255, 50)],
        'brown': [(10, 50, 20), (20, 150, 150)],
    }
    
    def __init__(self, n_colors=3):
        """
        Initialize color detector.
        
        Args:
            n_colors: Number of dominant colors to extract (default: 3)
        """
        self.n_colors = n_colors
    
    def extract_dominant_color(self, frame, bbox):
        """
        Extract dominant color from a bounding box region.
        
        Args:
            frame: Input frame (BGR image)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            tuple: (B, G, R) dominant color in BGR format
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return (128, 128, 128)  # Default gray
        
        # Reshape to list of pixels
        pixels = roi.reshape(-1, 3)
        
        # Remove very dark and very bright pixels (likely shadows/highlights)
        brightness = pixels.mean(axis=1)
        mask = (brightness > 30) & (brightness < 220)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) == 0:
            filtered_pixels = pixels
        
        # Use K-means to find dominant colors
        if len(filtered_pixels) < self.n_colors:
            # If too few pixels, just average
            dominant_color = filtered_pixels.mean(axis=0).astype(int)
        else:
            try:
                kmeans = KMeans(n_clusters=min(self.n_colors, len(filtered_pixels)), 
                              random_state=42, n_init=10)
                kmeans.fit(filtered_pixels)
                
                # Get the most common cluster (dominant color)
                labels = kmeans.labels_
                label_counts = Counter(labels)
                dominant_label = label_counts.most_common(1)[0][0]
                dominant_color = kmeans.cluster_centers_[dominant_label].astype(int)
            except:
                # Fallback to mean
                dominant_color = filtered_pixels.mean(axis=0).astype(int)
        
        return tuple(dominant_color)
    
    def classify_color(self, bgr_color):
        """
        Classify a BGR color into a color name.
        
        Args:
            bgr_color: (B, G, R) tuple
            
        Returns:
            str: Color name (e.g., 'red', 'blue', 'white')
        """
        B, G, R = bgr_color
        
        # Convert to HSV for better color classification
        hsv = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        H, S, V = hsv
        
        # Check brightness first (white, gray, black, silver)
        if V > 200 and S < 30:
            return 'white'
        elif V < 50:
            return 'black'
        elif 50 < V < 200 and S < 30:
            if 150 < V < 220:
                return 'silver'
            else:
                return 'gray'
        
        # Check hue for colored vehicles
        if 0 <= H <= 10 or 170 <= H <= 180:
            if S > 100:
                return 'red'
            else:
                return 'brown'
        elif 10 < H <= 20:
            if S > 100:
                return 'orange'
            else:
                return 'brown'
        elif 20 < H <= 30:
            return 'yellow'
        elif 50 < H <= 70:
            return 'green'
        elif 100 < H <= 130:
            return 'blue'
        elif 130 < H <= 160:
            return 'purple'
        elif 160 < H <= 170:
            return 'pink'
        
        # Default to most similar based on RGB
        return self._classify_by_rgb(bgr_color)
    
    def _classify_by_rgb(self, bgr_color):
        """Fallback classification using RGB values."""
        B, G, R = bgr_color
        
        # Simple RGB-based classification
        max_val = max(R, G, B)
        min_val = min(R, G, B)
        diff = max_val - min_val
        
        if diff < 30:  # Low saturation
            if max_val > 200:
                return 'white'
            elif max_val < 50:
                return 'black'
            elif 100 < max_val < 200:
                return 'gray'
            else:
                return 'silver'
        
        # High saturation colors
        if R > G and R > B:
            if R - G > 50:
                return 'red'
            else:
                return 'orange'
        elif G > R and G > B:
            return 'green'
        elif B > R and B > G:
            return 'blue'
        elif R > 100 and G > 100 and B < 100:
            return 'yellow'
        else:
            return 'gray'  # Default
    
    def detect_vehicle_color(self, frame, bbox):
        """
        Detect and classify vehicle color from bounding box.
        
        Args:
            frame: Input frame (BGR image)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            dict: {'bgr': (B, G, R), 'name': 'color_name', 'hex': '#RRGGBB'}
        """
        dominant_bgr = self.extract_dominant_color(frame, bbox)
        color_name = self.classify_color(dominant_bgr)
        
        # Convert to hex
        B, G, R = dominant_bgr
        hex_color = f"#{R:02x}{G:02x}{B:02x}"
        
        return {
            'bgr': dominant_bgr,
            'name': color_name,
            'hex': hex_color
        }

