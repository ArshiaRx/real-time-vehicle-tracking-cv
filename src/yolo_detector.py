"""
YOLOv8-based vehicle detector for accurate vehicle detection and counting.
"""

import cv2
import numpy as np
from collections import defaultdict


class YOLODetector:
    """Detects vehicles using YOLOv8."""
    
    # Vehicle class IDs in COCO dataset
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, model_size='n', confidence_threshold=0.3, min_box_size=20):
        """
        Initialize YOLO detector.
        
        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Minimum confidence for detections
            min_box_size: Minimum bounding box size in pixels (filter small detections)
        """
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(f'yolov8{model_size}.pt')
            self.available = True
            print(f"✅ YOLOv8{model_size} loaded successfully (confidence: {confidence_threshold}, min_box_size: {min_box_size}px)")
        except ImportError:
            print("❌ YOLOv8 not available. Install with: pip install ultralytics")
            self.available = False
            self.yolo = None
        except Exception as e:
            print(f"❌ Error loading YOLO: {e}")
            self.available = False
            self.yolo = None
            
        self.confidence_threshold = confidence_threshold
        self.min_box_size = min_box_size
        self.track_history = defaultdict(list)
        self.next_track_id = 0
        self.max_track_length = 30
        
    def detect(self, frame):
        """
        Detect vehicles in frame.
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            list: List of detections with format:
                  [{'bbox': [x1, y1, x2, y2], 'confidence': float, 
                    'class_id': int, 'class_name': str}]
        """
        if not self.available:
            return []
        
        # Run YOLO detection
        results = self.yolo(frame, verbose=False, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                
                # Only keep vehicle classes
                if class_id in self.VEHICLE_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': self.VEHICLE_CLASSES[class_id],
                        'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                    })
        
        return detections
    
    def track(self, frame):
        """
        Detect and track vehicles using YOLO's built-in tracker.
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            list: List of tracked objects with format:
                  [{'id': int, 'bbox': [x1, y1, x2, y2], 'confidence': float,
                    'position': np.array([x, y]), 'class_name': str}]
        """
        if not self.available:
            return []
        
        # Run YOLO tracking
        results = self.yolo.track(frame, persist=True, verbose=False, 
                                  conf=self.confidence_threshold)
        
        tracks = []
        for result in results:
            boxes = result.boxes
            if boxes.id is None:
                continue
                
            for box, track_id in zip(boxes, boxes.id):
                class_id = int(box.cls[0])
                
                # Only keep vehicle classes
                if class_id in self.VEHICLE_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    # Filter by minimum box size
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if box_width < self.min_box_size or box_height < self.min_box_size:
                        continue  # Skip small detections
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    track_id = int(track_id)
                    
                    # Update track history
                    self.track_history[track_id].append([center_x, center_y])
                    if len(self.track_history[track_id]) > self.max_track_length:
                        self.track_history[track_id].pop(0)
                    
                    tracks.append({
                        'id': track_id,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'position': np.array([center_x, center_y]),
                        'class_name': self.VEHICLE_CLASSES[class_id],
                        'class_id': class_id,
                        'history': self.track_history[track_id].copy()
                    })
        
        return tracks
    
    def reset(self):
        """Reset tracker state."""
        self.track_history = defaultdict(list)
        self.next_track_id = 0


