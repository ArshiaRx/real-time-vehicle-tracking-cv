"""
Test script to verify all components work correctly without requiring a video file.
"""

import numpy as np
import cv2
from src.optical_flow_tracker import OpticalFlowTracker
from src.kalman_filter import ObjectKalmanFilter
from src.vehicle_counter import VehicleCounter
from src.video_processor import VideoProcessor

def test_kalman_filter():
    """Test Kalman filter functionality."""
    print("Testing Kalman Filter...")
    kf = ObjectKalmanFilter(100, 200)
    kf.id = 1
    
    # Test initial state
    x, y = kf.get_position()
    assert x == 100 and y == 200, f"Initial position incorrect: ({x}, {y})"
    print("  [OK] Initial state correct")
    
    # Test prediction
    kf.predict()
    x_pred, y_pred = kf.get_position()
    print(f"  [OK] Prediction: ({x_pred:.1f}, {y_pred:.1f})")
    
    # Test update
    kf.update([105, 205])
    x_upd, y_upd = kf.get_position()
    print(f"  [OK] After update: ({x_upd:.1f}, {y_upd:.1f})")
    
    print("  [OK] Kalman Filter test passed!\n")
    return True

def test_vehicle_counter():
    """Test vehicle counter functionality."""
    print("Testing Vehicle Counter...")
    
    # Test line ROI
    counter = VehicleCounter(roi_type='line', roi_points=[(100, 100), (200, 100)])
    
    # Simulate object crossing from one side to another
    tracks = [
        {'id': 1, 'position': np.array([150, 90])},   # Above line
        {'id': 1, 'position': np.array([150, 110])}, # Below line (crossed)
    ]
    
    counter.update([tracks[0]])
    counts1 = counter.get_counts()
    assert counts1['total'] == 0, "Should not count before crossing"
    
    counter.update([tracks[1]])
    counts2 = counter.get_counts()
    assert counts2['total'] > 0, "Should count after crossing"
    print(f"  [OK] Line crossing detected: {counts2['total']} vehicles")
    
    # Test polygon ROI
    counter_poly = VehicleCounter(roi_type='polygon', roi_points=[(100, 100), (200, 100), (200, 200), (100, 200)])
    
    tracks_poly = [
        {'id': 2, 'position': np.array([50, 50])},    # Outside
        {'id': 2, 'position': np.array([150, 150])},  # Inside (entered)
    ]
    
    counter_poly.update([tracks_poly[0]])
    counter_poly.update([tracks_poly[1]])
    counts_poly = counter_poly.get_counts()
    assert counts_poly['total'] > 0, "Should count polygon entry"
    print(f"  [OK] Polygon entry detected: {counts_poly['total']} vehicles")
    
    print("  [OK] Vehicle Counter test passed!\n")
    return True

def test_optical_flow_tracker():
    """Test optical flow tracker with synthetic frames."""
    print("Testing Optical Flow Tracker...")
    
    tracker = OpticalFlowTracker()
    
    # Create synthetic frames with moving object
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add a moving white rectangle
    cv2.rectangle(frame1, (100, 100), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(frame2, (110, 110), (160, 160), (255, 255, 255), -1)
    
    # First update should initialize tracks
    tracks1 = tracker.update(frame1)
    print(f"  [OK] Initialized with {len(tracks1)} tracks")
    
    # Second update should track movement
    tracks2 = tracker.update(frame2)
    print(f"  [OK] Tracking {len(tracks2)} objects after movement")
    
    print("  [OK] Optical Flow Tracker test passed!\n")
    return True

def test_video_processor():
    """Test video processor integration."""
    print("Testing Video Processor...")
    
    processor = VideoProcessor(use_kalman=True, roi_type='line', 
                               roi_points=[(100, 100), (200, 100)])
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (100, 100), (255, 255, 255), -1)
    
    # Process frame
    result = processor.process_frame(frame)
    
    assert 'frame' in result, "Result should contain 'frame'"
    assert 'counts' in result, "Result should contain 'counts'"
    assert 'tracks' in result, "Result should contain 'tracks'"
    
    print(f"  [OK] Processed frame successfully")
    print(f"  [OK] Counts: {result['counts']}")
    print(f"  [OK] Tracks: {len(result['tracks'])}")
    
    print("  [OK] Video Processor test passed!\n")
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("Running System Tests")
    print("=" * 50)
    print()
    
    try:
        test_kalman_filter()
        test_vehicle_counter()
        test_optical_flow_tracker()
        test_video_processor()
        
        print("=" * 50)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("=" * 50)
        return True
    except Exception as e:
        print("=" * 50)
        print(f"[ERROR] TEST FAILED: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

