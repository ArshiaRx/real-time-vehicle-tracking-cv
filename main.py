"""
Main entry point for the Optical Flow Vehicle Tracking System.
Provides command-line interface and interactive video processing.
"""

import cv2
import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.video_processor import VideoProcessor
from src.utils import ROISelector


def process_video(video_path, output_path=None, use_kalman=True, roi_type='line', roi_points=None, 
                 show_display=True, save_output=True):
    """
    Process video file with tracking and counting.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output video (optional)
        use_kalman: Whether to use Kalman filtering
        roi_type: Type of ROI ('line' or 'polygon')
        roi_points: Predefined ROI points (optional)
        show_display: Whether to show real-time display
        save_output: Whether to save output video
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Initialize processor
    processor = VideoProcessor(use_kalman=use_kalman, roi_type=roi_type, roi_points=roi_points)
    
    # Select ROI if not provided
    if roi_points is None:
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return
        
        roi_selector = ROISelector()
        if roi_type == 'line':
            roi_points = roi_selector.select_line_roi(first_frame)
        else:
            roi_points = roi_selector.select_polygon_roi(first_frame)
        
        if roi_points is None:
            print("ROI selection cancelled")
            return
        
        processor.set_roi(roi_type, roi_points)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    # Setup video writer if saving output
    writer = None
    if save_output and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output video will be saved to: {output_path}")
    
    # Process frames
    frame_count = 0
    paused = False
    annotated_frame = None
    
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  's' - Step frame (when paused)")
    print("  'r' - Reset counters")
    print("  'q' or ESC - Quit")
    print("\nProcessing video...")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            result = processor.process_frame(frame)
            annotated_frame = result['frame']
            counts = result['counts']
            
            # Write frame if saving
            if writer:
                writer.write(annotated_frame)
            
            # Display progress
            progress = (frame_count / total_frames) * 100
            print(f"\rFrame {frame_count}/{total_frames} ({progress:.1f}%) - "
                  f"Total: {counts['total']}, Up: {counts['up']}, Down: {counts['down']}", 
                  end='', flush=True)
        
        # Display frame
        if show_display and annotated_frame is not None:
            # Resize frame for display (adjust these values as needed)
            display_width = 1280
            display_height = 720
            display_frame = cv2.resize(annotated_frame, (display_width, display_height))
            cv2.imshow('Optical Flow Vehicle Tracking', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                print(f"\n{'Paused' if paused else 'Resumed'}")
            elif key == ord('s') and paused:  # Step frame
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    result = processor.process_frame(frame)
                    annotated_frame = result['frame']
                    counts = result['counts']
                    if writer:
                        writer.write(annotated_frame)
                    print(f"\nFrame {frame_count} - Total: {counts['total']}, "
                          f"Up: {counts['up']}, Down: {counts['down']}")
            elif key == ord('r'):  # Reset
                processor.reset()
                print("\nCounters reset")
        else:
            # No display, just process
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if show_display:
        cv2.destroyAllWindows()
    
    # Final statistics
    final_counts = processor.vehicle_counter.get_counts()
    print(f"\n\nProcessing complete!")
    print(f"Final counts - Total: {final_counts['total']}, "
          f"Up: {final_counts['up']}, Down: {final_counts['down']}")


def process_webcam(use_kalman=True, roi_type='line', roi_points=None):
    """
    Process webcam feed with tracking and counting.
    
    Args:
        use_kalman: Whether to use Kalman filtering
        roi_type: Type of ROI ('line' or 'polygon')
        roi_points: Predefined ROI points (optional)
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {width}x{height}")
    
    # Initialize processor
    processor = VideoProcessor(use_kalman=use_kalman, roi_type=roi_type, roi_points=roi_points)
    
    # Select ROI if not provided
    if roi_points is None:
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam")
            return
        
        roi_selector = ROISelector()
        if roi_type == 'line':
            roi_points = roi_selector.select_line_roi(first_frame)
        else:
            roi_points = roi_selector.select_polygon_roi(first_frame)
        
        if roi_points is None:
            print("ROI selection cancelled")
            return
        
        processor.set_roi(roi_type, roi_points)
    
    print("\nControls:")
    print("  'r' - Reset counters")
    print("  'q' or ESC - Quit")
    print("\nProcessing webcam feed...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = processor.process_frame(frame)
        annotated_frame = result['frame']
        counts = result['counts']
        
        # Display
        # Resize frame for display (adjust these values as needed)
        display_width = 1280
        display_height = 720
        display_frame = cv2.resize(annotated_frame, (display_width, display_height))
        cv2.imshow('Optical Flow Vehicle Tracking', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('r'):  # Reset
            processor.reset()
            print("Counters reset")
    
    cap.release()
    cv2.destroyAllWindows()
    
    final_counts = processor.vehicle_counter.get_counts()
    print(f"\nFinal counts - Total: {final_counts['total']}, "
          f"Up: {final_counts['up']}, Down: {final_counts['down']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Optical Flow Vehicle Tracking System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file with default settings
  python main.py --video data/sample_video.mp4
  
  # Process video with Kalman filter and save output
  python main.py --video data/sample_video.mp4 --output output/result.mp4 --kalman
  
  # Process webcam feed
  python main.py --webcam
  
  # Use polygon ROI instead of line
  python main.py --video data/sample_video.mp4 --roi-type polygon
        """
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video file')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam as input')
    
    # Output
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--no-display', action='store_true', help='Disable real-time display')
    parser.add_argument('--no-save', action='store_true', help='Do not save output video')
    
    # Processing options
    parser.add_argument('--no-kalman', action='store_true', 
                       help='Disable Kalman filtering (use raw optical flow only)')
    parser.add_argument('--roi-type', type=str, choices=['line', 'polygon'], 
                       default='line', help='Type of ROI (default: line)')
    
    args = parser.parse_args()
    
    # Determine output path
    output_path = None
    if args.output:
        output_path = args.output
    elif args.video and not args.no_save:
        # Auto-generate output path
        input_path = Path(args.video)
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"{input_path.stem}_tracked{input_path.suffix}")
    
    # Process based on input source
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
        
        process_video(
            video_path=args.video,
            output_path=output_path,
            use_kalman=not args.no_kalman,
            roi_type=args.roi_type,
            roi_points=None,
            show_display=not args.no_display,
            save_output=not args.no_save
        )
    elif args.webcam:
        process_webcam(
            use_kalman=not args.no_kalman,
            roi_type=args.roi_type,
            roi_points=None
        )


if __name__ == '__main__':
    main()

