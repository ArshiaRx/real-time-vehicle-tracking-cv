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
                 show_display=True, save_output=True, use_yolo=False, speed=1.0,
                 direction_labels=None, yolo_confidence=0.4, min_box_size=20):
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
        use_yolo: Whether to use YOLOv8 for detection
        speed: Playback speed multiplier
        direction_labels: Tuple of (up_label, down_label) for custom direction names
        yolo_confidence: YOLO confidence threshold (0.0-1.0)
        min_box_size: Minimum bounding box size in pixels
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    playback_speed = speed
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
    if use_yolo:
        print("🚗 Using YOLOv8 for vehicle detection (accurate counting)")
    else:
        print("🔵 Using Optical Flow (feature-based tracking)")
    
    # Initialize processor
    processor = VideoProcessor(use_kalman=use_kalman, roi_type=roi_type, roi_points=roi_points, 
                              use_yolo=use_yolo, direction_labels=direction_labels,
                              yolo_confidence=yolo_confidence, min_box_size=min_box_size)
    
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
    
    # Setup display window with proper flags
    window_name = 'Optical Flow Vehicle Tracking'
    if show_display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # Set initial window size
            cv2.resizeWindow(window_name, 1280, 720)
        except Exception as e:
            print(f"Warning: Could not create resizable window: {e}")
            cv2.namedWindow(window_name)
    
    # Process frames
    frame_count = 0
    paused = False
    annotated_frame = None
    
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  's' - Step frame (when paused)")
    print("  't' - Toggle track visibility")
    print("  'v' - Toggle verbose mode")
    print("  'm' - Toggle minimal mode")
    print("  'l' - Toggle legend panel")
    print("  'h' - Toggle help panel")
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
            result = processor.process_frame(frame, total_frames=total_frames)
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
            try:
                # Display frame directly - window will handle resizing
                cv2.imshow(window_name, annotated_frame)
            except Exception as e:
                print(f"\nWarning: Display error: {e}")
                # Try to recover by recreating window
                try:
                    cv2.destroyWindow(window_name)
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name, annotated_frame)
                except:
                    pass  # Continue processing even if display fails
            
            # Adjust wait time based on FPS and speed multiplier
            delay = max(1, int((1000 / max(fps, 1)) / playback_speed))
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                print(f"\n{'Paused' if paused else 'Resumed'}")
            elif key == ord('s') and paused:  # Step frame
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    result = processor.process_frame(frame, total_frames=total_frames)
                    annotated_frame = result['frame']
                    counts = result['counts']
                    if writer:
                        writer.write(annotated_frame)
                    print(f"\nFrame {frame_count} - Total: {counts['total']}, "
                          f"Up: {counts['up']}, Down: {counts['down']}")
            elif key == ord('t'):  # Toggle tracks
                processor.show_tracks = not processor.show_tracks
                print(f"\nTracks {'shown' if processor.show_tracks else 'hidden'}")
            elif key == ord('v'):  # Verbose mode
                if processor.display_mode == 'verbose':
                    processor.display_mode = 'clean'
                    print("\nDisplay mode: Clean")
                else:
                    processor.display_mode = 'verbose'
                    print("\nDisplay mode: Verbose")
            elif key == ord('m'):  # Minimal mode
                if processor.display_mode == 'minimal':
                    processor.display_mode = 'clean'
                    print("\nDisplay mode: Clean")
                else:
                    processor.display_mode = 'minimal'
                    print("\nDisplay mode: Minimal")
            elif key == ord('l'):  # Toggle legend
                processor.show_legend = not processor.show_legend
                print(f"\nLegend panel {'shown' if processor.show_legend else 'hidden'}")
            elif key == ord('h'):  # Toggle help
                processor.show_help = not processor.show_help
                print(f"\nHelp panel {'shown' if processor.show_help else 'hidden'}")
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


def process_webcam(use_kalman=True, roi_type='line', roi_points=None, use_yolo=False, 
                   direction_labels=None, yolo_confidence=0.4, min_box_size=20):
    """
    Process webcam feed with tracking and counting.
    
    Args:
        use_kalman: Whether to use Kalman filtering
        roi_type: Type of ROI ('line' or 'polygon')
        roi_points: Predefined ROI points (optional)
        use_yolo: Whether to use YOLOv8 for detection
        direction_labels: Tuple of (up_label, down_label) for custom direction names
        yolo_confidence: YOLO confidence threshold (0.0-1.0)
        min_box_size: Minimum bounding box size in pixels
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {width}x{height}")
    if use_yolo:
        print("🚗 Using YOLOv8 for vehicle detection")
    
    # Initialize processor
    processor = VideoProcessor(use_kalman=use_kalman, roi_type=roi_type, roi_points=roi_points, 
                              use_yolo=use_yolo, direction_labels=direction_labels,
                              yolo_confidence=yolo_confidence, min_box_size=min_box_size)
    
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
    
    # Setup display window with proper flags
    window_name = 'Optical Flow Vehicle Tracking'
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Set initial window size
        cv2.resizeWindow(window_name, 1280, 720)
    except Exception as e:
        print(f"Warning: Could not create resizable window: {e}")
        cv2.namedWindow(window_name)
    
    print("\nControls:")
    print("  't' - Toggle track visibility")
    print("  'v' - Toggle verbose mode")
    print("  'm' - Toggle minimal mode")
    print("  'l' - Toggle legend panel")
    print("  'h' - Toggle help panel")
    print("  'r' - Reset counters")
    print("  'q' or ESC - Quit")
    print("\nProcessing webcam feed...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame  
        result = processor.process_frame(frame, total_frames=0)
        annotated_frame = result['frame']
        counts = result['counts']
        
        # Display
        try:
            # Display frame directly - window will handle resizing
            cv2.imshow(window_name, annotated_frame)
        except Exception as e:
            print(f"Warning: Display error: {e}")
            # Try to recover by recreating window
            try:
                cv2.destroyWindow(window_name)
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, annotated_frame)
            except:
                pass  # Continue processing even if display fails
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('t'):  # Toggle tracks
            processor.show_tracks = not processor.show_tracks
            print(f"Tracks {'shown' if processor.show_tracks else 'hidden'}")
        elif key == ord('v'):  # Verbose mode
            if processor.display_mode == 'verbose':
                processor.display_mode = 'clean'
                print("Display mode: Clean")
            else:
                processor.display_mode = 'verbose'
                print("Display mode: Verbose")
        elif key == ord('m'):  # Minimal mode
            if processor.display_mode == 'minimal':
                processor.display_mode = 'clean'
                print("Display mode: Clean")
            else:
                processor.display_mode = 'minimal'
                print("Display mode: Minimal")
        elif key == ord('l'):  # Toggle legend
            processor.show_legend = not processor.show_legend
            print(f"Legend panel {'shown' if processor.show_legend else 'hidden'}")
        elif key == ord('h'):  # Toggle help
            processor.show_help = not processor.show_help
            print(f"Help panel {'shown' if processor.show_help else 'hidden'}")
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
    parser.add_argument('--yolo', action='store_true',
                       help='Use YOLOv8 for accurate vehicle detection (recommended)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed multiplier (0.5=half speed, 2.0=double speed)')
    parser.add_argument('--direction-up', type=str, default='Up',
                       help='Label for "up" direction (e.g., "Northbound", "Entering")')
    parser.add_argument('--direction-down', type=str, default='Down',
                       help='Label for "down" direction (e.g., "Southbound", "Exiting")')
    
    # YOLO detection parameters
    parser.add_argument('--confidence', type=float, default=0.4,
                       help='YOLO confidence threshold (0.0-1.0, default: 0.4)')
    parser.add_argument('--min-box-size', type=int, default=20,
                       help='Minimum bounding box size in pixels (filter small detections)')
    
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
    direction_labels = (args.direction_up, args.direction_down)
    
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
            save_output=not args.no_save,
            use_yolo=args.yolo,
            speed=args.speed,
            direction_labels=direction_labels,
            yolo_confidence=args.confidence,
            min_box_size=args.min_box_size
        )
    elif args.webcam:
        process_webcam(
            use_kalman=not args.no_kalman,
            roi_type=args.roi_type,
            roi_points=None,
            use_yolo=args.yolo,
            direction_labels=direction_labels,
            yolo_confidence=args.confidence,
            min_box_size=args.min_box_size
        )


if __name__ == '__main__':
    main()

