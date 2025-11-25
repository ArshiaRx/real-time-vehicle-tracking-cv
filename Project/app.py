"""
Streamlit web application for Vehicle Tracking and Counting System.
Provides interactive interface for video processing, parameter adjustment, and result visualization.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os
import sys
from typing import Optional, Tuple
import imageio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.video_processor import VideoProcessor
from src.utils import ROISelector

# Page configuration
st.set_page_config(
    page_title="Vehicle Tracking & Counting System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'video_file' not in st.session_state:
    st.session_state.video_file = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = None
if 'roi_lines' not in st.session_state:
    st.session_state.roi_lines = None
if 'roi_type' not in st.session_state:
    st.session_state.roi_type = 'line'
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = None
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'intermediate_results' not in st.session_state:
    st.session_state.intermediate_results = None

def process_video_streamlit(video_path: str, processor: VideoProcessor, 
                            progress_bar, status_text) -> dict:
    """Process video and return results."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer with browser-compatible H.264 codec using imageio
    # imageio works without external DLLs and is compatible with Streamlit web apps
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    
    try:
        # Use imageio for H.264 encoding (browser-compatible, no DLL dependencies)
        writer = imageio.get_writer(
            output_path, 
            fps=fps, 
            codec='libx264',
            quality=4,  # 0-10, lower = faster encoding (4 is still acceptable quality, faster)
            macro_block_size=None,  # Auto-determine based on resolution
            pixelformat='yuv420p'  # Browser-compatible pixel format
        )
    except Exception as e:
        # Fallback to mp4v if imageio fails
        cap.release()
        return None
    
    frame_count = 0
    all_counts = []
    all_tracks = []
    
    # Process in batches for faster encoding (smaller chunks)
    batch_size = 1  # Write every frame immediately (maximum speed with no accuracy loss)
    frame_batch = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame (optimized for 2x speed - YOLO processing is now every 2nd frame internally)
            result = processor.process_frame(frame)
            annotated_frame = result['frame']
            counts = result['counts']
            tracks = result['tracks']
            
            # Convert to RGB and add to batch
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_batch.append(rgb_frame)
            
            # Store statistics
            all_counts.append({
                'frame': frame_count,
                'total': counts['total'],
                'up': counts['up'],
                'down': counts['down']
            })
            all_tracks.append(len(tracks))
            
            frame_count += 1
            
            # Write batch when it reaches batch_size or at end
            if len(frame_batch) >= batch_size or frame_count >= total_frames:
                for batch_frame in frame_batch:
                    writer.append_data(batch_frame)
                frame_batch = []  # Clear batch
            
            # Update progress and statistics in real-time (reduced frequency for speed)
            if frame_count % 50 == 0 or frame_count >= total_frames:  # Update every 50 frames for less overhead
                progress = (frame_count / total_frames) * 100
                progress_bar.progress(progress / 100)
                
                # Show current counts in status text (real-time update)
                current_total = counts['total']
                current_up = counts['up']
                current_down = counts['down']
                status_text.text(
                    f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                    f"Total: {current_total} (+{current_total - (all_counts[0]['total'] if all_counts else 0)}) | "
                    f"Up: {current_up} | Down: {current_down}"
                )
                
                # Store intermediate results for real-time statistics display
                # Update session state every 50 frames for less overhead
                if frame_count % 50 == 0:
                    st.session_state.intermediate_results = {
                        'count_history': all_counts.copy(),
                        'track_history': all_tracks.copy(),
                        'current_counts': counts.copy(),
                        'frames_processed': frame_count
                    }
        
        # Write any remaining frames in batch
        if frame_batch:
            for batch_frame in frame_batch:
                writer.append_data(batch_frame)
    finally:
        cap.release()
        writer.close()  # imageio uses close() instead of release()
    
    # Validate output file exists and has content
    if not os.path.exists(output_path):
        return None
    
    file_size = os.path.getsize(output_path)
    if file_size == 0:
        return None
    
    final_counts = processor.vehicle_counter.get_counts()
    
    return {
        'output_video': output_path,
        'total_frames': frame_count,
        'fps': fps,
        'final_counts': final_counts,
        'count_history': all_counts,
        'track_history': all_tracks
    }

# Main header
st.markdown('<div class="main-header">🚗 Vehicle Tracking & Counting System</div>', unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Video input
st.sidebar.subheader("📹 Video Input")
video_option = st.sidebar.radio(
    "Select video source:",
    ["Demo Videos", "Upload Video"],
    index=0
)

video_path = None
if video_option == "Demo Videos":
    # Dynamically scan data folder for video files
    data_folder = Path("data")
    demo_videos = {}
    
    if data_folder.exists():
        # Supported video extensions
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        for video_file in sorted(data_folder.iterdir()):
            if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                # Use filename (without extension) as display name
                display_name = video_file.stem
                demo_videos[display_name] = str(video_file)
    
    if demo_videos:
        selected_demo = st.sidebar.selectbox("Choose demo video:", list(demo_videos.keys()))
        video_path = demo_videos[selected_demo]
        if not os.path.exists(video_path):
            st.sidebar.error(f"Demo video not found: {video_path}")
            video_path = None
        st.session_state.video_path = video_path
    else:
        st.sidebar.warning("⚠️ No video files found in data folder.")
        st.session_state.video_path = None
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for processing",
        key="video_uploader"
    )
    
    # Check if a new file was uploaded or use existing one from session state
    if uploaded_file is not None:
        # Check if this is a new file (different from stored one)
        if st.session_state.video_file != uploaded_file.name or st.session_state.video_path is None:
            # New file uploaded - save it with chunked reading/writing
            upload_progress = st.sidebar.progress(0)
            upload_status = st.sidebar.empty()
            
            # Determine file extension from uploaded file name
            file_ext = os.path.splitext(uploaded_file.name)[1] or '.mp4'
            
            # Save uploaded file temporarily with chunked I/O
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            temp_file_path = tfile.name
            
            try:
                chunk_size = 1024  # 1KB chunks for faster upload speed
                file_size = uploaded_file.size
                bytes_written = 0
                
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                
                while True:
                    chunk = uploaded_file.read(chunk_size)
                    if not chunk:
                        break
                    tfile.write(chunk)
                    bytes_written += len(chunk)
                    
                    # Update progress
                    if file_size > 0:
                        progress = bytes_written / file_size
                        upload_progress.progress(progress)
                        upload_status.text(f"Uploading: {bytes_written // 1024 // 1024}MB / {file_size // 1024 // 1024}MB")
                
                tfile.flush()
                os.fsync(tfile.fileno())  # Ensure data is written to disk
                tfile.close()
                tfile = None  # Mark as closed
                
                # Validate file was written successfully
                if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
                    video_path = temp_file_path
                    st.session_state.video_path = video_path
                    st.session_state.video_file = uploaded_file.name
                    
                    upload_progress.empty()
                    upload_status.empty()
                    st.sidebar.success(f"✅ Video uploaded: {uploaded_file.name}")
                else:
                    raise Exception("Uploaded file is empty or could not be written")
            except Exception as e:
                # Ensure file is closed
                if tfile is not None:
                    try:
                        tfile.close()
                    except:
                        pass
                # Clean up on error
                if os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                st.sidebar.error(f"❌ Upload failed: {str(e)}")
                video_path = None
        else:
            # Use existing file from session state
            video_path = st.session_state.video_path
            if video_path and os.path.exists(video_path):
                st.sidebar.success(f"✅ Video ready: {uploaded_file.name}")
            else:
                # File was deleted or missing, need to re-upload
                st.session_state.video_path = None
                st.session_state.video_file = None
                video_path = None
    elif st.session_state.video_path and os.path.exists(st.session_state.video_path):
        # No new upload, but previous file exists in session
        video_path = st.session_state.video_path

# Processing parameters
st.sidebar.subheader("🔧 Processing Parameters")

use_yolo = st.sidebar.checkbox(
    "Use YOLOv8 Detection",
    value=True,
    help="Use deep learning-based detection (more accurate but slower)"
)

if use_yolo:
    yolo_confidence = st.sidebar.slider(
        "YOLO Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05,
        help="Minimum confidence for vehicle detection"
    )
    min_box_size = st.sidebar.slider(
        "Minimum Box Size (pixels)",
        min_value=10,
        max_value=100,
        value=20,
        step=5,
        help="Filter out small detections"
    )
    use_kalman = False
else:
    use_kalman = st.sidebar.checkbox(
        "Use Kalman Filter",
        value=True,
        help="Smooth tracking with Kalman filtering"
    )
    yolo_confidence = 0.4
    min_box_size = 20

# ROI configuration
st.sidebar.subheader("📍 Region of Interest (ROI)")
roi_selection_mode = st.sidebar.radio(
    "ROI Selection:",
    ["Auto-Detect (YOLOv8)", "Manual (Default)"],
    index=0,
    help="""Auto-Detect: Automatically finds optimal counting line using YOLOv8 vehicle movement analysis.
    
Manual: Uses default horizontal line in middle of frame."""
)
roi_type = st.sidebar.radio(
    "ROI Type:",
    ["Line", "Polygon"],
    index=0,
    help="""Line: Count vehicles crossing a line. Vehicles are counted when they cross from one side to the other.

Polygon: Count vehicles entering/exiting a polygon region. Vehicles are counted when they:
- Enter the polygon (counts as 'Up' or 'Enter')
- Exit the polygon (counts as 'Down' or 'Exit')
The polygon ROI creates a defined area - vehicles are tracked when they enter or leave this region."""
)

roi_type_lower = roi_type.lower()

# Multiple lines support (only for line ROI type)
use_multiple_lines = False
if roi_type_lower == 'line':
    st.sidebar.subheader("📏 Multiple Lines Configuration")
    use_multiple_lines = st.sidebar.checkbox(
        "Use Multiple Counting Lines",
        value=False,
        help="Enable multiple counting lines for better accuracy. Each vehicle is counted once (on first line crossed)."
    )
    
    if use_multiple_lines:
        # Initialize roi_lines in session state if not exists
        if st.session_state.roi_lines is None:
            st.session_state.roi_lines = []
        
        # Line management UI
        num_lines = st.sidebar.number_input(
            "Number of Lines",
            min_value=1,
            max_value=10,
            value=max(1, len(st.session_state.roi_lines)) if st.session_state.roi_lines else 1,
            step=1,
            help="Number of counting lines to use"
        )
        
        # Ensure roi_lines has the right number of lines
        if st.session_state.roi_lines is None or len(st.session_state.roi_lines) != num_lines:
            # Initialize or adjust number of lines
            if st.session_state.roi_lines is None:
                st.session_state.roi_lines = []
            
            # Add or remove lines to match num_lines
            while len(st.session_state.roi_lines) < num_lines:
                # Add default line
                if video_path and os.path.exists(video_path):
                    cap_temp = cv2.VideoCapture(video_path)
                    if cap_temp.isOpened():
                        ret, first_frame = cap_temp.read()
                        if ret:
                            h, w = first_frame.shape[:2]
                            # Default line: horizontal, spaced vertically
                            line_y = h // 2 + (len(st.session_state.roi_lines) - num_lines // 2) * (h // (num_lines + 1))
                            st.session_state.roi_lines.append([(w // 4, line_y), (3 * w // 4, line_y)])
                        cap_temp.release()
                    else:
                        st.session_state.roi_lines.append([(100, 100), (200, 100)])
                else:
                    st.session_state.roi_lines.append([(100, 100), (200, 100)])
            
            # Remove excess lines
            st.session_state.roi_lines = st.session_state.roi_lines[:num_lines]
        
        # Manual adjustment for each line
        st.sidebar.subheader("✏️ Manual Line Adjustment")
        for line_idx in range(num_lines):
            with st.sidebar.expander(f"Line {line_idx + 1}", expanded=(line_idx == 0)):
                if video_path and os.path.exists(video_path):
                    cap_temp = cv2.VideoCapture(video_path)
                    if cap_temp.isOpened():
                        w = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap_temp.release()
                    else:
                        w, h = 1920, 1080  # Default
                else:
                    w, h = 1920, 1080  # Default
                
                # Get current line coordinates
                if line_idx < len(st.session_state.roi_lines):
                    current_line = st.session_state.roi_lines[line_idx]
                    x1, y1 = current_line[0]
                    x2, y2 = current_line[1]
                else:
                    x1, y1 = w // 4, h // 2
                    x2, y2 = 3 * w // 4, h // 2
                
                # Sliders for line endpoints
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Start Point**")
                    x1_new = st.slider("X1", 0, w, x1, key=f"x1_{line_idx}")
                    y1_new = st.slider("Y1", 0, h, y1, key=f"y1_{line_idx}")
                with col2:
                    st.write("**End Point**")
                    x2_new = st.slider("X2", 0, w, x2, key=f"x2_{line_idx}")
                    y2_new = st.slider("Y2", 0, h, y2, key=f"y2_{line_idx}")
                
                # Update line if changed
                if (x1_new, y1_new) != (x1, y1) or (x2_new, y2_new) != (x2, y2):
                    st.session_state.roi_lines[line_idx] = [(x1_new, y1_new), (x2_new, y2_new)]
        
        # Reset to auto-detected button
        if st.sidebar.button("🔄 Reset to Auto-Detected", key="reset_lines"):
            if roi_selection_mode == "Auto-Detect (YOLOv8)" and use_yolo:
                with st.spinner("Re-detecting optimal line..."):
                    try:
                        from src.auto_roi_detector import auto_detect_roi_line
                        from src.yolo_detector import YOLODetector
                        
                        yolo_for_roi = YOLODetector(
                            model_size='n',
                            confidence_threshold=yolo_confidence,
                            min_box_size=min_box_size
                        )
                        
                        if yolo_for_roi.available and video_path:
                            auto_roi = auto_detect_roi_line(video_path, yolo_for_roi, num_sample_frames=50)
                            if auto_roi and len(auto_roi) == 2:
                                st.session_state.roi_lines = [auto_roi]
                                st.sidebar.success("✅ Reset to auto-detected line!")
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ Reset failed: {str(e)}")

# Display mode - removed from UI but kept internally for visualization
# Clean: Standard visualization with distance filtering
# Verbose: Detailed information with all track IDs  
# Minimal: Minimal overlay (currently not fully implemented)
display_mode = 'clean'
# Display mode affects track drawing (thickness, ID display, distance filtering)
# Clean: Standard visualization with distance filtering
# Verbose: Detailed information with all track IDs
# Minimal: Minimal overlay (currently not fully implemented)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Video Processing")
    
    if video_path and os.path.exists(video_path):
        # Display video info
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            st.info(f"📹 **Video Info:** {width}x{height} @ {fps} FPS, {total_frames} frames ({duration:.1f}s)")
            cap.release()
        
        # ROI setup: Auto-detect or set default ROI if not provided
        # For multiple lines, check roi_lines instead
        if (not use_multiple_lines and st.session_state.roi_points is None) or \
           (use_multiple_lines and (st.session_state.roi_lines is None or len(st.session_state.roi_lines) == 0)):
            if roi_selection_mode == "Auto-Detect (YOLOv8)" and use_yolo and roi_type_lower == 'line':
                # Auto-detect ROI line using YOLOv8 movement analysis
                with st.spinner("🔍 Analyzing video to detect optimal counting line..."):
                    try:
                        from src.auto_roi_detector import auto_detect_roi_line
                        from src.yolo_detector import YOLODetector
                        
                        # Initialize YOLO detector for ROI analysis
                        yolo_for_roi = YOLODetector(
                            model_size='n',
                            confidence_threshold=yolo_confidence,
                            min_box_size=min_box_size
                        )
                        
                        if yolo_for_roi.available:
                            # Auto-detect optimal ROI line
                            auto_roi = auto_detect_roi_line(
                                video_path, 
                                yolo_for_roi, 
                                num_sample_frames=50
                            )
                            
                            if auto_roi and len(auto_roi) == 2:
                                st.session_state.roi_points = auto_roi
                                # Also set as first line if using multiple lines
                                if use_multiple_lines:
                                    st.session_state.roi_lines = [auto_roi]
                                st.sidebar.success("✅ Optimal counting line detected!")
                            else:
                                # Fallback to default
                                cap_temp = cv2.VideoCapture(video_path)
                                if cap_temp.isOpened():
                                    ret, first_frame = cap_temp.read()
                                    if ret:
                                        h, w = first_frame.shape[:2]
                                        default_line = [(w // 4, h // 2), (3 * w // 4, h // 2)]
                                        st.session_state.roi_points = default_line
                                        if use_multiple_lines:
                                            st.session_state.roi_lines = [default_line]
                                    cap_temp.release()
                        else:
                            # YOLO not available, use default
                            cap_temp = cv2.VideoCapture(video_path)
                            if cap_temp.isOpened():
                                ret, first_frame = cap_temp.read()
                                if ret:
                                    h, w = first_frame.shape[:2]
                                    default_line = [(w // 4, h // 2), (3 * w // 4, h // 2)]
                                    st.session_state.roi_points = default_line
                                    if use_multiple_lines:
                                        st.session_state.roi_lines = [default_line]
                                cap_temp.release()
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ Auto-detection failed: {str(e)}. Using default ROI.")
                        # Fallback to default
                        cap_temp = cv2.VideoCapture(video_path)
                        if cap_temp.isOpened():
                            ret, first_frame = cap_temp.read()
                            if ret:
                                h, w = first_frame.shape[:2]
                                default_line = [(w // 4, h // 2), (3 * w // 4, h // 2)]
                                st.session_state.roi_points = default_line
                                if use_multiple_lines:
                                    st.session_state.roi_lines = [default_line]
                            cap_temp.release()
            else:
                # Manual/Default ROI - Read first frame to set default ROI
                cap_temp = cv2.VideoCapture(video_path)
                if cap_temp.isOpened():
                    ret, first_frame = cap_temp.read()
                    if ret:
                        h, w = first_frame.shape[:2]
                        # Set default ROI as horizontal line in middle of frame
                        if roi_type_lower == 'line':
                            default_roi = [(w // 4, h // 2), (3 * w // 4, h // 2)]
                            st.session_state.roi_points = default_roi
                            if use_multiple_lines:
                                st.session_state.roi_lines = [default_roi]
                        else:  # polygon - create a rectangular region
                            margin_x, margin_y = w // 8, h // 8
                            default_roi = [
                                (margin_x, margin_y),
                                (w - margin_x, margin_y),
                                (w - margin_x, h - margin_y),
                                (margin_x, h - margin_y)
                            ]
                            st.session_state.roi_points = default_roi
                    cap_temp.release()
        
        # Process button
        if st.button("🚀 Process Video", type="primary", width='stretch'):
            with st.spinner("Initializing processor..."):
                # Use ROI points/lines from session state (should be set above if None)
                roi_points = st.session_state.roi_points
                roi_lines = None
                if roi_type_lower == 'line' and use_multiple_lines and st.session_state.roi_lines:
                    roi_lines = st.session_state.roi_lines
                
                # Initialize processor with YOLO support
                processor = VideoProcessor(
                    use_kalman=use_kalman if not use_yolo else False,
                    roi_type=roi_type_lower,
                    roi_points=roi_points,
                    roi_lines=roi_lines,
                    use_yolo=use_yolo,
                    yolo_confidence=yolo_confidence if use_yolo else 0.4,
                    min_box_size=min_box_size if use_yolo else 20
                )
                processor.display_mode = display_mode  # Set display mode for visualization
                
                st.session_state.processor = processor
            
            # Process video
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing video frames..."):
                results = process_video_streamlit(
                    video_path,
                    processor,
                    progress_bar,
                    status_text
                )
            
            if results:
                # Clear intermediate results and set final results
                st.session_state.intermediate_results = None
                
                # Validate output video file before displaying
                output_video_path = results['output_video']
                if os.path.exists(output_video_path):
                    file_size = os.path.getsize(output_video_path)
                    if file_size > 0:
                        st.session_state.processing_results = results
                        st.success("✅ Video processing completed!")
                        
                        # Display output video
                        st.subheader("📹 Processed Video")
                        try:
                            st.video(output_video_path)
                        except Exception as e:
                            st.error(f"❌ Failed to display video: {str(e)}")
                            st.info("💡 Try downloading the video instead.")
                        
                        # Download button
                        try:
                            with open(output_video_path, 'rb') as f:
                                video_data = f.read()
                                st.download_button(
                                    label="💾 Download Processed Video",
                                    data=video_data,
                                    file_name="tracked_video.mp4",
                                    mime="video/mp4",
                                    width='stretch'
                                )
                        except Exception as e:
                            st.error(f"❌ Failed to prepare download: {str(e)}")
                    else:
                        st.error("❌ Processed video file is empty. Video encoding may have failed.")
                else:
                    st.error("❌ Processed video file not found. Video encoding may have failed.")
            else:
                st.error("❌ Failed to process video. Please check the video file and try again.")
    else:
        st.warning("⚠️ Please select or upload a video file to begin processing.")

with col2:
    st.subheader("📈 Statistics")
    
    # Only show statistics after video processing completes and video is displayed
    # Don't show statistics during processing - wait until video is shown
    if st.session_state.processing_results:
        results = st.session_state.processing_results
        final_counts = results['final_counts']
        
        # Metrics
        st.metric("Total Vehicles", final_counts['total'])
        st.metric("Direction Up", final_counts['up'])
        st.metric("Direction Down", final_counts['down'])
        
        # Show which lines were crossed if multiple lines were used
        if st.session_state.processing_results and 'processor' in st.session_state and st.session_state.processor:
            processor = st.session_state.processor
            if processor.vehicle_counter.roi_type == 'line' and processor.vehicle_counter.roi_lines and len(processor.vehicle_counter.roi_lines) > 1:
                st.subheader("📊 Per-Line Statistics")
                counts_per_line = processor.vehicle_counter.get_counts(per_line=True)
                if 'per_line' in counts_per_line:
                    for line_key, line_stats in counts_per_line['per_line'].items():
                        line_num = line_key.replace('line_', '')
                        st.metric(f"Line {int(line_num) + 1} Total", line_stats['total'])
                        with st.expander(f"Line {int(line_num) + 1} Details"):
                            st.write(f"Up: {line_stats['up']}")
                            st.write(f"Down: {line_stats['down']}")
        
        # Count history chart
        if results['count_history']:
            df_counts = pd.DataFrame(results['count_history'])
            
            fig = px.line(
                df_counts,
                x='frame',
                y=['total', 'up', 'down'],
                title='Vehicle Count Over Time',
                labels={'frame': 'Frame Number', 'value': 'Count', 'variable': 'Direction'},
                color_discrete_map={'total': '#1f77b4', 'up': '#2ca02c', 'down': '#d62728'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Track count chart
            if results['track_history']:
                df_tracks = pd.DataFrame({
                    'frame': range(len(results['track_history'])),
                    'tracks': results['track_history']
                })
                
                fig_tracks = px.line(
                    df_tracks,
                    x='frame',
                    y='tracks',
                    title='Active Tracks Over Time',
                    labels={'frame': 'Frame Number', 'tracks': 'Number of Tracks'}
                )
                fig_tracks.update_layout(height=250)
                st.plotly_chart(fig_tracks, use_container_width=True)
            
            # Export CSV
            csv = df_counts.to_csv(index=False)
            st.download_button(
                label="📥 Download Statistics (CSV)",
                data=csv,
                file_name="vehicle_counts.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("Process a video to see statistics here.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Multi-Modal Vehicle Tracking and Counting System | "
    "CPS843 - Introduction to Computer Vision | Fall 2025"
    "</div>",
    unsafe_allow_html=True
)

