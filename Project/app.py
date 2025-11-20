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
import time
from typing import Optional, Tuple
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.video_processor import VideoProcessor
from src.utils import ROISelector
from src.report_generator import ReportGenerator
import base64

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
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = None
if 'roi_type' not in st.session_state:
    st.session_state.roi_type = 'line'
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = None
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

def format_time(seconds: float) -> str:
    """
    Format time in seconds to MM:SS or HH:MM:SS format.
    
    Args:
        seconds: Time in seconds (can be negative)
        
    Returns:
        Formatted time string
    """
    if seconds < 0:
        return "00:00"
    
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def estimate_processing_time(total_frames: int, width: int, height: int, 
                             use_yolo: bool, yolo_model_size: str, 
                             has_gpu: bool = False) -> float:
    """
    Estimate processing time based on comprehensive video and system properties.
    
    Args:
        total_frames: Total number of frames in video
        width: Video width in pixels
        height: Video height in pixels
        use_yolo: Whether YOLO detection is enabled
        yolo_model_size: YOLO model size ('n', 's', 'm', 'l')
        has_gpu: Whether GPU is available
        
    Returns:
        Estimated time in seconds
    """
    # Calculate resolution factor (higher resolution = slower)
    total_pixels = width * height
    resolution_factor = 1.0
    
    # Resolution-based speed factors (1080p = baseline)
    if total_pixels >= 1920 * 1080:  # 1080p or higher
        resolution_factor = 1.0
    elif total_pixels >= 1280 * 720:  # 720p
        resolution_factor = 1.4  # Faster at lower resolution
    elif total_pixels >= 640 * 480:  # 480p
        resolution_factor = 2.0  # Much faster
    else:  # Lower than 480p
        resolution_factor = 2.5
    
    # Base processing FPS estimates (at 1080p)
    if use_yolo:
        if has_gpu:
            # GPU processing FPS at 1080p (varies by model size)
            base_fps_1080p = {
                'n': 45,  # Nano: fastest
                's': 35,  # Small
                'm': 25,  # Medium
                'l': 18,  # Large: slower but more accurate
                'x': 12   # XLarge: very slow
            }
            base_fps = base_fps_1080p.get(yolo_model_size, 25)
        else:
            # CPU processing FPS at 1080p
            base_fps_1080p = {
                'n': 12,  # Nano
                's': 10,  # Small
                'm': 8,   # Medium
                'l': 6,   # Large
                'x': 4    # XLarge
            }
            base_fps = base_fps_1080p.get(yolo_model_size, 8)
    else:
        # Optical flow only (much faster, less affected by resolution)
        base_fps = 50
        resolution_factor = 1.2  # Less impact for optical flow
    
    # Apply resolution factor
    processing_fps = base_fps * resolution_factor
    
    # Additional overhead factors
    color_detection_overhead = 0.05  # 5% overhead for color detection
    video_encoding_overhead = 0.10   # 10% overhead for video writing
    
    # Apply overheads
    if use_yolo:
        processing_fps = processing_fps * (1 - color_detection_overhead) * (1 - video_encoding_overhead)
    
    # Calculate estimated time
    estimated_time = total_frames / processing_fps
    
    return estimated_time

def process_video_streamlit(video_path: str, processor: VideoProcessor, 
                            progress_bar, status_text, use_yolo: bool = False,
                            yolo_model_size: str = 'n') -> dict:
    """Process video and return results."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer - try multiple codecs for compatibility
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    
    # Try codecs in order of preference (browser compatibility)
    codecs_to_try = [
        ('mp4v', 'mp4v'),  # Most compatible, works everywhere
        ('XVID', 'XVID'),  # Good alternative
        ('MJPG', 'MJPG'),  # Motion JPEG, very compatible
    ]
    
    writer = None
    used_codec = None
    for codec_name, fourcc_str in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            used_codec = codec_name
            break
        writer.release()
    
    if writer is None or not writer.isOpened():
        raise RuntimeError("Failed to initialize video writer with any codec")
    
    # Check if GPU is available
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except:
        has_gpu = False
    
    # Calculate estimated processing time BEFORE starting (using video properties)
    estimated_total_time = estimate_processing_time(
        total_frames, width, height, use_yolo, yolo_model_size, has_gpu
    )
    
    # Display initial estimate
    status_text.text(
        f"Initializing...\n"
        f"⏱️ Estimated time: {format_time(estimated_total_time)}\n"
        f"Starting processing..."
    )
    
    # Timing tracking
    start_time = time.time()
    frame_count = 0
    all_counts = []
    all_tracks = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = processor.process_frame(frame, total_frames=total_frames)
        annotated_frame = result['frame']
        counts = result['counts']
        tracks = result['tracks']
        
        # Write frame
        writer.write(annotated_frame)
        
        # Store statistics
        all_counts.append({
            'frame': frame_count,
            'total': counts['total'],
            'up': counts['up'],
            'down': counts['down']
        })
        all_tracks.append(len(tracks))
        
        frame_count += 1
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Calculate remaining time by subtracting elapsed from initial estimate
        remaining_time = max(0, estimated_total_time - elapsed_time)
        
        # Update progress
        progress = (frame_count / total_frames) * 100
        progress_bar.progress(progress / 100)
        
        # Format status text with countdown timer (removed elapsed time)
        status_msg = (
            f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)\n"
            f"⏱️ Time remaining: {format_time(remaining_time)}"
        )
        
        status_text.text(status_msg)
    
    cap.release()
    writer.release()
    
    final_counts = processor.vehicle_counter.get_counts()
    vehicle_stats = processor.vehicle_counter.get_vehicle_statistics()
    
    return {
        'output_video': output_path,
        'total_frames': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
        'final_counts': final_counts,
        'count_history': all_counts,
        'track_history': all_tracks,
        'vehicle_stats': vehicle_stats
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
    # Dynamically scan data folder for videos
    data_dir = Path("data")
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg']
    demo_videos = {}
    
    if data_dir.exists():
        for video_file in sorted(data_dir.iterdir()):
            if video_file.suffix.lower() in video_extensions:
                # Use filename without extension as display name
                display_name = video_file.stem.replace('_', ' ').replace('-', ' ').title()
                demo_videos[display_name] = str(video_file)
    
    if demo_videos:
        selected_demo = st.sidebar.selectbox("Choose demo video:", list(demo_videos.keys()))
        video_path = demo_videos[selected_demo]
        if not os.path.exists(video_path):
            st.sidebar.error(f"Demo video not found: {video_path}")
            video_path = None
    else:
        st.sidebar.warning("⚠️ No demo videos found in data folder")
        video_path = None
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for processing"
    )
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.sidebar.success(f"Video uploaded: {uploaded_file.name}")

# System Information
st.sidebar.subheader("💻 System Status")

# Check GPU availability and store in session state
if 'gpu_available' not in st.session_state:
    try:
        import torch
        st.session_state.gpu_available = torch.cuda.is_available()
        if st.session_state.gpu_available:
            st.session_state.gpu_name = torch.cuda.get_device_name(0)
            st.session_state.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            st.session_state.cuda_version = torch.version.cuda
        else:
            st.session_state.gpu_name = None
            st.session_state.gpu_memory = None
            st.session_state.cuda_version = None
    except ImportError:
        st.session_state.gpu_available = False
        st.session_state.gpu_name = None
    except Exception as e:
        st.session_state.gpu_available = False
        st.session_state.gpu_name = None
        st.session_state.gpu_error = str(e)[:100]

# Display GPU status
if st.session_state.gpu_available:
    st.sidebar.success(f"✅ GPU: {st.session_state.gpu_name}\n💾 VRAM: {st.session_state.gpu_memory:.1f} GB\n🔧 CUDA: {st.session_state.cuda_version}")
    st.sidebar.info("🚀 YOLO will use GPU acceleration")
else:
    st.sidebar.warning("⚠️ GPU not detected\nUsing CPU (slower)")
    if 'gpu_error' in st.session_state:
        st.sidebar.caption(f"Error: {st.session_state.gpu_error}")

# Processing parameters
st.sidebar.subheader("🔧 Processing Parameters")

use_yolo = st.sidebar.checkbox(
    "Use YOLOv8 Detection",
    value=True,
    help="Use deep learning-based detection (more accurate but slower)"
)

if use_yolo:
    # Model size selector - larger models = better accuracy but slower
    yolo_model_size = st.sidebar.selectbox(
        "YOLO Model Size",
        options=['n', 's', 'm', 'l'],
        index=0,
        format_func=lambda x: {
            'n': 'Nano (fastest, lowest accuracy)',
            's': 'Small (fast, good accuracy)',
            'm': 'Medium (balanced, recommended for RTX 4060)',
            'l': 'Large (slower, highest accuracy)'
        }[x],
        help="Larger models are more accurate but slower. RTX 4060 can handle 'm' or 'l' models well."
    )
    
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
    yolo_model_size = 'n'  # Default when YOLO is not used

# ROI configuration
st.sidebar.subheader("📍 Region of Interest (ROI)")
roi_type = st.sidebar.radio(
    "ROI Type:",
    ["Line", "Polygon"],
    index=0,
    help="Line: Count vehicles crossing a line\nPolygon: Count vehicles entering/exiting a region"
)

roi_type_lower = roi_type.lower()

# Display mode
st.sidebar.subheader("🎨 Display Options")
display_mode = st.sidebar.selectbox(
    "Display Mode:",
    ["Clean", "Verbose", "Minimal"],
    index=0,
    help="Clean: Standard visualization\nVerbose: Detailed information\nMinimal: Minimal overlay"
)

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
        
        # Process button
        if st.button("🚀 Process Video", type="primary", use_container_width=True):
            with st.spinner("Initializing processor..."):
                # Initialize processor
                processor = VideoProcessor(
                    use_kalman=use_kalman if not use_yolo else False,
                    roi_type=roi_type_lower,
                    roi_points=st.session_state.roi_points,
                    use_yolo=use_yolo,
                    yolo_confidence=yolo_confidence,
                    min_box_size=min_box_size,
                    yolo_model_size=yolo_model_size if use_yolo else 'n'
                )
                processor.display_mode = display_mode.lower()
                
                st.session_state.processor = processor
            
            # Process video
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing video frames..."):
                results = process_video_streamlit(
                    video_path,
                    processor,
                    progress_bar,
                    status_text,
                    use_yolo=use_yolo,
                    yolo_model_size=yolo_model_size if use_yolo else 'n'
                )
            
            if results:
                st.session_state.processing_results = results
                st.success("✅ Video processing completed!")
                
                # Display output video with interactive controls
                st.subheader("📹 Processed Video")
                
                # Speed control
                col_speed1, col_speed2 = st.columns([1, 3])
                with col_speed1:
                    speed_options = {"0.5x": 0.5, "1x": 1.0, "1.5x": 1.5, "2x": 2.0}
                    selected_speed_label = st.selectbox(
                        "Playback Speed:",
                        list(speed_options.keys()),
                        index=1,
                        key="video_speed"
                    )
                    selected_speed = speed_options[selected_speed_label]
                
                # Verify video file exists and is accessible
                video_path = results['output_video']
                video_width = results.get('width', 640)
                video_height = results.get('height', 480)
                
                if not os.path.exists(video_path):
                    st.error(f"❌ Video file not found: {video_path}")
                else:
                    # Calculate aspect ratio for responsive sizing
                    aspect_ratio = video_width / video_height if video_height > 0 else 16/9
                    max_width = min(1200, video_width)  # Maximum width for video player
                    display_width = max_width
                    display_height = int(display_width / aspect_ratio)
                    
                    try:
                        # Read video file and encode to base64 for embedding
                        with open(video_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                            video_base64 = base64.b64encode(video_bytes).decode()
                        
                        # Create HTML5 video player with proper sizing and controls
                        video_html = f"""
                        <div style="width: 100%; max-width: {display_width}px; margin: 20px auto; background: #f0f0f0; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            <video 
                                id="processedVideo" 
                                width="{display_width}" 
                                height="{display_height}"
                                controls 
                                preload="metadata"
                                style="width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.2); background: #000; display: block; min-height: 300px;">
                                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                        </div>
                        <script>
                            (function() {{
                                function setVideoSpeed() {{
                                    var video = document.getElementById('processedVideo');
                                    if (video) {{
                                        video.playbackRate = {selected_speed};
                                    }}
                                }}
                                
                                // Wait for video element to be available
                                var checkVideo = setInterval(function() {{
                                    var video = document.getElementById('processedVideo');
                                    if (video) {{
                                        clearInterval(checkVideo);
                                        
                                        // Set initial speed
                                        setVideoSpeed();
                                        
                                        // Re-apply speed on various events
                                        video.addEventListener('loadedmetadata', setVideoSpeed);
                                        video.addEventListener('play', setVideoSpeed);
                                        video.addEventListener('canplay', setVideoSpeed);
                                        video.addEventListener('loadeddata', setVideoSpeed);
                                        
                                        // Monitor for speed changes (in case user changes selectbox)
                                        setInterval(function() {{
                                            if (video && video.playbackRate !== {selected_speed}) {{
                                                video.playbackRate = {selected_speed};
                                            }}
                                        }}, 200);
                                    }}
                                }}, 100);
                                
                                // Also try immediately in case video is already loaded
                                setTimeout(setVideoSpeed, 50);
                            }})();
                        </script>
                        """
                        st.markdown(video_html, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.warning(f"⚠️ Could not load video as embedded player: {str(e)[:100]}")
                        st.caption("Using Streamlit's native video player as fallback...")
                        # Fallback to Streamlit's video player
                        try:
                            st.video(video_path)
                        except Exception as e2:
                            st.error(f"❌ Error with fallback player: {str(e2)[:100]}")
                            st.info(f"Video file location: {video_path}")
                            # Provide download option
                            with open(video_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Download Processed Video",
                                    data=f.read(),
                                    file_name="tracked_video.mp4",
                                    mime="video/mp4"
                                )
                
                # Report generation section
                st.subheader("📊 Analysis Report")
                if st.button("📄 Generate Professional Report", type="primary", use_container_width=True):
                    with st.spinner("Generating report..."):
                        report_generator = ReportGenerator()
                        video_info = {
                            'width': results.get('width'),
                            'height': results.get('height'),
                            'fps': results.get('fps'),
                            'total_frames': results.get('total_frames'),
                            'duration': results.get('total_frames', 0) / results.get('fps', 1) if results.get('fps', 0) > 0 else 0
                        }
                        html_report = report_generator.generate_html_report(results, video_info)
                        st.session_state.report_html = html_report
                        
                        st.success("✅ Report generated successfully!")
                
                # Display and download report
                if 'report_html' in st.session_state and st.session_state.report_html:
                    st.markdown("### 📋 Report Preview")
                    try:
                        import streamlit.components.v1 as components
                        components.html(st.session_state.report_html, height=800, scrolling=True)
                    except:
                        # Fallback: show download button only
                        st.info("📄 Report generated! Download to view in your browser.")
                    
                    # Download report
                    st.download_button(
                        label="💾 Download Report (HTML)",
                        data=st.session_state.report_html,
                        file_name=f"vehicle_tracking_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
            else:
                st.error("❌ Failed to process video")
    else:
        st.warning("⚠️ Please select or upload a video file to begin processing.")

with col2:
    st.subheader("📈 Statistics")
    
    if st.session_state.processing_results:
        results = st.session_state.processing_results
        final_counts = results['final_counts']
        
        # Metrics
        st.metric("Total Vehicles", final_counts['total'])
        st.metric("Direction Up", final_counts['up'])
        st.metric("Direction Down", final_counts['down'])
        
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

