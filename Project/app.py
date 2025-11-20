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
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = None
if 'roi_type' not in st.session_state:
    st.session_state.roi_type = 'line'
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = None
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

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
    
    # Setup video writer
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
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
        progress = (frame_count / total_frames) * 100
        progress_bar.progress(progress / 100)
        status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
    
    cap.release()
    writer.release()
    
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
    demo_videos = {
        "Sample Traffic Test 1": "data/sample_traffic_test.mp4",
        "Sample Traffic Test 2": "data/sample_traffic_test2.mp4"
    }
    selected_demo = st.sidebar.selectbox("Choose demo video:", list(demo_videos.keys()))
    video_path = demo_videos[selected_demo]
    if not os.path.exists(video_path):
        st.sidebar.error(f"Demo video not found: {video_path}")
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
                    min_box_size=min_box_size
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
                    status_text
                )
            
            if results:
                st.session_state.processing_results = results
                st.success("✅ Video processing completed!")
                
                # Display output video
                st.subheader("📹 Processed Video")
                st.video(results['output_video'])
                
                # Download button
                with open(results['output_video'], 'rb') as f:
                    st.download_button(
                        label="💾 Download Processed Video",
                        data=f.read(),
                        file_name="tracked_video.mp4",
                        mime="video/mp4",
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

