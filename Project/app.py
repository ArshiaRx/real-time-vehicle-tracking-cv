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

# Custom CSS - Enhanced Professional Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Enhanced Neon Glow Animations - Brighter & Sharper */
    @keyframes neon-pulse {
        0%, 100% { 
            text-shadow: 0 0 5px #B794F6, 
                         0 0 10px #B794F6, 
                         0 0 15px #B794F6, 
                         0 0 20px #9D8DF1,
                         0 0 30px #9D8DF1,
                         0 0 40px #7C6FD8;
            filter: brightness(1) drop-shadow(0 0 5px #B794F6);
        }
        50% { 
            text-shadow: 0 0 10px #B794F6, 
                         0 0 20px #B794F6, 
                         0 0 30px #9D8DF1,
                         0 0 40px #9D8DF1,
                         0 0 50px #7C6FD8,
                         0 0 60px #5B51B8;
            filter: brightness(1.3) drop-shadow(0 0 10px #B794F6);
        }
    }
    
    @keyframes neon-breathe {
        0%, 100% { 
            opacity: 0.9; 
            filter: brightness(1);
        }
        50% { 
            opacity: 1; 
            filter: brightness(1.4);
        }
    }
    
    @keyframes neon-flow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes neon-shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    @keyframes neon-border-pulse {
        0%, 100% { 
            box-shadow: 0 0 5px rgba(183, 148, 246, 0.9), 
                        0 0 10px rgba(183, 148, 246, 0.7), 
                        0 0 15px rgba(157, 141, 241, 0.6),
                        0 0 20px rgba(157, 141, 241, 0.5),
                        0 0 30px rgba(157, 141, 241, 0.4),
                        inset 0 0 10px rgba(183, 148, 246, 0.3),
                        inset 0 0 20px rgba(183, 148, 246, 0.2);
        }
        50% { 
            box-shadow: 0 0 10px rgba(183, 148, 246, 1), 
                        0 0 20px rgba(183, 148, 246, 0.8), 
                        0 0 30px rgba(157, 141, 241, 0.7),
                        0 0 40px rgba(157, 141, 241, 0.6),
                        0 0 50px rgba(157, 141, 241, 0.5),
                        inset 0 0 15px rgba(183, 148, 246, 0.4),
                        inset 0 0 30px rgba(183, 148, 246, 0.3);
        }
    }
    
    @keyframes neon-green-pulse {
        0%, 100% { 
            box-shadow: 0 0 5px rgba(46, 204, 113, 0.9), 
                        0 0 10px rgba(46, 204, 113, 0.7), 
                        0 0 15px rgba(46, 204, 113, 0.6),
                        0 0 20px rgba(46, 204, 113, 0.5),
                        0 0 30px rgba(46, 204, 113, 0.4),
                        inset 0 0 10px rgba(46, 204, 113, 0.3);
        }
        50% { 
            box-shadow: 0 0 10px rgba(46, 204, 113, 1), 
                        0 0 20px rgba(46, 204, 113, 0.8), 
                        0 0 30px rgba(46, 204, 113, 0.7),
                        0 0 40px rgba(46, 204, 113, 0.6),
                        0 0 50px rgba(46, 204, 113, 0.5),
                        inset 0 0 15px rgba(46, 204, 113, 0.4);
        }
    }
    
    @keyframes neon-red-pulse {
        0%, 100% { 
            box-shadow: 0 0 5px rgba(231, 76, 60, 0.9), 
                        0 0 10px rgba(231, 76, 60, 0.7), 
                        0 0 15px rgba(231, 76, 60, 0.6),
                        0 0 20px rgba(231, 76, 60, 0.5),
                        0 0 30px rgba(231, 76, 60, 0.4),
                        inset 0 0 10px rgba(231, 76, 60, 0.3);
        }
        50% { 
            box-shadow: 0 0 10px rgba(231, 76, 60, 1), 
                        0 0 20px rgba(231, 76, 60, 0.8), 
                        0 0 30px rgba(231, 76, 60, 0.7),
                        0 0 40px rgba(231, 76, 60, 0.6),
                        0 0 50px rgba(231, 76, 60, 0.5),
                        inset 0 0 15px rgba(231, 76, 60, 0.4);
        }
    }
    
    /* Main Header with Enhanced Bright Neon Glow */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #B794F6 0%, #9D8DF1 30%, #7C6FD8 60%, #5B51B8 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.75rem;
        letter-spacing: 0.02em;
        animation: fadeInDown 0.8s ease-out, neon-pulse 2.5s ease-in-out infinite, neon-flow 4s ease infinite;
        filter: drop-shadow(0 0 5px #B794F6) drop-shadow(0 0 10px #9D8DF1) drop-shadow(0 0 15px #7C6FD8);
        text-shadow: 0 0 5px #B794F6, 
                     0 0 10px #B794F6, 
                     0 0 15px #9D8DF1,
                     0 0 20px #9D8DF1,
                     0 0 30px #7C6FD8;
    }
    
    /* Subtitle with Enhanced Neon Glow */
    .subtitle {
        text-align: center;
        color: var(--text-color);
        opacity: 0.85;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
        animation: fadeIn 1s ease-out;
        text-shadow: 0 0 8px rgba(183, 148, 246, 0.6), 
                     0 0 15px rgba(157, 141, 241, 0.5), 
                     0 0 25px rgba(157, 141, 241, 0.4);
    }
    
    /* Enhanced Metric Cards with Sharp Neon Glow */
    .metric-card {
        background: linear-gradient(135deg, var(--secondary-background-color) 0%, rgba(183, 148, 246, 0.08) 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid rgba(183, 148, 246, 0.5);
        box-shadow: 0 0 5px rgba(183, 148, 246, 0.8), 
                    0 0 10px rgba(183, 148, 246, 0.6), 
                    0 0 15px rgba(157, 141, 241, 0.5),
                    0 0 20px rgba(157, 141, 241, 0.4),
                    0 0 30px rgba(157, 141, 241, 0.3),
                    inset 0 0 10px rgba(183, 148, 246, 0.2),
                    0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 10px rgba(183, 148, 246, 1), 
                    0 0 20px rgba(183, 148, 246, 0.8), 
                    0 0 30px rgba(157, 141, 241, 0.7),
                    0 0 40px rgba(157, 141, 241, 0.6),
                    0 0 50px rgba(157, 141, 241, 0.5),
                    inset 0 0 15px rgba(183, 148, 246, 0.3),
                    0 8px 12px rgba(0, 0, 0, 0.15);
        border-color: rgba(183, 148, 246, 0.8);
    }
    
    /* Info Cards with Sharp Neon Glow */
    .info-card {
        background: linear-gradient(135deg, rgba(183, 148, 246, 0.12) 0%, rgba(157, 141, 241, 0.06) 100%);
        padding: 1.25rem;
        border-radius: 0.75rem;
        border: 2px solid rgba(183, 148, 246, 0.5);
        margin: 0.75rem 0;
        box-shadow: 0 0 5px rgba(183, 148, 246, 0.7), 
                    0 0 10px rgba(183, 148, 246, 0.5), 
                    0 0 15px rgba(157, 141, 241, 0.4),
                    0 0 20px rgba(157, 141, 241, 0.3),
                    inset 0 0 8px rgba(183, 148, 246, 0.2),
                    0 2px 8px rgba(0, 0, 0, 0.1);
        animation: slideInLeft 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        box-shadow: 0 0 10px rgba(183, 148, 246, 0.9), 
                    0 0 20px rgba(183, 148, 246, 0.7), 
                    0 0 30px rgba(157, 141, 241, 0.6),
                    0 0 40px rgba(157, 141, 241, 0.5),
                    inset 0 0 12px rgba(183, 148, 246, 0.3),
                    0 4px 12px rgba(0, 0, 0, 0.15);
        border-color: rgba(183, 148, 246, 0.8);
    }
    
    /* Feature Badge with Sharp Neon Glow */
    .feature-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        background: linear-gradient(135deg, #B794F6 0%, #9D8DF1 50%, #7C6FD8 100%);
        background-size: 200% 200%;
        color: white;
        border-radius: 2rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        border: 1px solid rgba(183, 148, 246, 0.6);
        box-shadow: 0 0 5px rgba(183, 148, 246, 0.9), 
                    0 0 10px rgba(183, 148, 246, 0.7), 
                    0 0 15px rgba(157, 141, 241, 0.6),
                    0 0 20px rgba(157, 141, 241, 0.5),
                    inset 0 0 8px rgba(183, 148, 246, 0.3),
                    0 2px 8px rgba(0, 0, 0, 0.2);
        animation: fadeIn 0.8s ease-out, neon-border-pulse 2.5s ease-in-out infinite, neon-flow 3s ease infinite;
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.8), 
                     0 0 10px rgba(255, 255, 255, 0.5);
        transition: all 0.3s ease;
    }
    
    .feature-badge:hover {
        box-shadow: 0 0 10px rgba(183, 148, 246, 1), 
                    0 0 20px rgba(183, 148, 246, 0.8), 
                    0 0 30px rgba(157, 141, 241, 0.7),
                    0 0 40px rgba(157, 141, 241, 0.6),
                    inset 0 0 12px rgba(183, 148, 246, 0.4),
                    0 4px 12px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px) scale(1.05);
        filter: brightness(1.2);
    }
    
    /* Stats Container with Sharp Neon Glow */
    .stats-container {
        background: linear-gradient(135deg, rgba(183, 148, 246, 0.1) 0%, rgba(157, 141, 241, 0.04) 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid rgba(183, 148, 246, 0.5);
        margin: 0.75rem 0;
        box-shadow: 0 0 5px rgba(183, 148, 246, 0.7), 
                    0 0 10px rgba(183, 148, 246, 0.5), 
                    0 0 15px rgba(157, 141, 241, 0.4),
                    0 0 20px rgba(157, 141, 241, 0.3),
                    inset 0 0 10px rgba(183, 148, 246, 0.2),
                    0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .stats-container:hover {
        box-shadow: 0 0 10px rgba(183, 148, 246, 0.9), 
                    0 0 20px rgba(183, 148, 246, 0.7), 
                    0 0 30px rgba(157, 141, 241, 0.6),
                    0 0 40px rgba(157, 141, 241, 0.5),
                    inset 0 0 15px rgba(183, 148, 246, 0.3),
                    0 6px 16px rgba(0, 0, 0, 0.12);
        border-color: rgba(183, 148, 246, 0.8);
    }
    
    /* Enhanced Sidebar with Sharp Neon Glow */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(183, 148, 246, 0.1) 0%, rgba(157, 141, 241, 0.04) 100%);
        border-right: 3px solid rgba(183, 148, 246, 0.6);
        box-shadow: 0 0 15px rgba(183, 148, 246, 0.4), 
                    0 0 30px rgba(157, 141, 241, 0.3), 
                    inset -3px 0 25px rgba(183, 148, 246, 0.2);
        transition: all 0.3s ease;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Sidebar Toggle Button Styling */
    button[key="sidebar_toggle"] {
        background: linear-gradient(135deg, #9D8DF1 0%, #7C6FD8 100%) !important;
        border: none !important;
        border-radius: 50% !important;
        width: 45px !important;
        height: 45px !important;
        min-width: 45px !important;
        padding: 0 !important;
        box-shadow: 0 0 15px rgba(157, 141, 241, 0.6), 
                    0 0 30px rgba(157, 141, 241, 0.4),
                    0 4px 12px rgba(0, 0, 0, 0.3) !important;
        font-size: 1.2rem !important;
        color: white !important;
        transition: all 0.3s ease !important;
        animation: neon-border-pulse 3s ease-in-out infinite !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
    }
    
    button[key="sidebar_toggle"]:hover {
        box-shadow: 0 0 20px rgba(157, 141, 241, 0.8), 
                    0 0 40px rgba(157, 141, 241, 0.6),
                    0 6px 16px rgba(0, 0, 0, 0.4) !important;
        transform: scale(1.1) !important;
    }
    
    /* Collapsed Sidebar */
    .sidebar-collapsed section[data-testid="stSidebar"] {
        width: 0 !important;
        min-width: 0 !important;
        overflow: hidden;
        transition: width 0.3s ease;
    }
    
    .sidebar-collapsed section[data-testid="stSidebar"] > div {
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s ease;
    }
    
    /* Professional Spacing System */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100%;
    }
    
    /* Consistent Column Spacing */
    .stColumn {
        padding: 0.75rem;
    }
    
    /* Typography Spacing Hierarchy */
    h1 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
        line-height: 1.2;
    }
    
    h2 {
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        font-size: 1.75rem;
        line-height: 1.3;
    }
    
    h3 {
        margin-top: 1.25rem;
        margin-bottom: 0.75rem;
        font-size: 1.25rem;
        line-height: 1.4;
    }
    
    h4 {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        line-height: 1.5;
    }
    
    /* Consistent Section Spacing */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Processing Status Spacing */
    .processing-status {
        margin: 1rem 0;
        padding: 1rem;
    }
    
    /* Chart Container Spacing */
    .chart-container {
        margin: 1rem 0;
    }
    
    /* Info Card Spacing */
    .info-card {
        margin: 1rem 0;
    }
    
    /* Stats Container Spacing */
    .stats-container {
        margin: 1rem 0;
    }
    
    /* Sidebar Headers with Sharp Neon Glow */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-color);
        text-shadow: 0 0 8px rgba(183, 148, 246, 0.6), 
                     0 0 15px rgba(157, 141, 241, 0.5), 
                     0 0 25px rgba(157, 141, 241, 0.4);
        border-bottom: 2px solid rgba(183, 148, 246, 0.5);
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 0 rgba(183, 148, 246, 0.3);
        font-weight: 700;
    }
    
    section[data-testid="stSidebar"] h1:first-child,
    section[data-testid="stSidebar"] h2:first-child,
    section[data-testid="stSidebar"] h3:first-child {
        margin-top: 0;
    }
    
    /* Sidebar Input Grouping */
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 1rem;
    }
    
    section[data-testid="stSidebar"] .stSubheader {
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Sidebar Radio Buttons with Sharp Neon Glow */
    section[data-testid="stSidebar"] label[data-baseweb="radio"] {
        color: var(--text-color);
        margin-bottom: 0.5rem;
    }
    
    section[data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] {
        box-shadow: 0 0 8px rgba(183, 148, 246, 0.8), 
                    0 0 15px rgba(183, 148, 246, 0.6), 
                    0 0 25px rgba(157, 141, 241, 0.5);
    }
    
    /* Sidebar Checkboxes with Sharp Neon Glow */
    section[data-testid="stSidebar"] [data-baseweb="checkbox"] [aria-checked="true"] {
        box-shadow: 0 0 8px rgba(183, 148, 246, 0.8), 
                    0 0 15px rgba(183, 148, 246, 0.6), 
                    0 0 25px rgba(157, 141, 241, 0.5);
    }
    
    /* Sidebar Sliders with Sharp Neon Glow */
    section[data-testid="stSidebar"] [data-baseweb="slider"] {
        margin: 0.75rem 0;
    }
    
    section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
        box-shadow: 0 0 8px rgba(183, 148, 246, 0.8), 
                    0 0 15px rgba(183, 148, 246, 0.6), 
                    0 0 25px rgba(157, 141, 241, 0.5);
    }
    
    /* Sidebar Selectbox/Dropdown with Sharp Neon Glow */
    section[data-testid="stSidebar"] [data-baseweb="select"] {
        border: 1px solid rgba(183, 148, 246, 0.5);
        box-shadow: 0 0 5px rgba(183, 148, 246, 0.4), 
                    0 0 10px rgba(157, 141, 241, 0.3);
        margin-bottom: 0.75rem;
    }
    
    section[data-testid="stSidebar"] [data-baseweb="select"]:focus,
    section[data-testid="stSidebar"] [data-baseweb="select"]:hover {
        border-color: rgba(183, 148, 246, 0.8);
        box-shadow: 0 0 10px rgba(183, 148, 246, 0.6), 
                    0 0 20px rgba(183, 148, 246, 0.5), 
                    0 0 30px rgba(157, 141, 241, 0.4);
    }
    
    /* Sidebar Section Separators with Neon Glow */
    section[data-testid="stSidebar"] hr {
        border: none;
        border-top: 2px solid rgba(183, 148, 246, 0.4);
        box-shadow: 0 0 8px rgba(183, 148, 246, 0.5), 
                    0 0 15px rgba(157, 141, 241, 0.4);
        margin: 1.5rem 0;
    }
    
    /* Sidebar Expanders with Sharp Neon Glow */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(183, 148, 246, 0.12) 0%, rgba(157, 141, 241, 0.06) 100%);
        border: 1px solid rgba(183, 148, 246, 0.4);
        box-shadow: 0 0 5px rgba(183, 148, 246, 0.4), 
                    0 0 10px rgba(157, 141, 241, 0.3);
        margin-bottom: 0.5rem;
    }
    
    section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        box-shadow: 0 0 10px rgba(183, 148, 246, 0.6), 
                    0 0 20px rgba(183, 148, 246, 0.5), 
                    0 0 30px rgba(157, 141, 241, 0.4);
        border-color: rgba(183, 148, 246, 0.7);
    }
    
    /* Sidebar Input Grouping */
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 1rem;
    }
    
    section[data-testid="stSidebar"] .stSubheader {
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Button Enhancements with Sharp Neon Glow */
    .stButton > button {
        font-weight: 600;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        border: 2px solid rgba(183, 148, 246, 0.6);
        transition: all 0.3s ease;
        box-shadow: 0 0 5px rgba(183, 148, 246, 0.9), 
                    0 0 10px rgba(183, 148, 246, 0.7), 
                    0 0 15px rgba(157, 141, 241, 0.6),
                    0 0 20px rgba(157, 141, 241, 0.5),
                    inset 0 0 8px rgba(183, 148, 246, 0.3),
                    0 4px 12px rgba(0, 0, 0, 0.2);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.9rem;
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.8), 
                     0 0 10px rgba(255, 255, 255, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: neon-shimmer 3s infinite;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 10px rgba(183, 148, 246, 1), 
                    0 0 20px rgba(183, 148, 246, 0.8), 
                    0 0 30px rgba(157, 141, 241, 0.7),
                    0 0 40px rgba(157, 141, 241, 0.6),
                    inset 0 0 12px rgba(183, 148, 246, 0.4),
                    0 6px 20px rgba(0, 0, 0, 0.3);
        text-shadow: 0 0 10px rgba(255, 255, 255, 1), 
                     0 0 20px rgba(255, 255, 255, 0.8);
        border-color: rgba(183, 148, 246, 0.9);
        filter: brightness(1.15);
    }
    
    /* Download Button with Sharp Green Neon Glow */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        font-weight: 600;
        border-radius: 0.75rem;
        border: 2px solid rgba(46, 204, 113, 0.7);
        box-shadow: 0 0 5px rgba(46, 204, 113, 0.9), 
                    0 0 10px rgba(46, 204, 113, 0.7), 
                    0 0 15px rgba(46, 204, 113, 0.6),
                    0 0 20px rgba(46, 204, 113, 0.5),
                    inset 0 0 8px rgba(46, 204, 113, 0.3),
                    0 4px 12px rgba(0, 0, 0, 0.2);
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.8), 
                     0 0 10px rgba(255, 255, 255, 0.5);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stDownloadButton > button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: neon-shimmer 3s infinite;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 10px rgba(46, 204, 113, 1), 
                    0 0 20px rgba(46, 204, 113, 0.8), 
                    0 0 30px rgba(46, 204, 113, 0.7),
                    0 0 40px rgba(46, 204, 113, 0.6),
                    inset 0 0 12px rgba(46, 204, 113, 0.4),
                    0 6px 20px rgba(0, 0, 0, 0.3);
        text-shadow: 0 0 10px rgba(255, 255, 255, 1), 
                     0 0 20px rgba(255, 255, 255, 0.8);
        border-color: rgba(46, 204, 113, 0.9);
        filter: brightness(1.15);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #9D8DF1 0%, #7C6FD8 50%, #5B51B8 100%);
        border-radius: 1rem;
        height: 1rem;
        box-shadow: 0 2px 8px rgba(157, 141, 241, 0.3);
    }
    
    /* Live Counter Display with Better Spacing */
    .live-counter-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
        justify-items: center;
        max-width: 100%;
    }
    
    .live-counter {
        flex: 1;
        min-width: 160px;
        max-width: 220px;
        padding: 1rem 0.75rem;
        border-radius: 0.75rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .live-counter::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        animation: pulse 2s infinite;
        box-shadow: 0 0 10px currentColor;
    }
    
    .live-counter.up {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.18) 0%, rgba(46, 204, 113, 0.08) 100%);
        border: 2px solid rgba(46, 204, 113, 0.7);
        box-shadow: 0 0 5px rgba(46, 204, 113, 0.9), 
                    0 0 10px rgba(46, 204, 113, 0.7), 
                    0 0 15px rgba(46, 204, 113, 0.6),
                    0 0 20px rgba(46, 204, 113, 0.5),
                    0 0 30px rgba(46, 204, 113, 0.4),
                    inset 0 0 10px rgba(46, 204, 113, 0.3),
                    0 2px 8px rgba(0, 0, 0, 0.12);
        animation: neon-green-pulse 2.5s ease-in-out infinite, neon-breathe 3s ease-in-out infinite;
    }
    
    .live-counter.up::before {
        background: linear-gradient(90deg, #2ecc71, #27ae60);
        box-shadow: 0 0 10px #2ecc71, 0 0 20px #2ecc71, 0 0 30px rgba(46, 204, 113, 0.6);
    }
    
    .live-counter.up:hover {
        box-shadow: 0 0 10px rgba(46, 204, 113, 1), 
                    0 0 20px rgba(46, 204, 113, 0.8), 
                    0 0 30px rgba(46, 204, 113, 0.7),
                    0 0 40px rgba(46, 204, 113, 0.6),
                    0 0 50px rgba(46, 204, 113, 0.5),
                    inset 0 0 15px rgba(46, 204, 113, 0.4),
                    0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
        filter: brightness(1.2);
    }
    
    .live-counter.down {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.18) 0%, rgba(231, 76, 60, 0.08) 100%);
        border: 2px solid rgba(231, 76, 60, 0.7);
        box-shadow: 0 0 5px rgba(231, 76, 60, 0.9), 
                    0 0 10px rgba(231, 76, 60, 0.7), 
                    0 0 15px rgba(231, 76, 60, 0.6),
                    0 0 20px rgba(231, 76, 60, 0.5),
                    0 0 30px rgba(231, 76, 60, 0.4),
                    inset 0 0 10px rgba(231, 76, 60, 0.3),
                    0 2px 8px rgba(0, 0, 0, 0.12);
        animation: neon-red-pulse 2.5s ease-in-out infinite, neon-breathe 3s ease-in-out infinite;
    }
    
    .live-counter.down::before {
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        box-shadow: 0 0 10px #e74c3c, 0 0 20px #e74c3c, 0 0 30px rgba(231, 76, 60, 0.6);
    }
    
    .live-counter.down:hover {
        box-shadow: 0 0 10px rgba(231, 76, 60, 1), 
                    0 0 20px rgba(231, 76, 60, 0.8), 
                    0 0 30px rgba(231, 76, 60, 0.7),
                    0 0 40px rgba(231, 76, 60, 0.6),
                    0 0 50px rgba(231, 76, 60, 0.5),
                    inset 0 0 15px rgba(231, 76, 60, 0.4),
                    0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
        filter: brightness(1.2);
    }
    
    .live-counter.total {
        background: linear-gradient(135deg, rgba(183, 148, 246, 0.18) 0%, rgba(157, 141, 241, 0.08) 100%);
        border: 2px solid rgba(183, 148, 246, 0.7);
        box-shadow: 0 0 5px rgba(183, 148, 246, 0.9), 
                    0 0 10px rgba(183, 148, 246, 0.7), 
                    0 0 15px rgba(157, 141, 241, 0.6),
                    0 0 20px rgba(157, 141, 241, 0.5),
                    0 0 30px rgba(157, 141, 241, 0.4),
                    inset 0 0 10px rgba(183, 148, 246, 0.3),
                    0 2px 8px rgba(0, 0, 0, 0.12);
        animation: neon-border-pulse 2.5s ease-in-out infinite, neon-breathe 3s ease-in-out infinite;
    }
    
    .live-counter.total::before {
        background: linear-gradient(90deg, #B794F6, #9D8DF1, #7C6FD8);
        box-shadow: 0 0 10px #B794F6, 0 0 20px #9D8DF1, 0 0 30px rgba(157, 141, 241, 0.6);
    }
    
    .live-counter.total:hover {
        box-shadow: 0 0 10px rgba(183, 148, 246, 1), 
                    0 0 20px rgba(183, 148, 246, 0.8), 
                    0 0 30px rgba(157, 141, 241, 0.7),
                    0 0 40px rgba(157, 141, 241, 0.6),
                    0 0 50px rgba(157, 141, 241, 0.5),
                    inset 0 0 15px rgba(183, 148, 246, 0.4),
                    0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
        filter: brightness(1.2);
    }
    
    .counter-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
        margin-bottom: 0.35rem;
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.6), 
                     0 0 10px rgba(255, 255, 255, 0.4);
    }
    
    .counter-value {
        font-size: 2.25rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.35rem 0;
        animation: neon-breathe 2s ease-in-out infinite;
    }
    
    .live-counter.up .counter-value {
        text-shadow: 0 0 5px #2ecc71, 
                     0 0 10px #2ecc71, 
                     0 0 15px #2ecc71,
                     0 0 20px rgba(46, 204, 113, 0.8),
                     0 0 30px rgba(46, 204, 113, 0.6);
    }
    
    .live-counter.down .counter-value {
        text-shadow: 0 0 5px #e74c3c, 
                     0 0 10px #e74c3c, 
                     0 0 15px #e74c3c,
                     0 0 20px rgba(231, 76, 60, 0.8),
                     0 0 30px rgba(231, 76, 60, 0.6);
    }
    
    .live-counter.total .counter-value {
        text-shadow: 0 0 5px #B794F6, 
                     0 0 10px #B794F6, 
                     0 0 15px #9D8DF1,
                     0 0 20px rgba(157, 141, 241, 0.8),
                     0 0 30px rgba(157, 141, 241, 0.6);
    }
    
    .counter-icon {
        font-size: 1.5rem;
        margin-bottom: 0.35rem;
        opacity: 0.9;
    }
    
    .counter-direction {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 0.35rem;
    }
    
    .processing-status {
        background: linear-gradient(135deg, rgba(183, 148, 246, 0.12) 0%, rgba(157, 141, 241, 0.06) 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        border: 2px solid rgba(183, 148, 246, 0.6);
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 0 5px rgba(183, 148, 246, 0.7), 
                    0 0 10px rgba(183, 148, 246, 0.5), 
                    0 0 15px rgba(157, 141, 241, 0.4),
                    0 0 20px rgba(157, 141, 241, 0.3),
                    inset 0 0 8px rgba(183, 148, 246, 0.2);
    }
    
    .processing-status[style*="green"] {
        border-color: rgba(46, 204, 113, 0.7);
        box-shadow: 0 0 5px rgba(46, 204, 113, 0.8), 
                    0 0 10px rgba(46, 204, 113, 0.6), 
                    0 0 15px rgba(46, 204, 113, 0.5),
                    0 0 20px rgba(46, 204, 113, 0.4),
                    inset 0 0 8px rgba(46, 204, 113, 0.3);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.4rem 0.85rem;
        background: linear-gradient(135deg, #B794F6 0%, #9D8DF1 50%, #7C6FD8 100%);
        background-size: 200% 200%;
        color: white;
        border-radius: 2rem;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(183, 148, 246, 0.6);
        animation: pulse 2s infinite, neon-border-pulse 2.5s ease-in-out infinite, neon-flow 3s ease infinite;
        box-shadow: 0 0 5px rgba(183, 148, 246, 0.9), 
                    0 0 10px rgba(183, 148, 246, 0.7), 
                    0 0 15px rgba(157, 141, 241, 0.6),
                    0 0 20px rgba(157, 141, 241, 0.5),
                    inset 0 0 6px rgba(183, 148, 246, 0.3);
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.8), 
                     0 0 10px rgba(255, 255, 255, 0.5);
    }
    
    .status-badge[style*="green"] {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        border-color: rgba(46, 204, 113, 0.7);
        box-shadow: 0 0 5px rgba(46, 204, 113, 0.9), 
                    0 0 10px rgba(46, 204, 113, 0.7), 
                    0 0 15px rgba(46, 204, 113, 0.6),
                    0 0 20px rgba(46, 204, 113, 0.5),
                    inset 0 0 6px rgba(46, 204, 113, 0.3);
        animation: pulse 2s infinite, neon-green-pulse 2.5s ease-in-out infinite;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-color) 0%, #7C6FD8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.8;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(157, 141, 241, 0.1) 0%, rgba(157, 141, 241, 0.05) 100%);
        border-radius: 0.5rem;
        font-weight: 600;
        padding: 0.75rem 1rem;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        border-radius: 0.5rem;
        font-weight: 600;
        padding: 0 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #9D8DF1 0%, #7C6FD8 100%);
        box-shadow: 0 4px 12px rgba(157, 141, 241, 0.3);
    }
    
    /* Chart Container with Sharp Neon Glow */
    .chart-container {
        background: linear-gradient(135deg, var(--secondary-background-color) 0%, rgba(183, 148, 246, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid rgba(183, 148, 246, 0.5);
        box-shadow: 0 0 5px rgba(183, 148, 246, 0.6), 
                    0 0 10px rgba(183, 148, 246, 0.4), 
                    0 0 15px rgba(157, 141, 241, 0.3),
                    0 0 20px rgba(157, 141, 241, 0.2),
                    inset 0 0 10px rgba(183, 148, 246, 0.15),
                    0 4px 12px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        box-shadow: 0 0 10px rgba(183, 148, 246, 0.8), 
                    0 0 20px rgba(183, 148, 246, 0.6), 
                    0 0 30px rgba(157, 141, 241, 0.5),
                    0 0 40px rgba(157, 141, 241, 0.4),
                    inset 0 0 15px rgba(183, 148, 246, 0.25),
                    0 6px 16px rgba(0, 0, 0, 0.12);
        border-color: rgba(183, 148, 246, 0.8);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.8;
        }
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: var(--primary-color) transparent transparent transparent;
    }
    
    /* Success/Error/Warning Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 0.75rem;
        border-left-width: 4px;
        padding: 1rem 1.5rem;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Video Container */
    video {
        border-radius: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .metric-card {
            padding: 1rem;
        }
        .live-counter {
            min-width: 140px;
            max-width: 180px;
            padding: 0.75rem 0.5rem;
        }
        .counter-value {
            font-size: 1.75rem;
        }
    }
    
    /* Additional Layout Optimizations */
    /* Reduce header spacing */
    .main-header {
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        margin-bottom: 1.5rem;
    }
    
    /* Compact feature badges */
    .feature-badge {
        margin: 0.2rem;
        padding: 0.4rem 0.8rem;
        font-size: 0.8rem;
    }
    
    /* Optimize main content columns */
    [data-testid="column"] {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    
    /* Better video info card spacing */
    .info-card {
        margin: 0.75rem 0;
        padding: 1rem;
    }
    
    /* Compact chart containers */
    .chart-container {
        padding: 1.25rem;
        margin: 0.75rem 0;
    }
    
    /* Optimize stats container */
    .stats-container {
        padding: 1.5rem;
        margin: 0.75rem 0;
    }
    
    /* Better spacing for metrics */
    div[data-testid="stMetricContainer"] {
        margin-bottom: 0.5rem;
    }
    
    /* Compact expanders */
    .streamlit-expanderHeader {
        padding: 0.6rem 0.85rem;
    }
    
    /* Optimize sidebar spacing */
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 0.75rem;
    }
    
    section[data-testid="stSidebar"] .stSubheader {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
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
if 'sidebar_collapsed' not in st.session_state:
    st.session_state.sidebar_collapsed = False

def process_video_streamlit(video_path: str, processor: VideoProcessor, 
                            progress_bar, status_text, live_counter_placeholder=None) -> dict:
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
                
                # Update live counter display with color-coded indicators
                if live_counter_placeholder:
                    live_counter_placeholder.markdown(f'''
                    <div class="live-counter-container">
                        <div class="live-counter total">
                            <div class="counter-icon">🚗</div>
                            <div class="counter-label">Total Vehicles</div>
                            <div class="counter-value">{current_total}</div>
                            <div class="counter-direction">Detected</div>
                        </div>
                        <div class="live-counter up">
                            <div class="counter-icon">⬆️</div>
                            <div class="counter-label">Moving Up</div>
                            <div class="counter-value" style="color: #2ecc71;">{current_up}</div>
                            <div class="counter-direction">Heading North/Forward</div>
                        </div>
                        <div class="live-counter down">
                            <div class="counter-icon">⬇️</div>
                            <div class="counter-label">Moving Down</div>
                            <div class="counter-value" style="color: #e74c3c;">{current_down}</div>
                            <div class="counter-direction">Heading South/Backward</div>
                        </div>
                    </div>
                    <div class="processing-status">
                        <div class="status-badge">🔄 Processing in Progress</div>
                        <div style="margin-top: 0.5rem; opacity: 0.7;">
                            Frame {frame_count:,} of {total_frames:,} • {progress:.1f}% Complete
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
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

# Main header with enhanced design
st.markdown('''
<div class="main-header">🚗 Vehicle Tracking & Counting System</div>
<div class="subtitle">
    Advanced Computer Vision | Multi-Modal Detection | Real-Time Analysis
</div>
''', unsafe_allow_html=True)

# Feature badges
st.markdown('''
<div style="text-align: center; margin-bottom: 2rem;">
    <span class="feature-badge">🎯 YOLOv8 Detection</span>
    <span class="feature-badge">🔄 Optical Flow</span>
    <span class="feature-badge">📊 Real-Time Analytics</span>
    <span class="feature-badge">🎨 Auto ROI Detection</span>
</div>
''', unsafe_allow_html=True)

# Inject JavaScript for sidebar collapse
if st.session_state.sidebar_collapsed:
    st.markdown('''
    <script>
        document.body.classList.add('sidebar-collapsed');
    </script>
    <style>
        section[data-testid="stSidebar"] {
            width: 0 !important;
            min-width: 0 !important;
            overflow: hidden;
        }
    </style>
    ''', unsafe_allow_html=True)

# Sidebar toggle button
if st.button("☰", key="sidebar_toggle", help="Toggle Sidebar"):
    st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
    st.rerun()

# Sidebar for configuration
    st.sidebar.markdown('''
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h2 style="color: var(--primary-color); font-weight: 800; margin-bottom: 0.5rem;">⚙️ Configuration</h2>
        <p style="opacity: 0.7; font-size: 0.85rem;">Customize your analysis</p>
    </div>
    ''', unsafe_allow_html=True)

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
    st.markdown('<h3 style="color: var(--primary-color); font-weight: 700; margin-bottom: 1.5rem;">🎬 Video Processing</h3>', unsafe_allow_html=True)
    
    if video_path and os.path.exists(video_path):
        # Display video info
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            st.markdown(f'''
            <div class="info-card">
                <h4 style="margin: 0 0 0.75rem 0; color: var(--primary-color); font-weight: 700;">📹 Video Information</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem;">
                    <div><strong>Resolution:</strong> {width} × {height}px</div>
                    <div><strong>Frame Rate:</strong> {fps} FPS</div>
                    <div><strong>Total Frames:</strong> {total_frames:,}</div>
                    <div><strong>Duration:</strong> {duration:.1f}s</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
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
        if st.button("🚀 Process Video", type="primary", use_container_width=True):
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
            # Create live counter display
            st.markdown('<h4 style="color: var(--primary-color); font-weight: 700; margin: 1.5rem 0;">🎬 Live Processing Status</h4>', unsafe_allow_html=True)
            live_counter_placeholder = st.empty()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing video frames..."):
                results = process_video_streamlit(
                    video_path,
                    processor,
                    progress_bar,
                    status_text,
                    live_counter_placeholder
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
                        
                        # Show final counter summary
                        final_counts = results['final_counts']
                        live_counter_placeholder.markdown(f'''
                        <div class="live-counter-container">
                            <div class="live-counter total">
                                <div class="counter-icon">🚗</div>
                                <div class="counter-label">Total Vehicles</div>
                                <div class="counter-value">{final_counts['total']}</div>
                                <div class="counter-direction">✅ Final Count</div>
                            </div>
                            <div class="live-counter up">
                                <div class="counter-icon">⬆️</div>
                                <div class="counter-label">Moved Up</div>
                                <div class="counter-value" style="color: #2ecc71;">{final_counts['up']}</div>
                                <div class="counter-direction">✅ North/Forward</div>
                            </div>
                            <div class="live-counter down">
                                <div class="counter-icon">⬇️</div>
                                <div class="counter-label">Moved Down</div>
                                <div class="counter-value" style="color: #e74c3c;">{final_counts['down']}</div>
                                <div class="counter-direction">✅ South/Backward</div>
                            </div>
                        </div>
                        <div class="processing-status" style="background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(46, 204, 113, 0.05) 100%); border-color: rgba(46, 204, 113, 0.3);">
                            <div class="status-badge" style="background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);">✅ Processing Complete</div>
                            <div style="margin-top: 0.5rem; opacity: 0.7;">
                                Successfully processed {results['total_frames']:,} frames at {results['fps']} FPS
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        st.success("✅ Video processing completed successfully!")
                        
                        # Display output video
                        st.markdown('<h3 style="color: var(--primary-color); font-weight: 700; margin: 2rem 0 1rem 0;">📹 Processed Video</h3>', unsafe_allow_html=True)
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
                                    use_container_width=True
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
        st.markdown('''
        <div class="info-card" style="text-align: center; padding: 3rem 2rem;">
            <h3 style="color: var(--primary-color); font-weight: 700; margin-bottom: 1rem;">🎥 Ready to Get Started?</h3>
            <p style="opacity: 0.8; margin-bottom: 1.5rem; font-size: 1.1rem;">
                Select a demo video or upload your own to begin vehicle tracking and counting
            </p>
            <div style="margin-top: 2rem;">
                <span class="feature-badge" style="margin: 0.5rem;">👈 Choose Video Source</span>
                <span class="feature-badge" style="margin: 0.5rem;">⚙️ Configure Settings</span>
                <span class="feature-badge" style="margin: 0.5rem;">🚀 Process Video</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

with col2:
    st.markdown('<h3 style="color: var(--primary-color); font-weight: 700; margin-bottom: 1.5rem;">📈 Statistics & Analytics</h3>', unsafe_allow_html=True)
    
    # Only show statistics after video processing completes and video is displayed
    # Don't show statistics during processing - wait until video is shown
    if st.session_state.processing_results:
        results = st.session_state.processing_results
        final_counts = results['final_counts']
        
        # Enhanced Metrics with Icons
        st.markdown('''
        <div class="stats-container">
            <h4 style="color: var(--text-color); margin-bottom: 1rem; font-weight: 700;">Vehicle Count Summary</h4>
        </div>
        ''', unsafe_allow_html=True)
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("🚗 Total", final_counts['total'])
        with col_metric2:
            st.metric("⬆️ Up", final_counts['up'])
        with col_metric3:
            st.metric("⬇️ Down", final_counts['down'])
        
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
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<h4 style="color: var(--primary-color); font-weight: 700; margin-bottom: 1rem;">📊 Vehicle Count Over Time</h4>', unsafe_allow_html=True)
            
            df_counts = pd.DataFrame(results['count_history'])
            
            fig = px.line(
                df_counts,
                x='frame',
                y=['total', 'up', 'down'],
                labels={'frame': 'Frame Number', 'value': 'Count', 'variable': 'Direction'},
                color_discrete_map={'total': '#9D8DF1', 'up': '#2ecc71', 'down': '#e74c3c'}
            )
            fig.update_layout(
                height=300,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", size=12),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(0,0,0,0)'
                ),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            fig.update_traces(line=dict(width=3))
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(157, 141, 241, 0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(157, 141, 241, 0.1)')
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Track count chart
            if results['track_history']:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<h4 style="color: var(--primary-color); font-weight: 700; margin-bottom: 1rem;">🎯 Active Tracks Over Time</h4>', unsafe_allow_html=True)
                
                df_tracks = pd.DataFrame({
                    'frame': range(len(results['track_history'])),
                    'tracks': results['track_history']
                })
                
                fig_tracks = px.area(
                    df_tracks,
                    x='frame',
                    y='tracks',
                    labels={'frame': 'Frame Number', 'tracks': 'Number of Tracks'}
                )
                fig_tracks.update_layout(
                    height=250,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif", size=12),
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                fig_tracks.update_traces(
                    fillcolor='rgba(157, 141, 241, 0.3)',
                    line=dict(color='#9D8DF1', width=3)
                )
                fig_tracks.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(157, 141, 241, 0.1)')
                fig_tracks.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(157, 141, 241, 0.1)')
                
                st.plotly_chart(fig_tracks, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
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
        st.markdown('''
        <div class="stats-container" style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h4 style="color: var(--text-color); opacity: 0.8; font-weight: 600;">Statistics Panel</h4>
            <p style="opacity: 0.6; margin-top: 0.5rem;">
                Process a video to view detailed analytics and insights
            </p>
        </div>
        ''', unsafe_allow_html=True)

# Footer with enhanced design
st.markdown("---")
st.markdown('''
<div style="text-align: center; padding: 2rem 0;">
    <div style="margin-bottom: 1rem;">
        <span class="feature-badge" style="background: linear-gradient(135deg, rgba(157, 141, 241, 0.2) 0%, rgba(157, 141, 241, 0.1) 100%); color: var(--text-color); box-shadow: none;">
            Multi-Modal Vehicle Tracking & Counting System
        </span>
    </div>
    <div style="color: var(--text-color); opacity: 0.6; font-size: 0.9rem;">
        Introduction to Computer Vision 
    </div>
    <div style="margin-top: 1rem; opacity: 0.5; font-size: 0.8rem;">
        Powered by YOLOv8 • OpenCV • Streamlit
    </div>
</div>
''', unsafe_allow_html=True)

