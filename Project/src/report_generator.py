"""
Professional report generator for vehicle tracking and counting results.
Generates HTML reports with charts and statistics.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import base64
from pathlib import Path


class ReportGenerator:
    """Generates professional HTML reports with vehicle statistics."""
    
    def __init__(self):
        """Initialize report generator."""
        pass
    
    def generate_html_report(self, results, video_info=None):
        """
        Generate a professional HTML report.
        
        Args:
            results: Dictionary with processing results including:
                - final_counts: {'total', 'up', 'down'}
                - vehicle_stats: Statistics from get_vehicle_statistics()
                - count_history: List of count data over time
                - track_history: List of track counts over time
            video_info: Dictionary with video information (optional)
                - width, height, fps, total_frames, duration
        
        Returns:
            str: HTML report content
        """
        final_counts = results.get('final_counts', {})
        vehicle_stats = results.get('vehicle_stats', {})
        count_history = results.get('count_history', {})
        track_history = results.get('track_history', {})
        
        # Extract statistics
        type_counts = vehicle_stats.get('type_counts', {})
        color_counts = vehicle_stats.get('color_counts', {})
        direction_type_counts = vehicle_stats.get('direction_type_counts', {})
        direction_color_counts = vehicle_stats.get('direction_color_counts', {})
        
        # Generate charts
        type_chart_html = self._generate_type_chart(type_counts)
        color_chart_html = self._generate_color_chart(color_counts)
        direction_chart_html = self._generate_direction_chart(final_counts)
        timeline_chart_html = self._generate_timeline_chart(count_history)
        track_timeline_chart_html = self._generate_track_timeline_chart(track_history)
        
        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vehicle Tracking Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 8px 8px 0 0;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            margin-bottom: 40px;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
        }}
        .section h2 {{
            color: #1f77b4;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #666;
            font-size: 1em;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .table-container {{
            overflow-x: auto;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #e0e0e0;
            margin-top: 40px;
        }}
        .info-box {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗 Vehicle Tracking & Counting Report</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <!-- Executive Summary -->
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{final_counts.get('total', 0)}</div>
                    <div class="metric-label">Total Vehicles</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{final_counts.get('up', 0)}</div>
                    <div class="metric-label">Direction Up</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{final_counts.get('down', 0)}</div>
                    <div class="metric-label">Direction Down</div>
                </div>
            </div>
            {self._generate_video_info(video_info)}
        </div>
        
        <!-- Vehicle Type Breakdown -->
        <div class="section">
            <h2>Vehicle Type Distribution</h2>
            <div class="chart-container">
                {type_chart_html}
            </div>
            {self._generate_type_table(type_counts, direction_type_counts)}
        </div>
        
        <!-- Vehicle Color Distribution -->
        <div class="section">
            <h2>Vehicle Color Distribution</h2>
            <div class="chart-container">
                {color_chart_html}
            </div>
            {self._generate_color_table(color_counts, direction_color_counts)}
        </div>
        
        <!-- Direction Analysis -->
        <div class="section">
            <h2>Direction Analysis</h2>
            <div class="chart-container">
                {direction_chart_html}
            </div>
        </div>
        
        <!-- Timeline Analysis -->
        <div class="section">
            <h2>Vehicle Count Over Time</h2>
            <div class="chart-container">
                {timeline_chart_html}
            </div>
        </div>
        
        <!-- Track Analysis -->
        <div class="section">
            <h2>Active Tracks Over Time</h2>
            <div class="chart-container">
                {track_timeline_chart_html}
            </div>
        </div>
        
        <div class="footer">
            <p>Multi-Modal Vehicle Tracking and Counting System | CPS843 - Introduction to Computer Vision | Fall 2025</p>
            <p>Generated using Lucas-Kanade Optical Flow, YOLOv8, and Kalman Filtering</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _generate_video_info(self, video_info):
        """Generate video information section."""
        if not video_info:
            return ""
        
        width = video_info.get('width', 'N/A')
        height = video_info.get('height', 'N/A')
        fps = video_info.get('fps', 'N/A')
        total_frames = video_info.get('total_frames', 'N/A')
        duration = total_frames / fps if isinstance(fps, (int, float)) and fps > 0 else 'N/A'
        
        return f"""
        <div class="info-box">
            <strong>Video Information:</strong><br>
            Resolution: {width}x{height} | FPS: {fps} | Frames: {total_frames} | Duration: {duration:.1f}s
        </div>
        """
    
    def _generate_type_chart(self, type_counts):
        """Generate vehicle type pie chart."""
        if not type_counts:
            return "<p>No vehicle type data available.</p>"
        
        labels = list(type_counts.keys())
        values = list(type_counts.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=px.colors.qualitative.Set3[:len(labels)]
        )])
        fig.update_layout(
            title="Vehicle Type Distribution",
            height=400,
            showlegend=True
        )
        
        return fig.to_html(include_plotlyjs=False, div_id="type-chart")
    
    def _generate_color_chart(self, color_counts):
        """Generate vehicle color pie chart."""
        if not color_counts:
            return "<p>No vehicle color data available.</p>"
        
        labels = list(color_counts.keys())
        values = list(color_counts.values())
        
        # Map colors to actual color values
        color_map = {
            'red': '#FF0000', 'blue': '#0000FF', 'green': '#00FF00',
            'yellow': '#FFFF00', 'orange': '#FFA500', 'purple': '#800080',
            'pink': '#FFC0CB', 'white': '#FFFFFF', 'gray': '#808080',
            'silver': '#C0C0C0', 'black': '#000000', 'brown': '#A52A2A'
        }
        
        colors = [color_map.get(label.lower(), '#808080') for label in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=colors
        )])
        fig.update_layout(
            title="Vehicle Color Distribution",
            height=400,
            showlegend=True
        )
        
        return fig.to_html(include_plotlyjs=False, div_id="color-chart")
    
    def _generate_direction_chart(self, final_counts):
        """Generate direction comparison chart."""
        directions = ['Up', 'Down']
        counts = [final_counts.get('up', 0), final_counts.get('down', 0)]
        
        fig = go.Figure(data=[go.Bar(
            x=directions,
            y=counts,
            marker_color=['#2ca02c', '#d62728']
        )])
        fig.update_layout(
            title="Vehicle Count by Direction",
            xaxis_title="Direction",
            yaxis_title="Count",
            height=300
        )
        
        return fig.to_html(include_plotlyjs=False, div_id="direction-chart")
    
    def _generate_timeline_chart(self, count_history):
        """Generate timeline chart for counts over time."""
        if not count_history:
            return "<p>No timeline data available.</p>"
        
        df = pd.DataFrame(count_history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df['total'],
            name='Total',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df['up'],
            name='Up',
            line=dict(color='#2ca02c', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df['down'],
            name='Down',
            line=dict(color='#d62728', width=2)
        ))
        fig.update_layout(
            title="Vehicle Count Over Time",
            xaxis_title="Frame Number",
            yaxis_title="Count",
            height=400,
            hovermode='x unified'
        )
        
        return fig.to_html(include_plotlyjs=False, div_id="timeline-chart")
    
    def _generate_track_timeline_chart(self, track_history):
        """Generate timeline chart for active tracks."""
        if not track_history:
            return "<p>No track data available.</p>"
        
        frames = list(range(len(track_history)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frames,
            y=track_history,
            name='Active Tracks',
            line=dict(color='#9467bd', width=2),
            fill='tozeroy',
            fillcolor='rgba(148, 103, 189, 0.2)'
        ))
        fig.update_layout(
            title="Active Tracks Over Time",
            xaxis_title="Frame Number",
            yaxis_title="Number of Tracks",
            height=300,
            hovermode='x unified'
        )
        
        return fig.to_html(include_plotlyjs=False, div_id="track-timeline-chart")
    
    def _generate_type_table(self, type_counts, direction_type_counts):
        """Generate vehicle type breakdown table."""
        if not type_counts:
            return ""
        
        table_html = """
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Vehicle Type</th>
                        <th>Total Count</th>
                        <th>Direction Up</th>
                        <th>Direction Down</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for vehicle_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            up_count = direction_type_counts.get('up', {}).get(vehicle_type, 0)
            down_count = direction_type_counts.get('down', {}).get(vehicle_type, 0)
            table_html += f"""
                    <tr>
                        <td><strong>{vehicle_type.capitalize()}</strong></td>
                        <td>{count}</td>
                        <td>{up_count}</td>
                        <td>{down_count}</td>
                    </tr>
            """
        
        table_html += """
                </tbody>
            </table>
        </div>
        """
        
        return table_html
    
    def _generate_color_table(self, color_counts, direction_color_counts):
        """Generate vehicle color breakdown table."""
        if not color_counts:
            return ""
        
        table_html = """
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Vehicle Color</th>
                        <th>Total Count</th>
                        <th>Direction Up</th>
                        <th>Direction Down</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True):
            up_count = direction_color_counts.get('up', {}).get(color, 0)
            down_count = direction_color_counts.get('down', {}).get(color, 0)
            table_html += f"""
                    <tr>
                        <td><strong>{color.capitalize()}</strong></td>
                        <td>{count}</td>
                        <td>{up_count}</td>
                        <td>{down_count}</td>
                    </tr>
            """
        
        table_html += """
                </tbody>
            </table>
        </div>
        """
        
        return table_html

