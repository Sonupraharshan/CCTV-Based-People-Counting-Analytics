"""
Streamlit Web Dashboard for CCTV People Counting
Interactive interface for configuring and monitoring the counting system.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import tempfile
from datetime import datetime, timedelta

from src.pipeline import PeopleCountingPipeline
from src.utils.database import Database


# Page configuration
st.set_page_config(
    page_title="CCTV People Counting System",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .count-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .count-number {
        font-size: 3rem;
        font-weight: bold;
    }
    .count-label {
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'counting_line' not in st.session_state:
        st.session_state.counting_line = None
    if 'db' not in st.session_state:
        st.session_state.db = Database()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = None


def draw_line_tool(frame: np.ndarray, existing_line=None):
    """Interactive line drawing tool using sliders for precise control."""
    st.subheader("📏 Define Counting Line")
    
    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        # Initialize preset values in session state if needed
        if 'preset_values' not in st.session_state:
            st.session_state.preset_values = None
        
        # Get defaults
        if st.session_state.preset_values:
            # Use preset values
            default_x1, default_y1, default_x2, default_y2 = st.session_state.preset_values
            st.session_state.preset_values = None  # Clear after using
        elif existing_line:
            default_x1, default_y1 = int(existing_line[0][0]), int(existing_line[0][1])
            default_x2, default_y2 = int(existing_line[1][0]), int(existing_line[1][1])
        else:
            # Default: horizontal line across middle
            default_x1, default_y1 = int(w * 0.1), int(h * 0.5)
            default_x2, default_y2 = int(w * 0.9), int(h * 0.5)
        
        # Quick preset buttons BEFORE sliders
        st.write("**Quick Presets:**")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("⬌ Horizontal"):
                st.session_state.preset_values = (int(w * 0.1), int(h * 0.5), int(w * 0.9), int(h * 0.5))
                st.rerun()
        
        with preset_col2:
            if st.button("⬍ Vertical"):
                st.session_state.preset_values = (int(w * 0.5), int(h * 0.1), int(w * 0.5), int(h * 0.9))
                st.rerun()
        
        with preset_col3:
            if st.button("⬉ Diagonal"):
                st.session_state.preset_values = (int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))
                st.rerun()
        
        st.write("**Adjust the line using sliders** - see live preview below")
        
        # Create two columns for start and end points
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 Start Point**")
            x1 = st.slider("X1", 0, w, default_x1, key="slider_x1")
            y1 = st.slider("Y1", 0, h, default_y1, key="slider_y1")
            st.caption(f"Position: ({x1}, {y1})")
        
        with col2:
            st.markdown("**🔴 End Point**")
            x2 = st.slider("X2", 0, w, default_x2, key="slider_x2")
            y2 = st.slider("Y2", 0, h, default_y2, key="slider_y2")
            st.caption(f"Position: ({x2}, {y2})")
        
        # Draw line on frame
        display_frame = frame_rgb.copy()
        
        # Draw the counting line
        cv2.line(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
        
        # Draw start point (green circle)
        cv2.circle(display_frame, (x1, y1), 12, (0, 255, 0), -1)
        cv2.circle(display_frame, (x1, y1), 15, (255, 255, 255), 2)
        cv2.putText(display_frame, "START", (x1 + 20, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw end point (red circle)
        cv2.circle(display_frame, (x2, y2), 12, (0, 0, 255), -1)
        cv2.circle(display_frame, (x2, y2), 15, (255, 255, 255), 2)
        cv2.putText(display_frame, "END", (x2 + 20, y2 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw direction arrow
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.arrowedLine(display_frame, (x1, y1), (mid_x, mid_y), (255, 255, 0), 3, tipLength=0.3)
        
        # Display the frame with line
        st.image(display_frame, caption="Live Preview - Counting Line")
        
        # Calculate line length
        import math
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Info
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"📐 Line Length: {line_length:.0f} pixels")
        with col_info2:
            st.info(f"📏 Frame Size: {w}×{h} pixels")
        
        return ((x1, y1), (x2, y2))
    
    return existing_line


def main():
    """Main application."""
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">🚶 CCTV People Counting System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        model_path = st.text_input("Model Path", value="yolov8n.pt")
        tracker_type = st.selectbox("Tracker Type", ["strongsort", "bytetrack"])
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        
        # Counting settings
        st.subheader("Counting Settings")
        min_track_length = st.slider("Min Track Length", 5, 30, 10)
        debounce_frames = st.slider("Debounce Frames", 10, 60, 30)
        
        # Display settings
        st.subheader("Display Settings")
        show_trajectories = st.checkbox("Show Trajectories", value=True)
        trajectory_length = st.slider("Trajectory Length", 10, 50, 30)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📹 Live Monitoring", "📊 Analytics", "📥 Export", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Video Source")
            
            # Source selection
            source_type = st.radio("Select Source", ["Upload Video", "RTSP Stream", "Webcam"], horizontal=True)
            
            video_path = None
            
            if source_type == "Upload Video":
                uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
                if uploaded_file:
                    # Save to temp file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                else:
                    # Video was removed - clear the frame
                    if st.session_state.current_frame is not None:
                        st.session_state.current_frame = None
                        st.session_state.counting_line = None
            
            elif source_type == "RTSP Stream":
                rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://username:password@ip:port/stream")
                if rtsp_url:
                    video_path = rtsp_url
                else:
                    # URL was cleared
                    if st.session_state.current_frame is not None:
                        st.session_state.current_frame = None
                        st.session_state.counting_line = None
            
            else:  # Webcam
                video_path = 0
            
            # Load first frame for line drawing
            if video_path is not None and st.session_state.current_frame is None:
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                if ret:
                    st.session_state.current_frame = frame
                cap.release()
            
            # Line drawing
            if st.session_state.current_frame is not None:
                new_line = draw_line_tool(
                    st.session_state.current_frame,
                    existing_line=st.session_state.counting_line
                )
                
                if new_line:
                    st.session_state.counting_line = new_line
                    st.success(f"✅ Counting line set: Start{new_line[0]}, End{new_line[1]}")
            
            # Control buttons
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("▶️ Start Processing", disabled=st.session_state.processing):
                    if video_path and st.session_state.counting_line:
                        st.session_state.processing = True
                        process_video(video_path, model_path, tracker_type, conf_threshold, min_track_length)
                    else:
                        st.error("Please upload video and set counting line!")
            
            with col_b:
                if st.button("⏹️ Stop", disabled=not st.session_state.processing):
                    st.session_state.processing = False
                    st.info("Processing stopped")
            
            with col_c:
                if st.button("🔄 Reset"):
                    # Clear all session state related to video and line
                    st.session_state.counting_line = None
                    st.session_state.current_frame = None
                    st.session_state.processing = False
                    if 'preset_values' in st.session_state:
                        st.session_state.preset_values = None
                    
                    # Clear database counts
                    if st.session_state.db:
                        st.session_state.db.clear_events()
                    
                    # Reset pipeline if exists
                    if st.session_state.pipeline:
                        st.session_state.pipeline.reset()
                        st.session_state.pipeline = None
                    
                    st.success("✅ System reset - all counts cleared!")
                    st.rerun()
        
        with col2:
            st.subheader("Real-Time Counts")
            
            # Get current stats from database
            if st.session_state.db:
                stats = st.session_state.db.get_statistics()
                
                # IN count
                st.markdown(f"""
                    <div class="count-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                        <div class="count-number">{stats['total_in']}</div>
                        <div class="count-label">↑ IN</div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # OUT count
                st.markdown(f"""
                    <div class="count-card" style="background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);">
                        <div class="count-number">{stats['total_out']}</div>
                        <div class="count-label">↓ OUT</div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # NET count
                net_count = stats['total_in'] - stats['total_out']
                st.markdown(f"""
                    <div class="count-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                        <div class="count-number">{net_count}</div>
                        <div class="count-label">NET Count</div>
                    </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📊 Historical Analytics")
        
        # Time range selector
        col1, col2 = st.columns(2)
        with col1:
            time_range = st.selectbox("Time Range", ["Last Hour", "Last 24 Hours", "Last 7 Days", "All Time"])
        with col2:
            interval = st.selectbox("Aggregation", ["Minute", "Hour", "Day"])
        
        # Calculate time range
        if time_range == "Last Hour":
            start_time = time.time() - 3600
        elif time_range == "Last 24 Hours":
            start_time = time.time() - 86400
        elif time_range == "Last 7 Days":
            start_time = time.time() - 604800
        else:
            start_time = None
        
        # Get data
        if st.session_state.db:
            df_counts = st.session_state.db.get_count_by_time(
                interval=interval.lower(),
                start_time=start_time
            )
            
            if not df_counts.empty:
                # Time series chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df_counts['time_period'],
                    y=df_counts['count_in'],
                    mode='lines+markers',
                    name='IN',
                    line=dict(color='#38ef7d', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df_counts['time_period'],
                    y=df_counts['count_out'],
                    mode='lines+markers',
                    name='OUT',
                    line=dict(color='#f45c43', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title=f"People Count Over Time ({interval})",
                    xaxis_title="Time",
                    yaxis_title="Count",
                    hovermode='x unified',
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Direction distribution pie chart
                col1, col2 = st.columns(2)
                
                with col1:
                    total_in = df_counts['count_in'].sum()
                    total_out = df_counts['count_out'].sum()
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['IN', 'OUT'],
                        values=[total_in, total_out],
                        marker=dict(colors=['#38ef7d', '#f45c43'])
                    )])
                    
                    fig_pie.update_layout(
                        title="Direction Distribution",
                        height=300
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Peak hours
                    st.subheader("📈 Peak Periods")
                    peak_period = df_counts.nlargest(5, 'total_count')[['time_period', 'total_count']]
                    st.dataframe(peak_period, hide_index=True)
            else:
                st.info("No data available for the selected time range.")
    
    with tab3:
        st.subheader("📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Export Options**")
            export_format = st.radio("Format", ["CSV", "JSON"], horizontal=True)
            include_all = st.checkbox("Include all events", value=True)
            
            if not include_all:
                export_start = st.date_input("Start Date")
                export_end = st.date_input("End Date")
            
            if st.button("📥 Export Events"):
                if st.session_state.db:
                    # Export to temp file
                    output_path = f"events_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}"
                    
                    if export_format == "CSV":
                        st.session_state.db.export_csv(output_path)
                    
                    st.success(f"✅ Exported to {output_path}")
                    
                    # Download button
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download File",
                            data=f,
                            file_name=output_path,
                            mime='text/csv' if export_format == 'CSV' else 'application/json'
                        )
        
        with col2:
            st.write("**Recent Events**")
            if st.session_state.db:
                recent_events = st.session_state.db.get_events(limit=10)
                if not recent_events.empty:
                    st.dataframe(
                        recent_events[['datetime', 'track_id', 'direction', 'frame_number']],
                        hide_index=True
                    )
    
    with tab4:
        st.subheader("ℹ️ About CCTV People Counting System")
        
        st.markdown("""
        ### Features
        - **Real-time Detection**: YOLOv8-based people detection
        - **Robust Tracking**: StrongSORT and ByteTrack support
        - **Accurate Counting**: Line-crossing with anti-double-count
        - **Multiple Sources**: Video files, RTSP streams, webcams
        - **Analytics Dashboard**: Historical trends and statistics
        - **Data Export**: CSV and JSON export options
        
        ### System Requirements
        - Python 3.9+
        - CUDA-capable GPU (recommended)
        - 8GB RAM minimum
        
        ### Model Information
        - **Detector**: YOLOv8 (Ultralytics)
        - **Tracker**: StrongSORT / ByteTrack
        - **Frameworks**: PyTorch, OpenCV, Streamlit
        
        ### Support
        - GitHub: [CCTV-Based-People-Counting-Analytics](https://github.com/Sonupraharshan/CCTV-Based-People-Counting-Analytics)
        - Documentation: See README.md
        """)


def process_video(source, model_path, tracker_type, conf_threshold, min_track_length):
    """Process video and display results in Streamlit."""
    import sys
    from src.utils.video_reader import VideoReader
    from src.utils.overlay import draw_tracks, draw_counting_line, draw_count_overlay, draw_fps
    
    # Initialize pipeline if needed
    if st.session_state.pipeline is None:
        st.session_state.pipeline = PeopleCountingPipeline(
            detector_model=model_path,
            tracker_type=tracker_type,
            conf_threshold=conf_threshold
        )
    
    # Set counting line
    st.session_state.pipeline.set_counting_line(
        st.session_state.counting_line,
        min_track_length=min_track_length
    )
    
    # Create placeholders for dynamic updates
    st.write("### Processing Video...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    video_placeholder = st.empty()
    
    # Create metric placeholders that update in place
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        metric_in = st.empty()
    with stats_col2:
        metric_out = st.empty()
    with stats_col3:
        metric_net = st.empty()
    
    # Initialize video reader
    video_reader = VideoReader(source)
    video_reader.start()
    
    total_frames = int(video_reader.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if hasattr(video_reader, 'cap') else 1000
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while video_reader.is_opened():
            # Read frame
            ret, frame, frame_num = video_reader.read(timeout=2.0)
            
            if not ret:
                break
            
            # Process frame
            boxes, scores, class_ids = st.session_state.pipeline.detector.detect(frame)
            
            # Update tracker
            tracks = st.session_state.pipeline.tracker.update(
                detections=boxes if len(boxes) > 0 else [],
                scores=scores if len(scores) > 0 else [],
                frame=frame
            )
            
            # Update counter
            events = st.session_state.pipeline.counter.update(tracks)
            
            # Log events to database
            for event in events:
                st.session_state.db.insert_event(event)
            
            # Create visualization
            vis_frame = frame.copy()
            vis_frame = draw_tracks(vis_frame, tracks, draw_trajectory=True, trajectory_length=30)
            vis_frame = draw_counting_line(vis_frame, st.session_state.pipeline.counter.line)
            vis_frame = draw_count_overlay(
                vis_frame,
                st.session_state.pipeline.counter.count_in,
                st.session_state.pipeline.counter.count_out,
                position='top-right'
            )
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            vis_frame = draw_fps(vis_frame, current_fps)
            
            # Update UI every 10 frames to avoid slowdown
            if frame_count % 10 == 0:
                # Update progress
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                
                # Update status
                status_text.text(f"Frame {frame_count}/{total_frames} | FPS: {current_fps:.1f}")
                
                # Display frame
                vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(vis_frame_rgb, channels="RGB")
                
                # Update metrics in place
                metric_in.metric("🟢 IN Count", st.session_state.pipeline.counter.count_in)
                metric_out.metric("🔴 OUT Count", st.session_state.pipeline.counter.count_out)
                metric_net.metric("🔵 NET Count", st.session_state.pipeline.counter.get_net_count())
        
        # Final update
        progress_bar.progress(1.0)
        status_text.text(f"✅ Processing complete! Processed {frame_count} frames")
        
        # Final statistics
        st.success(f"""
        **Processing Complete!**
        - Total Frames: {frame_count}
        - IN: {st.session_state.pipeline.counter.count_in}
        - OUT: {st.session_state.pipeline.counter.count_out}
        - NET: {st.session_state.pipeline.counter.get_net_count()}
        - Average FPS: {current_fps:.2f}
        """)
        
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    finally:
        video_reader.stop()
        st.session_state.processing = False


if __name__ == "__main__":
    main()
