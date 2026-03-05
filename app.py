"""
Road Safety Object Detection Dashboard
======================================
A Streamlit-based web application for detecting road objects
and hazards in images and videos.

Author: Isaac O Adeboyejo
Date: 2026
Course: Computer Vision Module
"""

import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from datetime import datetime

# Import custom modules
from detector import RoadObjectDetector, HazardLevel
from utils import (
    get_video_info,
    get_image_info,
    calculate_detection_statistics,
    calculate_video_statistics,
    generate_detection_report,
    create_output_directory
)


# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Road Safety Detection Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .hazard-high {
        color: #ff4444;
        font-weight: bold;
    }
    .hazard-medium {
        color: #ffaa00;
        font-weight: bold;
    }
    .hazard-low {
        color: #00cc00;
        font-weight: bold;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== CACHED RESOURCES ====================
@st.cache_resource
def load_detector():
    """Load the object detection model (cached for performance)."""
    return RoadObjectDetector(model_size='n', device='cpu')


@st.cache_data
def get_sample_data():
    """Get sample detection data for demonstration."""
    return {
        'objects': ['car', 'person', 'bicycle', 'motorcycle', 'truck', 'bus', 'dog'],
        'counts': [45, 23, 12, 8, 15, 5, 3],
        'hazard_levels': ['LOW', 'HIGH', 'MEDIUM', 'MEDIUM', 'LOW', 'LOW', 'HIGH']
    }


# ==================== VISUALIZATION FUNCTIONS ====================
def create_hazard_gauge(high: int, medium: int, low: int) -> go.Figure:
    """
    Create a gauge chart showing overall hazard score.
    
    Args:
        high: Count of high hazard detections
        medium: Count of medium hazard detections
        low: Count of low hazard detections
        
    Returns:
        Plotly figure object
    """
    total = high + medium + low
    if total == 0:
        score = 0
    else:
        # Weighted score: HIGH=3, MEDIUM=2, LOW=1
        score = ((high * 3) + (medium * 2) + (low * 1)) / total
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Hazard Score", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 3], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1], 'color': '#00cc00'},
                {'range': [1, 2], 'color': '#ffaa00'},
                {'range': [2, 3], 'color': '#ff4444'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_object_distribution_chart(detections: list) -> go.Figure:
    """
    Create a bar chart showing distribution of detected objects.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Plotly figure object
    """
    if not detections:
        fig = go.Figure()
        fig.add_annotation(
            text="No detections to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    df = pd.DataFrame(detections)
    object_counts = df['object'].value_counts().reset_index()
    object_counts.columns = ['Object', 'Count']
    
    # Add hazard level colors
    hazard_colors = {
        'HIGH': '#ff4444',
        'MEDIUM': '#ffaa00',
        'LOW': '#00cc00'
    }
    
    colors = []
    for obj in object_counts['Object']:
        obj_hazard = df[df['object'] == obj]['hazard_level'].iloc[0]
        colors.append(hazard_colors.get(obj_hazard, '#1E88E5'))
    
    fig = px.bar(
        object_counts,
        x='Object',
        y='Count',
        title='Detected Objects Distribution',
        color='Object',
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Object Type",
        yaxis_title="Count",
        height=350
    )
    
    return fig


def create_confidence_histogram(detections: list) -> go.Figure:
    """
    Create a histogram of detection confidence scores.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Plotly figure object
    """
    if not detections:
        fig = go.Figure()
        fig.add_annotation(
            text="No detections to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    confidences = [d['confidence'] for d in detections]
    
    fig = px.histogram(
        x=confidences,
        nbins=20,
        title='Detection Confidence Distribution',
        labels={'x': 'Confidence Score', 'y': 'Count'},
        color_discrete_sequence=['#1E88E5']
    )
    
    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Frequency",
        height=300
    )
    
    return fig


def create_hazard_pie_chart(hazard_summary: dict) -> go.Figure:
    """
    Create a pie chart showing hazard level distribution.
    
    Args:
        hazard_summary: Dictionary with hazard counts
        
    Returns:
        Plotly figure object
    """
    labels = ['High Risk', 'Medium Risk', 'Low Risk']
    values = [
        hazard_summary.get('HIGH', 0),
        hazard_summary.get('MEDIUM', 0),
        hazard_summary.get('LOW', 0)
    ]
    colors = ['#ff4444', '#ffaa00', '#00cc00']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title="Hazard Level Distribution",
        height=300
    )
    
    return fig


def create_timeline_chart(frame_data: list) -> go.Figure:
    """
    Create a timeline chart showing detections over video frames.
    
    Args:
        frame_data: List of dictionaries with frame-by-frame data
        
    Returns:
        Plotly figure object
    """
    if not frame_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No frame data to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    frames = [f['frame'] for f in frame_data]
    totals = [f['total_objects'] for f in frame_data]
    high = [f.get('high_hazards', 0) for f in frame_data]
    medium = [f.get('medium_hazards', 0) for f in frame_data]
    low = [f.get('low_hazards', 0) for f in frame_data]
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Total Detections', 'Hazard Breakdown'))
    
    # Total detections
    fig.add_trace(
        go.Scatter(x=frames, y=totals, mode='lines', name='Total',
                   line=dict(color='#1E88E5', width=2)),
        row=1, col=1
    )
    
    # Hazard breakdown
    fig.add_trace(
        go.Scatter(x=frames, y=high, mode='lines', name='High',
                   line=dict(color='#ff4444', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=frames, y=medium, mode='lines', name='Medium',
                   line=dict(color='#ffaa00', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=frames, y=low, mode='lines', name='Low',
                   line=dict(color='#00cc00', width=2)),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="Detection Timeline Analysis"
    )
    
    fig.update_xaxes(title_text="Frame Number", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    return fig


# ==================== MAIN APPLICATION ====================
def main():
    """Main application function."""
    
    # ==================== HEADER ====================
    st.markdown('<p class="main-header">🚗 Road Safety Object Detection Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload images or videos to detect road objects and potential hazards</p>',
        unsafe_allow_html=True
    )
    
    # ==================== LOAD DETECTOR ====================
    with st.spinner("Loading detection model..."):
        detector = load_detector()
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        st.markdown("---")
        
        st.header("📊 Detectable Objects")
        st.markdown("**High Risk** 🔴")
        st.caption("person, dog, cat, horse, cow, sheep")
        
        st.markdown("**Medium Risk** 🟡")
        st.caption("bicycle, motorcycle, bench, fire hydrant")
        
        st.markdown("**Low Risk** 🟢")
        st.caption("car, bus, truck, train")
        
        st.markdown("---")
        
        st.header("ℹ️ Model Information")
        model_info = detector.get_model_info()
        st.caption(f"Model: {model_info['model_type']}")
        st.caption(f"Size: {model_info['model_size'].upper()}")
        st.caption(f"Device: {model_info['device'].upper()}")
        
        st.markdown("---")
        
        st.header("📖 Instructions")
        st.markdown("""
        1. Select a tab (Image or Video)
        2. Upload your file
        3. Click process/analyze
        4. View results and statistics
        """)
    
    # ==================== MAIN TABS ====================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📷 Image Detection",
        "🎥 Video Detection",
        "📈 Statistics",
        "ℹ️ About"
    ])
    
    # ==================== IMAGE DETECTION TAB ====================
    with tab1:
        st.header("Single Image Detection")
        st.markdown("Upload an image to detect road objects and hazards.")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="image_uploader",
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_image is not None:
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📥 Original Image")
                st.image(uploaded_image, use_container_width=True)
            
            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(uploaded_image.read())
                tmp_path = tmp.name
            
            # Get image info
            try:
                image_info = get_image_info(tmp_path)
                st.caption(f"Resolution: {image_info['resolution']} | Size: {image_info['file_size_kb']} KB")
            except Exception as e:
                st.warning(f"Could not read image info: {e}")
            
            # Process button
            if st.button("🔍 Analyze Image", key="analyze_image", type="primary"):
                with st.spinner("Detecting objects..."):
                    results = detector.detect_image(tmp_path, confidence_threshold)
                
                with col2:
                    st.subheader("🎯 Detection Results")
                    # Convert BGR to RGB for display
                    annotated_rgb = cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True)
                
                # Results Section
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Metrics Row
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(
                        "Total Objects",
                        results['total_objects'],
                        help="Total number of road objects detected"
                    )
                
                with metric_col2:
                    st.metric(
                        "🔴 High Hazard",
                        results['hazard_summary']['HIGH'],
                        help="Pedestrians, animals - unpredictable movement"
                    )
                
                with metric_col3:
                    st.metric(
                        "🟡 Medium Hazard",
                        results['hazard_summary']['MEDIUM'],
                        help="Bicycles, motorcycles - moderate risk"
                    )
                
                with metric_col4:
                    st.metric(
                        "🟢 Low Hazard",
                        results['hazard_summary']['LOW'],
                        help="Cars, buses, trucks - predictable behavior"
                    )
                
                # Charts Row
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.plotly_chart(
                        create_hazard_gauge(
                            results['hazard_summary']['HIGH'],
                            results['hazard_summary']['MEDIUM'],
                            results['hazard_summary']['LOW']
                        ),
                        use_container_width=True
                    )
                
                with chart_col2:
                    st.plotly_chart(
                        create_hazard_pie_chart(results['hazard_summary']),
                        use_container_width=True
                    )
                
                # Detailed Results
                if results['detections']:
                    st.subheader("🔍 Detailed Detections")
                    
                    # Object distribution chart
                    st.plotly_chart(
                        create_object_distribution_chart(results['detections']),
                        use_container_width=True
                    )
                    
                    # Detection table
                    df = pd.DataFrame(results['detections'])
                    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
                    df = df[['object', 'confidence', 'hazard_level']]
                    df.columns = ['Object', 'Confidence', 'Hazard Level']
                    
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Alerts
                    high_count = results['hazard_summary']['HIGH']
                    if high_count > 0:
                        st.error(f"⚠️ **ALERT**: {high_count} high-hazard object(s) detected! Exercise caution.")
                    elif results['hazard_summary']['MEDIUM'] > 0:
                        st.warning(f"⚡ **NOTICE**: Medium-hazard objects present. Maintain awareness.")
                    else:
                        st.success("✅ **CLEAR**: No significant hazards detected in this image.")
                else:
                    st.info("No road objects detected in this image. Try adjusting the confidence threshold.")
            
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    # ==================== VIDEO DETECTION TAB ====================
    with tab2:
        st.header("Video Detection")
        st.markdown("Upload a video to analyze road objects frame by frame.")
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="video_uploader",
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_video is not None:
            # Save video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_video.read())
                video_path = tmp.name
            
            # Display video preview
            st.subheader("📹 Video Preview")
            st.video(uploaded_video)
            
            # Get video info
            try:
                video_info = get_video_info(video_path)
                info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                with info_col1:
                    st.metric("Duration", video_info['duration_formatted'])
                with info_col2:
                    st.metric("Frames", video_info['frame_count'])
                with info_col3:
                    st.metric("FPS", video_info['fps'])
                with info_col4:
                    st.metric("Resolution", video_info['resolution'])
            except Exception as e:
                st.warning(f"Could not read video info: {e}")
            
            # Processing options
            st.subheader("⚙️ Processing Options")
            proc_col1, proc_col2 = st.columns(2)
            
            with proc_col1:
                skip_frames = st.slider(
                    "Skip Frames",
                    min_value=0,
                    max_value=10,
                    value=2,
                    help="Skip N frames between detections (0 = process every frame)"
                )
            
            with proc_col2:
                max_frames = st.number_input(
                    "Max Frames to Process",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    help="Limit processing to first N frames"
                )
            
            # Process button
            if st.button("🚀 Process Video", key="process_video", type="primary"):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_display = st.empty()
                
                # Store results
                frame_results = []
                all_detections = []
                
                # Get total frames
                total_frames = min(video_info.get('frame_count', 100), max_frames)
                
                # Process video
                processed_count = 0
                for result in detector.detect_video(video_path, confidence_threshold, skip_frames):
                    if processed_count >= max_frames:
                        break
                    
                    processed_count += 1
                    frame_num = result.frame_number
                    
                    # Update progress
                    progress = min(processed_count / (total_frames / (skip_frames + 1)), 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_num}... ({processed_count} frames analyzed)")
                    
                    # Store frame results
                    frame_data = {
                        'frame': frame_num,
                        'total_objects': result.total_objects,
                        'high_hazards': result.high_hazards,
                        'medium_hazards': result.medium_hazards,
                        'low_hazards': result.low_hazards,
                        'detections': [
                            {
                                'object': d.object_name,
                                'confidence': d.confidence,
                                'hazard_level': d.hazard_level.value
                            }
                            for d in result.detections
                        ]
                    }
                    frame_results.append(frame_data)
                    
                    # Collect all detections
                    for d in result.detections:
                        all_detections.append({
                            'frame': frame_num,
                            'object': d.object_name,
                            'confidence': d.confidence,
                            'hazard_level': d.hazard_level.value
                        })
                    
                    # Display frame (every 5th processed frame)
                    if processed_count % 5 == 0:
                        annotated_rgb = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                        frame_display.image(annotated_rgb, use_container_width=True,
                                          caption=f"Frame {frame_num}")
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Processing complete! Analyzed {processed_count} frames.")
                
                # Display Results
                st.markdown("---")
                st.subheader("📊 Video Analysis Results")
                
                if frame_results:
                    # Calculate statistics
                    stats = calculate_video_statistics([
                        {
                            'frame_number': f['frame'],
                            'total_objects': f['total_objects'],
                            'hazard_summary': {
                                'HIGH': f['high_hazards'],
                                'MEDIUM': f['medium_hazards'],
                                'LOW': f['low_hazards']
                            },
                            'detections': f['detections']
                        }
                        for f in frame_results
                    ])
                    
                    # Summary metrics
                    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                    
                    with sum_col1:
                        st.metric("Frames Processed", stats.get('total_frames_processed', 0))
                    with sum_col2:
                        st.metric("Total Detections", stats.get('total_detections', 0))
                    with sum_col3:
                        avg_det = stats.get('detections_per_frame', {}).get('mean', 0)
                        st.metric("Avg Detections/Frame", f"{avg_det:.1f}")
                    with sum_col4:
                        max_det = stats.get('detections_per_frame', {}).get('max', 0)
                        st.metric("Max Detections/Frame", max_det)
                    
                    # Timeline chart
                    st.plotly_chart(
                        create_timeline_chart(frame_results),
                        use_container_width=True
                    )
                    
                    # Object distribution
                    if all_detections:
                        dist_col1, dist_col2 = st.columns(2)
                        
                        with dist_col1:
                            st.plotly_chart(
                                create_object_distribution_chart(all_detections),
                                use_container_width=True
                            )
                        
                        with dist_col2:
                            st.plotly_chart(
                                create_confidence_histogram(all_detections),
                                use_container_width=True
                            )
                        
                        # Detection summary table
                        st.subheader("📋 Detection Summary")
                        df = pd.DataFrame(all_detections)
                        summary = df.groupby('object').agg({
                            'confidence': ['count', 'mean'],
                            'hazard_level': 'first'
                        }).round(3)
                        summary.columns = ['Count', 'Avg Confidence', 'Hazard Level']
                        summary = summary.sort_values('Count', ascending=False)
                        st.dataframe(summary, use_container_width=True)
                    
                    # Alerts
                    total_high = sum(f['high_hazards'] for f in frame_results)
                    if total_high > 10:
                        st.error(f"⚠️ **HIGH ALERT**: {total_high} high-hazard detections throughout the video!")
                    elif total_high > 0:
                        st.warning(f"⚡ **CAUTION**: {total_high} high-hazard detections found.")
                    else:
                        st.success("✅ **CLEAR**: No high-hazard objects detected in this video.")
            
            # Cleanup
            try:
                os.unlink(video_path)
            except:
                pass
    
    # ==================== STATISTICS TAB ====================
    with tab3:
        st.header("📈 System Statistics & Performance")
        
        st.subheader("Model Performance Metrics")
        st.markdown("""
        The YOLOv8 Nano model used in this application provides a balance between
        speed and accuracy, making it suitable for real-time detection on CPU.
        """)
        
        # Model comparison table
        st.subheader("YOLOv8 Model Variants")
        model_data = {
            'Model': ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x'],
            'Parameters': ['3.2M', '11.2M', '25.9M', '43.7M', '68.2M'],
            'mAP (COCO)': ['37.3', '44.9', '50.2', '52.9', '53.9'],
            'Speed (CPU)': ['Fast', 'Medium', 'Slow', 'Very Slow', 'Slowest'],
            'Use Case': ['Real-time/Edge', 'Balanced', 'Accuracy', 'High Accuracy', 'Maximum Accuracy']
        }
        st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)
        
        # Hazard classification explanation
        st.subheader("Hazard Classification Logic")
        hazard_data = {
            'Hazard Level': ['🔴 HIGH', '🟡 MEDIUM', '🟢 LOW'],
            'Objects': [
                'person, dog, cat, horse, cow, sheep',
                'bicycle, motorcycle, bench, fire hydrant',
                'car, bus, truck, train'
            ],
            'Reasoning': [
                'Unpredictable movement patterns, high injury risk',
                'Moderate risk, semi-predictable behavior',
                'Predictable movement, follows traffic rules'
            ],
            'Recommended Action': [
                'Immediate braking/evasive action',
                'Reduce speed, increase awareness',
                'Normal driving, maintain distance'
            ]
        }
        st.dataframe(pd.DataFrame(hazard_data), use_container_width=True, hide_index=True)
        
        # Sample visualization
        st.subheader("Sample Detection Distribution")
        sample_data = get_sample_data()
        fig = px.bar(
            x=sample_data['objects'],
            y=sample_data['counts'],
            color=sample_data['hazard_levels'],
            color_discrete_map={'HIGH': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#00cc00'},
            title='Example Object Distribution (Sample Data)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== ABOUT TAB ====================
    with tab4:
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ## Road Safety Object Detection Dashboard
        
        This application uses computer vision and deep learning to detect road objects
        and assess potential hazards for autonomous vehicle navigation.
        
        ### Features
        - **Single Image Detection**: Upload and analyze individual images
        - **Video Processing**: Frame-by-frame analysis of video files
        - **Hazard Classification**: Automatic risk assessment of detected objects
        - **Interactive Visualizations**: Charts and graphs for data analysis
        
        ### Technology Stack
        - **Detection Model**: YOLOv8 (You Only Look Once v8)
        - **Framework**: Streamlit for web interface
        - **Visualization**: Plotly for interactive charts
        - **Processing**: OpenCV for image/video handling
        
        ### Use Cases
        1. Autonomous vehicle development and testing
        2. Road safety analysis
        3. Traffic monitoring
        4. Driver assistance systems
        
        ### Limitations
        - Works best with clear, well-lit images
        - May struggle with heavily occluded objects
        - Optimized for road scenes (urban/suburban)
        
        ### References
        - Ultralytics YOLOv8 Documentation
        - OpenCV Library
        - COCO Dataset (training data)
        
        ---
        
        **Developed for Computer Vision Module Assignment**
        
        *[Your Name] | [Your University] | 2025*
        """)
        
        st.markdown("---")
        
        # System info
        st.subheader("System Information")
        import sys
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.code(f"""
Python Version: {sys.version.split()[0]}
OpenCV Version: {cv2.__version__}
Streamlit Version: {st.__version__}
            """)
        with info_col2:
            st.code(f"""
NumPy Version: {np.__version__}
Pandas Version: {pd.__version__}
Detection Model: YOLOv8n
            """)


# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()
