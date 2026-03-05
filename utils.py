"""
Utility Functions for Road Safety Dashboard
============================================
This module provides helper functions for video processing,
image manipulation, and statistical calculations.

Author: [Your Name]
Date: 2025
"""

import cv2
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def get_video_info(video_path: str) -> Dict:
    """
    Extract detailed information about a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video properties:
            - frame_count: Total number of frames
            - fps: Frames per second
            - width: Frame width in pixels
            - height: Frame height in pixels
            - duration_seconds: Video duration in seconds
            - duration_formatted: Duration as MM:SS string
            - codec: Video codec fourcc code
            - file_size_mb: File size in megabytes
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Extract properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # Calculate duration
    duration_seconds = frame_count / fps if fps > 0 else 0
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    duration_formatted = f"{minutes:02d}:{seconds:02d}"
    
    # Get file size
    file_size_bytes = os.path.getsize(video_path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    
    cap.release()
    
    return {
        'frame_count': frame_count,
        'fps': round(fps, 2),
        'width': width,
        'height': height,
        'duration_seconds': round(duration_seconds, 2),
        'duration_formatted': duration_formatted,
        'codec': codec,
        'file_size_mb': file_size_mb,
        'resolution': f"{width}x{height}"
    }


def get_image_info(image_path: str) -> Dict:
    """
    Extract information about an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing image properties
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")
    
    height, width, channels = image.shape
    file_size_bytes = os.path.getsize(image_path)
    file_size_kb = round(file_size_bytes / 1024, 2)
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'resolution': f"{width}x{height}",
        'file_size_kb': file_size_kb,
        'aspect_ratio': round(width / height, 2)
    }


def resize_image(
    image: np.ndarray, 
    max_size: int = 640,
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize an image while optionally maintaining aspect ratio.
    
    Args:
        image: Input image as numpy array (BGR format)
        max_size: Maximum dimension (width or height)
        maintain_aspect: Whether to maintain the original aspect ratio
        
    Returns:
        Resized image as numpy array
    """
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
    
    if maintain_aspect:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
    else:
        new_width = max_size
        new_height = max_size
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def create_output_directory(base_path: str = "outputs") -> str:
    """
    Create the output directory structure.
    
    Args:
        base_path: Base path for outputs
        
    Returns:
        Path to the created directory
    """
    directories = [
        base_path,
        os.path.join(base_path, "processed_images"),
        os.path.join(base_path, "processed_videos"),
        os.path.join(base_path, "reports"),
        os.path.join(base_path, "statistics")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return base_path


def calculate_detection_statistics(detections: List[Dict]) -> Dict:
    """
    Calculate comprehensive statistics from a list of detections.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary containing various statistics
    """
    if not detections:
        return {
            'total_detections': 0,
            'unique_objects': 0,
            'object_counts': {},
            'hazard_distribution': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'average_confidence': 0,
            'confidence_range': {'min': 0, 'max': 0}
        }
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(detections)
    
    # Object counts
    object_counts = df['object'].value_counts().to_dict()
    
    # Hazard distribution
    hazard_distribution = df['hazard_level'].value_counts().to_dict()
    for level in ['HIGH', 'MEDIUM', 'LOW']:
        if level not in hazard_distribution:
            hazard_distribution[level] = 0
    
    # Confidence statistics
    confidences = df['confidence'].tolist()
    
    return {
        'total_detections': len(detections),
        'unique_objects': df['object'].nunique(),
        'object_counts': object_counts,
        'hazard_distribution': hazard_distribution,
        'average_confidence': round(np.mean(confidences), 4),
        'confidence_range': {
            'min': round(min(confidences), 4),
            'max': round(max(confidences), 4)
        },
        'most_common_object': df['object'].mode().iloc[0] if len(df) > 0 else None
    }


def calculate_video_statistics(frame_results: List[Dict]) -> Dict:
    """
    Calculate statistics from video processing results.
    
    Args:
        frame_results: List of dictionaries containing frame-by-frame results
        
    Returns:
        Dictionary with comprehensive video statistics
    """
    if not frame_results:
        return {}
    
    # Extract data
    all_detections = []
    frame_detection_counts = []
    frame_hazard_counts = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
    
    for frame in frame_results:
        frame_detection_counts.append(frame.get('total_objects', 0))
        
        if 'hazard_summary' in frame:
            for level in ['HIGH', 'MEDIUM', 'LOW']:
                frame_hazard_counts[level].append(frame['hazard_summary'].get(level, 0))
        
        if 'detections' in frame:
            for det in frame['detections']:
                det_copy = det.copy()
                det_copy['frame'] = frame.get('frame_number', 0)
                all_detections.append(det_copy)
    
    # Calculate statistics
    stats = {
        'total_frames_processed': len(frame_results),
        'total_detections': len(all_detections),
        'detections_per_frame': {
            'mean': round(np.mean(frame_detection_counts), 2),
            'max': max(frame_detection_counts),
            'min': min(frame_detection_counts),
            'std': round(np.std(frame_detection_counts), 2)
        }
    }
    
    # Add hazard statistics
    for level in ['HIGH', 'MEDIUM', 'LOW']:
        if frame_hazard_counts[level]:
            stats[f'{level.lower()}_hazard_stats'] = {
                'total': sum(frame_hazard_counts[level]),
                'max_per_frame': max(frame_hazard_counts[level]),
                'frames_with_hazard': sum(1 for x in frame_hazard_counts[level] if x > 0)
            }
    
    # Add detection statistics
    if all_detections:
        stats['detection_statistics'] = calculate_detection_statistics(all_detections)
    
    return stats


def generate_detection_report(
    statistics: Dict,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a text report from detection statistics.
    
    Args:
        statistics: Dictionary of statistics from calculate_detection_statistics
        output_path: Optional path to save the report
        
    Returns:
        Report as a formatted string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines = [
        "=" * 60,
        "ROAD SAFETY DETECTION REPORT",
        f"Generated: {timestamp}",
        "=" * 60,
        "",
        "SUMMARY",
        "-" * 40,
        f"Total Detections: {statistics.get('total_detections', 0)}",
        f"Unique Object Types: {statistics.get('unique_objects', 0)}",
        f"Average Confidence: {statistics.get('average_confidence', 0):.2%}",
        "",
        "HAZARD DISTRIBUTION",
        "-" * 40,
    ]
    
    hazard_dist = statistics.get('hazard_distribution', {})
    for level in ['HIGH', 'MEDIUM', 'LOW']:
        count = hazard_dist.get(level, 0)
        report_lines.append(f"  {level}: {count}")
    
    report_lines.extend([
        "",
        "OBJECT COUNTS",
        "-" * 40,
    ])
    
    object_counts = statistics.get('object_counts', {})
    for obj, count in sorted(object_counts.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {obj}: {count}")
    
    report_lines.extend([
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


def save_results_json(
    results: Dict,
    output_path: str,
    indent: int = 2
) -> None:
    """
    Save detection results to a JSON file.
    
    Args:
        results: Dictionary of results to save
        output_path: Path to save the JSON file
        indent: Indentation level for formatting
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    converted_results = convert_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(converted_results, f, indent=indent)


def draw_hazard_overlay(
    image: np.ndarray,
    detections: List[Dict],
    alpha: float = 0.3
) -> np.ndarray:
    """
    Draw a semi-transparent hazard overlay on an image.
    
    Args:
        image: Input image as numpy array
        detections: List of detection dictionaries with bbox and hazard_level
        alpha: Transparency level (0.0 to 1.0)
        
    Returns:
        Image with hazard overlay
    """
    overlay = image.copy()
    
    # Color mapping for hazard levels (BGR format)
    hazard_colors = {
        'HIGH': (0, 0, 255),      # Red
        'MEDIUM': (0, 165, 255),  # Orange
        'LOW': (0, 255, 0),       # Green
        'NONE': (255, 255, 255)   # White
    }
    
    for detection in detections:
        bbox = detection.get('bbox', [])
        hazard = detection.get('hazard_level', 'LOW')
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            color = hazard_colors.get(hazard, (255, 255, 255))
            
            # Draw filled rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    # Blend overlay with original image
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 30,
    max_frames: Optional[int] = None
) -> List[str]:
    """
    Extract frames from a video at specified intervals.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame
        max_frames: Maximum number of frames to extract (None for all)
        
    Returns:
        List of paths to extracted frame images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break
        
        frame_count += 1
        
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            frame_paths.append(output_path)
            extracted_count += 1
            
            if max_frames and extracted_count >= max_frames:
                break
    
    cap.release()
    return frame_paths


# Test the module if run directly
if __name__ == "__main__":
    print("Utility module loaded successfully!")
    
    # Test directory creation
    output_dir = create_output_directory("test_outputs")
    print(f"Created output directory: {output_dir}")
    
    # Clean up test directory
    import shutil
    if os.path.exists("test_outputs"):
        shutil.rmtree("test_outputs")
    print("Test completed successfully!")
