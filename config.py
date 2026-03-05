"""
Configuration settings for the Road Safety Detection Dashboard.
"""

# Model Configuration
MODEL_CONFIG = {
    'default_model_size': 'n',  # n, s, m, l, x
    'default_device': 'cpu',
    'default_confidence': 0.5,
}

# Hazard Classification
HAZARD_CONFIG = {
    'high_threshold': 3,      # Score >= 3 is high risk
    'medium_threshold': 2,    # Score >= 2 is medium risk
    'weights': {
        'HIGH': 3,
        'MEDIUM': 2,
        'LOW': 1
    }
}

# Video Processing
VIDEO_CONFIG = {
    'default_skip_frames': 2,
    'default_max_frames': 100,
    'supported_formats': ['mp4', 'avi', 'mov', 'mkv']
}

# Image Processing
IMAGE_CONFIG = {
    'supported_formats': ['jpg', 'jpeg', 'png', 'bmp'],
    'max_size': 1920
}

# UI Configuration
UI_CONFIG = {
    'page_title': 'Road Safety Detection Dashboard',
    'page_icon': '🚗',
    'layout': 'wide'
}
