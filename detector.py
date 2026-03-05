"""
Road Object Detection Module
============================
This module provides object detection capabilities for road safety applications.
It uses YOLOv8 for detecting vehicles, pedestrians, and other road objects.

Author: Isaac O Adeboyejo
Date: 2026
"""

from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, List, Generator, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class HazardLevel(Enum):
    """Enumeration for hazard levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


@dataclass
class Detection:
    """Data class to store detection information."""
    object_name: str
    confidence: float
    hazard_level: HazardLevel
    bbox: List[float]
    class_id: int


@dataclass
class FrameResult:
    """Data class to store frame processing results."""
    frame_number: int
    detections: List[Detection]
    annotated_frame: np.ndarray
    total_objects: int
    high_hazards: int
    medium_hazards: int
    low_hazards: int


class RoadObjectDetector:
    """
    A class for detecting road objects using YOLOv8.
    
    This detector is specifically configured to identify objects
    relevant to road safety, including pedestrians, vehicles,
    cyclists, and animals.
    
    Attributes:
        model: The YOLOv8 model instance
        road_objects: Dictionary mapping class IDs to object names
        hazard_levels: Dictionary mapping object names to hazard levels
    """
    
    # Class-level constants for road-relevant objects from COCO dataset
    ROAD_OBJECTS = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        6: 'train',
        7: 'truck',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        12: 'parking meter',
        13: 'bench',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
    }
    
    # Hazard classification based on unpredictability and risk
    HAZARD_CLASSIFICATION = {
        'person': HazardLevel.HIGH,
        'dog': HazardLevel.HIGH,
        'cat': HazardLevel.HIGH,
        'horse': HazardLevel.HIGH,
        'cow': HazardLevel.HIGH,
        'sheep': HazardLevel.HIGH,
        'bicycle': HazardLevel.MEDIUM,
        'motorcycle': HazardLevel.MEDIUM,
        'bench': HazardLevel.MEDIUM,
        'fire hydrant': HazardLevel.MEDIUM,
        'parking meter': HazardLevel.MEDIUM,
        'car': HazardLevel.LOW,
        'bus': HazardLevel.LOW,
        'truck': HazardLevel.LOW,
        'train': HazardLevel.LOW,
        'traffic light': HazardLevel.NONE,
        'stop sign': HazardLevel.NONE,
    }
    
    def __init__(self, model_size: str = 'n', device: str = 'cpu'):
        """
        Initialize the Road Object Detector.
        
        Args:
            model_size: Size of YOLOv8 model ('n', 's', 'm', 'l', 'x')
                       'n' = nano (fastest, least accurate)
                       's' = small
                       'm' = medium
                       'l' = large
                       'x' = extra large (slowest, most accurate)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_size = model_size
        self.device = device
        
        # Load the YOLOv8 model
        print(f"Loading YOLOv8{model_size} model...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        print("Model loaded successfully!")
        
        # Store configuration
        self.road_objects = self.ROAD_OBJECTS
        self.hazard_levels = self.HAZARD_CLASSIFICATION
    
    def _process_results(self, results, confidence_threshold: float) -> List[Detection]:
        """
        Process YOLO results and extract relevant detections.
        
        Args:
            results: Raw YOLO detection results
            confidence_threshold: Minimum confidence for valid detections
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Only include road-relevant objects above confidence threshold
            if class_id in self.road_objects and conf >= confidence_threshold:
                obj_name = self.road_objects[class_id]
                hazard = self.hazard_levels.get(obj_name, HazardLevel.LOW)
                
                detection = Detection(
                    object_name=obj_name,
                    confidence=conf,
                    hazard_level=hazard,
                    bbox=box.xyxy[0].tolist(),
                    class_id=class_id
                )
                detections.append(detection)
        
        return detections
    
    def _count_hazards(self, detections: List[Detection]) -> Tuple[int, int, int]:
        """
        Count detections by hazard level.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Tuple of (high_count, medium_count, low_count)
        """
        high = sum(1 for d in detections if d.hazard_level == HazardLevel.HIGH)
        medium = sum(1 for d in detections if d.hazard_level == HazardLevel.MEDIUM)
        low = sum(1 for d in detections if d.hazard_level == HazardLevel.LOW)
        return high, medium, low
    
    def detect_image(
        self, 
        image_path: str, 
        confidence: float = 0.5,
        save_result: bool = False,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Detect objects in a single image.
        
        Args:
            image_path: Path to the input image
            confidence: Minimum confidence threshold (0.0 to 1.0)
            save_result: Whether to save the annotated image
            output_path: Path to save the annotated image
            
        Returns:
            Dictionary containing:
                - total_objects: Number of objects detected
                - detections: List of detection dictionaries
                - annotated_image: Image with bounding boxes drawn
                - hazard_summary: Count of each hazard level
        """
        # Run inference
        results = self.model(
            image_path, 
            conf=confidence,
            device=self.device,
            verbose=False
        )[0]
        
        # Process detections
        detections = self._process_results(results, confidence)
        high, medium, low = self._count_hazards(detections)
        
        # Get annotated image
        annotated_image = results.plot()
        
        # Save if requested
        if save_result and output_path:
            cv2.imwrite(output_path, annotated_image)
        
        # Convert detections to dictionaries for easier handling
        detection_dicts = [
            {
                'object': d.object_name,
                'confidence': round(d.confidence, 4),
                'hazard_level': d.hazard_level.value,
                'bbox': d.bbox
            }
            for d in detections
        ]
        
        return {
            'total_objects': len(detections),
            'detections': detection_dicts,
            'annotated_image': annotated_image,
            'hazard_summary': {
                'HIGH': high,
                'MEDIUM': medium,
                'LOW': low
            }
        }
    
    def detect_video(
        self, 
        video_path: str, 
        confidence: float = 0.5,
        skip_frames: int = 0
    ) -> Generator[FrameResult, None, None]:
        """
        Process a video file and yield frame-by-frame results.
        
        Args:
            video_path: Path to the input video file
            confidence: Minimum confidence threshold
            skip_frames: Number of frames to skip between detections
                        (0 = process every frame, 1 = every other frame, etc.)
            
        Yields:
            FrameResult objects containing detection data for each processed frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_count = 0
        processed_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                break
            
            frame_count += 1
            
            # Skip frames if specified
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                continue
            
            processed_count += 1
            
            # Run detection
            results = self.model(
                frame, 
                conf=confidence,
                device=self.device,
                verbose=False
            )[0]
            
            # Process detections
            detections = self._process_results(results, confidence)
            high, medium, low = self._count_hazards(detections)
            
            # Create frame result
            frame_result = FrameResult(
                frame_number=frame_count,
                detections=detections,
                annotated_frame=results.plot(),
                total_objects=len(detections),
                high_hazards=high,
                medium_hazards=medium,
                low_hazards=low
            )
            
            yield frame_result
        
        cap.release()
    
    def detect_frame(self, frame: np.ndarray, confidence: float = 0.5) -> Dict:
        """
        Detect objects in a single frame (numpy array).
        
        Args:
            frame: Input frame as numpy array (BGR format)
            confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with detection results
        """
        results = self.model(
            frame, 
            conf=confidence,
            device=self.device,
            verbose=False
        )[0]
        
        detections = self._process_results(results, confidence)
        high, medium, low = self._count_hazards(detections)
        
        detection_dicts = [
            {
                'object': d.object_name,
                'confidence': round(d.confidence, 4),
                'hazard_level': d.hazard_level.value,
                'bbox': d.bbox
            }
            for d in detections
        ]
        
        return {
            'total_objects': len(detections),
            'detections': detection_dicts,
            'annotated_image': results.plot(),
            'hazard_summary': {
                'HIGH': high,
                'MEDIUM': medium,
                'LOW': low
            }
        }
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'YOLOv8',
            'model_size': self.model_size,
            'device': self.device,
            'num_classes': len(self.road_objects),
            'detectable_objects': list(self.road_objects.values())
        }
    
    def get_hazard_info(self) -> Dict[str, List[str]]:
        """
        Get information about hazard classifications.
        
        Returns:
            Dictionary mapping hazard levels to lists of objects
        """
        hazard_info = {
            'HIGH': [],
            'MEDIUM': [],
            'LOW': [],
            'NONE': []
        }
        
        for obj, level in self.hazard_levels.items():
            hazard_info[level.value].append(obj)
        
        return hazard_info


class ObjectTracker:
    """
    Simple object tracking across frames.
    
    This class provides basic tracking functionality to maintain
    object identities across consecutive video frames.
    """
    
    def __init__(self, max_distance: float = 100.0):
        """
        Initialize the object tracker.
        
        Args:
            max_distance: Maximum distance for matching objects between frames
        """
        self.max_distance = max_distance
        self.tracked_objects = {}
        self.next_id = 0
        self.frame_count = 0
    
    def _calculate_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculate the center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(
        self, 
        point1: Tuple[float, float], 
        point2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries with 'bbox' key
            
        Returns:
            List of detections with added 'track_id' key
        """
        self.frame_count += 1
        
        # Calculate centers for new detections
        new_centers = [self._calculate_center(d['bbox']) for d in detections]
        
        # Simple nearest-neighbor matching
        matched_detections = []
        used_ids = set()
        
        for i, detection in enumerate(detections):
            new_center = new_centers[i]
            best_match_id = None
            best_distance = self.max_distance
            
            # Find closest tracked object
            for obj_id, obj_data in self.tracked_objects.items():
                if obj_id in used_ids:
                    continue
                    
                distance = self._calculate_distance(new_center, obj_data['center'])
                
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = obj_id
            
            # Assign ID (existing or new)
            if best_match_id is not None:
                track_id = best_match_id
                used_ids.add(track_id)
            else:
                track_id = self.next_id
                self.next_id += 1
            
            # Update tracked object
            self.tracked_objects[track_id] = {
                'center': new_center,
                'last_seen': self.frame_count,
                'object': detection['object']
            }
            
            # Add track ID to detection
            detection_with_id = detection.copy()
            detection_with_id['track_id'] = track_id
            matched_detections.append(detection_with_id)
        
        # Remove old tracked objects (not seen in last 10 frames)
        self.tracked_objects = {
            k: v for k, v in self.tracked_objects.items()
            if self.frame_count - v['last_seen'] < 10
        }
        
        return matched_detections
    
    def reset(self):
        """Reset the tracker state."""
        self.tracked_objects = {}
        self.next_id = 0
        self.frame_count = 0


# Test the module if run directly
if __name__ == "__main__":
    print("=" * 50)
    print("Road Object Detector - Module Test")
    print("=" * 50)
    
    # Initialize detector
    detector = RoadObjectDetector(model_size='n')
    
    # Print model info
    print("\nModel Information:")
    info = detector.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Print hazard info
    print("\nHazard Classifications:")
    hazard_info = detector.get_hazard_info()
    for level, objects in hazard_info.items():
        if objects:
            print(f"  {level}: {', '.join(objects)}")
    
    print("\n" + "=" * 50)
    print("Module loaded successfully!")
    print("=" * 50)
