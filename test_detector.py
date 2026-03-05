"""
Unit tests for the Road Object Detector module.
"""

import unittest
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import RoadObjectDetector, HazardLevel


class TestRoadObjectDetector(unittest.TestCase):
    """Test cases for RoadObjectDetector class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.detector = RoadObjectDetector(model_size='n')
    
    def test_initialization(self):
        """Test detector initializes correctly."""
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(self.detector.model_size, 'n')
        self.assertEqual(self.detector.device, 'cpu')
    
    def test_road_objects_defined(self):
        """Test that road objects are properly defined."""
        self.assertIn(0, self.detector.road_objects)  # person
        self.assertIn(2, self.detector.road_objects)  # car
        self.assertEqual(self.detector.road_objects[0], 'person')
    
    def test_hazard_levels_defined(self):
        """Test that hazard levels are properly defined."""
        self.assertEqual(
            self.detector.hazard_levels['person'],
            HazardLevel.HIGH
        )
        self.assertEqual(
            self.detector.hazard_levels['car'],
            HazardLevel.LOW
        )
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        info = self.detector.get_model_info()
        self.assertIn('model_type', info)
        self.assertIn('model_size', info)
        self.assertEqual(info['model_type'], 'YOLOv8')
    
    def test_get_hazard_info(self):
        """Test hazard info retrieval."""
        info = self.detector.get_hazard_info()
        self.assertIn('HIGH', info)
        self.assertIn('MEDIUM', info)
        self.assertIn('LOW', info)
        self.assertIn('person', info['HIGH'])
    
    def test_detect_frame_with_blank_image(self):
        """Test detection on a blank image."""
        blank_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = self.detector.detect_frame(blank_image)
        
        self.assertIn('total_objects', result)
        self.assertIn('detections', result)
        self.assertIn('annotated_image', result)
        self.assertEqual(result['total_objects'], 0)
    
    def test_count_hazards(self):
        """Test hazard counting function."""
        from detector import Detection
        
        detections = [
            Detection('person', 0.9, HazardLevel.HIGH, [0,0,100,100], 0),
            Detection('car', 0.8, HazardLevel.LOW, [0,0,100,100], 2),
            Detection('bicycle', 0.7, HazardLevel.MEDIUM, [0,0,100,100], 1),
        ]
        
        high, medium, low = self.detector._count_hazards(detections)
        
        self.assertEqual(high, 1)
        self.assertEqual(medium, 1)
        self.assertEqual(low, 1)


class TestHazardLevel(unittest.TestCase):
    """Test cases for HazardLevel enum."""
    
    def test_hazard_level_values(self):
        """Test hazard level enum values."""
        self.assertEqual(HazardLevel.HIGH.value, "HIGH")
        self.assertEqual(HazardLevel.MEDIUM.value, "MEDIUM")
        self.assertEqual(HazardLevel.LOW.value, "LOW")


if __name__ == '__main__':
    unittest.main()
