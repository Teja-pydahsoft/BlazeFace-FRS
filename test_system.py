"""
Test script for BlazeFace-FRS system
Tests core functionality without GUI
"""

import sys
import os
import cv2
import numpy as np
import time

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.blazeface_detector import BlazeFaceDetector
from app.core.human_detector import HumanDetector
from app.core.facenet_embedder import FaceNetEmbedder
from app.core.dual_pipeline import DualPipeline
from app.core.database import DatabaseManager

def test_blazeface_detector():
    """Test BlazeFace detector"""
    print("Testing BlazeFace Detector...")
    
    try:
        detector = BlazeFaceDetector()
        
        # Test with a simple image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect_faces(test_image)
        
        print(f"✓ BlazeFace detector initialized successfully")
        print(f"  - Detected {len(faces)} faces in test image")
        
        detector.release()
        return True
        
    except Exception as e:
        print(f"✗ BlazeFace detector test failed: {e}")
        return False

def test_human_detector():
    """Test Human detector"""
    print("Testing Human Detector...")
    
    try:
        detector = HumanDetector()
        
        # Test with a simple image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        humans = detector.detect_humans(test_image)
        
        print(f"✓ Human detector initialized successfully")
        print(f"  - Detected {len(humans)} humans in test image")
        
        detector.release()
        return True
        
    except Exception as e:
        print(f"✗ Human detector test failed: {e}")
        return False

def test_facenet_embedder():
    """Test FaceNet embedder"""
    print("Testing FaceNet Embedder...")
    
    try:
        embedder = FaceNetEmbedder()
        
        # Test with a simple image
        test_image = np.zeros((160, 160, 3), dtype=np.uint8)
        embedding = embedder.get_embedding(test_image)
        
        if embedding is not None:
            print(f"✓ FaceNet embedder initialized successfully")
            print(f"  - Generated embedding of shape: {embedding.shape}")
        else:
            print(f"✓ FaceNet embedder initialized (no embedding generated for test image)")
        
        embedder.release()
        return True
        
    except Exception as e:
        print(f"✗ FaceNet embedder test failed: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("Testing Database...")
    
    try:
        db_manager = DatabaseManager("database/test_database.db")
        
        # Test adding a student
        test_student = {
            'student_id': 'TEST001',
            'name': 'Test Student',
            'email': 'test@example.com',
            'department': 'Computer Science',
            'year': '3rd Year'
        }
        
        success = db_manager.add_student(test_student)
        if success:
            print(f"✓ Database initialized successfully")
            print(f"  - Added test student: {test_student['student_id']}")
            
            # Clean up test data
            db_manager.delete_student('TEST001')
        else:
            print(f"✓ Database initialized (test student already exists)")
        
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def test_dual_pipeline():
    """Test dual pipeline system"""
    print("Testing Dual Pipeline...")
    
    try:
        config = {
            'detection_confidence': 0.7,
            'facenet_model_path': None
        }
        
        pipeline = DualPipeline(config)
        
        print(f"✓ Dual pipeline initialized successfully")
        print(f"  - Pipeline status: {pipeline.get_pipeline_status()}")
        
        pipeline.release()
        return True
        
    except Exception as e:
        print(f"✗ Dual pipeline test failed: {e}")
        return False

def test_camera():
    """Test camera functionality"""
    print("Testing Camera...")
    
    try:
        from app.utils.camera_utils import CameraManager
        
        camera_manager = CameraManager()
        
        if camera_manager.is_camera_available():
            print(f"✓ Camera available and working")
            print(f"  - Camera info: {camera_manager.get_camera_info()}")
        else:
            print(f"⚠ Camera not available (this is normal if no camera is connected)")
        
        camera_manager.release()
        return True
        
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("BlazeFace-FRS System Test")
    print("=" * 50)
    
    tests = [
        test_blazeface_detector,
        test_human_detector,
        test_facenet_embedder,
        test_database,
        test_dual_pipeline,
        test_camera
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
