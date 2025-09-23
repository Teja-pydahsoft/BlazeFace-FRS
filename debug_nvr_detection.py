"""
Debug script for NVR camera face detection issues
Tests different detection parameters and image preprocessing
"""

import sys
import os
import cv2
import numpy as np
import time
import logging

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.camera_utils import CameraManager
from app.core.blazeface_detector import BlazeFaceDetector

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_detection_parameters():
    """Test different detection parameters"""
    print("=" * 60)
    print("TESTING DIFFERENT DETECTION PARAMETERS")
    print("=" * 60)
    
    # NVR camera configuration
    nvr_url = "rtsp://admin:nvr@pydah@192.168.3.235:554/stream1"
    
    try:
        # Initialize camera
        print(f"Initializing NVR camera: {nvr_url}")
        camera_manager = CameraManager(nvr_url)
        
        if not camera_manager.is_initialized:
            print("❌ Failed to initialize NVR camera")
            return False
        
        print("✅ NVR camera initialized successfully")
        
        # Get a test frame
        ret, frame = camera_manager.get_frame()
        if not ret or frame is None:
            print("❌ Failed to capture test frame")
            return False
        
        print(f"✅ Captured test frame: {frame.shape}")
        
        # Save original frame for inspection
        cv2.imwrite("debug_original_frame.jpg", frame)
        print("💾 Saved original frame: debug_original_frame.jpg")
        
        # Test different confidence thresholds
        confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for conf_thresh in confidence_thresholds:
            print(f"\n--- Testing confidence threshold: {conf_thresh} ---")
            
            # Initialize detector with current threshold
            detector = BlazeFaceDetector(min_detection_confidence=conf_thresh)
            
            # Test detection
            faces = detector.detect_faces(frame)
            
            if faces:
                print(f"  ✅ Found {len(faces)} faces with confidence >= {conf_thresh}")
                for i, face in enumerate(faces):
                    x, y, w, h, conf = face
                    print(f"    Face {i}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                
                # Draw and save frame with detections
                frame_with_faces = detector.draw_faces(frame.copy(), faces)
                cv2.imwrite(f"debug_detection_conf_{conf_thresh}.jpg", frame_with_faces)
                print(f"  💾 Saved detection frame: debug_detection_conf_{conf_thresh}.jpg")
            else:
                print(f"  ❌ No faces found with confidence >= {conf_thresh}")
            
            detector.release()
        
        # Test different model selections
        print(f"\n--- Testing different model selections ---")
        model_selections = [0, 1]  # 0 for short-range, 1 for full-range
        
        for model_sel in model_selections:
            print(f"\n--- Testing model selection: {model_sel} ---")
            
            detector = BlazeFaceDetector(
                min_detection_confidence=0.3,
                model_selection=model_sel
            )
            
            faces = detector.detect_faces(frame)
            
            if faces:
                print(f"  ✅ Model {model_sel}: Found {len(faces)} faces")
                for i, face in enumerate(faces):
                    x, y, w, h, conf = face
                    print(f"    Face {i}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                
                frame_with_faces = detector.draw_faces(frame.copy(), faces)
                cv2.imwrite(f"debug_model_{model_sel}.jpg", frame_with_faces)
                print(f"  💾 Saved model frame: debug_model_{model_sel}.jpg")
            else:
                print(f"  ❌ Model {model_sel}: No faces found")
            
            detector.release()
        
        # Test image preprocessing
        print(f"\n--- Testing image preprocessing ---")
        
        # Test different image sizes
        sizes = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
        
        for width, height in sizes:
            print(f"\n--- Testing size: {width}x{height} ---")
            
            # Resize frame
            resized_frame = cv2.resize(frame, (width, height))
            
            detector = BlazeFaceDetector(min_detection_confidence=0.3)
            faces = detector.detect_faces(resized_frame)
            
            if faces:
                print(f"  ✅ Size {width}x{height}: Found {len(faces)} faces")
                for i, face in enumerate(faces):
                    x, y, w, h, conf = face
                    print(f"    Face {i}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                
                frame_with_faces = detector.draw_faces(resized_frame.copy(), faces)
                cv2.imwrite(f"debug_size_{width}x{height}.jpg", frame_with_faces)
                print(f"  💾 Saved size frame: debug_size_{width}x{height}.jpg")
            else:
                print(f"  ❌ Size {width}x{height}: No faces found")
            
            detector.release()
        
        # Test different color spaces
        print(f"\n--- Testing different color spaces ---")
        
        # Convert to different color spaces
        color_spaces = {
            'BGR': frame,
            'RGB': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            'GRAY': cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            'HSV': cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
            'LAB': cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        }
        
        for color_name, color_frame in color_spaces.items():
            print(f"\n--- Testing color space: {color_name} ---")
            
            # Convert back to BGR if needed
            if len(color_frame.shape) == 2:  # Grayscale
                test_frame = cv2.cvtColor(color_frame, cv2.COLOR_GRAY2BGR)
            elif color_frame.shape[2] == 3:  # RGB, HSV, LAB
                test_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR) if color_name == 'RGB' else color_frame
            else:
                test_frame = color_frame
            
            detector = BlazeFaceDetector(min_detection_confidence=0.3)
            faces = detector.detect_faces(test_frame)
            
            if faces:
                print(f"  ✅ Color {color_name}: Found {len(faces)} faces")
                for i, face in enumerate(faces):
                    x, y, w, h, conf = face
                    print(f"    Face {i}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                
                frame_with_faces = detector.draw_faces(test_frame.copy(), faces)
                cv2.imwrite(f"debug_color_{color_name}.jpg", frame_with_faces)
                print(f"  💾 Saved color frame: debug_color_{color_name}.jpg")
            else:
                print(f"  ❌ Color {color_name}: No faces found")
            
            detector.release()
        
        # Test image enhancement
        print(f"\n--- Testing image enhancement ---")
        
        # Apply different enhancements
        enhanced_frames = {
            'original': frame,
            'brightness_+50': cv2.convertScaleAbs(frame, alpha=1.0, beta=50),
            'brightness_-50': cv2.convertScaleAbs(frame, alpha=1.0, beta=-50),
            'contrast_1.5': cv2.convertScaleAbs(frame, alpha=1.5, beta=0),
            'contrast_0.5': cv2.convertScaleAbs(frame, alpha=0.5, beta=0),
            'clahe': cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        }
        
        for enh_name, enh_frame in enhanced_frames.items():
            print(f"\n--- Testing enhancement: {enh_name} ---")
            
            # Convert grayscale back to BGR if needed
            if len(enh_frame.shape) == 2:
                test_frame = cv2.cvtColor(enh_frame, cv2.COLOR_GRAY2BGR)
            else:
                test_frame = enh_frame
            
            detector = BlazeFaceDetector(min_detection_confidence=0.3)
            faces = detector.detect_faces(test_frame)
            
            if faces:
                print(f"  ✅ Enhancement {enh_name}: Found {len(faces)} faces")
                for i, face in enumerate(faces):
                    x, y, w, h, conf = face
                    print(f"    Face {i}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                
                frame_with_faces = detector.draw_faces(test_frame.copy(), faces)
                cv2.imwrite(f"debug_enhancement_{enh_name}.jpg", frame_with_faces)
                print(f"  💾 Saved enhancement frame: debug_enhancement_{enh_name}.jpg")
            else:
                print(f"  ❌ Enhancement {enh_name}: No faces found")
            
            detector.release()
        
        # Cleanup
        camera_manager.release()
        
        print(f"\n✅ Debug testing completed")
        print("Check the generated debug_*.jpg files to see detection results")
        return True
        
    except Exception as e:
        print(f"❌ Error during debug testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("NVR Camera Face Detection Debug")
    print("This script tests different detection parameters and preprocessing")
    
    success = test_detection_parameters()
    
    if success:
        print(f"\n{'='*60}")
        print("DEBUG TESTING COMPLETED")
        print(f"{'='*60}")
        print("Check the generated debug_*.jpg files to analyze detection results")
    else:
        print(f"\n{'='*60}")
        print("DEBUG TESTING FAILED")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
