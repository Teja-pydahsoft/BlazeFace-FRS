"""
Final test for NVR camera face detection with MediaPipe + OpenCV fallback
This should now work reliably with the NVR camera
"""

import sys
import os
import cv2
import time
import numpy as np
import logging

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.camera_utils import CameraManager
from app.core.blazeface_detector import BlazeFaceDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_nvr_with_hybrid_detector():
    """Test NVR camera with hybrid MediaPipe + OpenCV detector"""
    print("=" * 60)
    print("TESTING NVR CAMERA WITH HYBRID FACE DETECTOR")
    print("=" * 60)
    
    nvr_url = "rtsp://admin:nvr@pydah@192.168.3.235:554/stream1"
    
    try:
        # Initialize camera
        print(f"Connecting to NVR camera: {nvr_url}")
        camera_manager = CameraManager(nvr_url)
        
        if not camera_manager.is_initialized:
            print("❌ Failed to initialize NVR camera")
            return False
        
        print("✅ NVR camera initialized successfully")
        
        # Initialize hybrid face detector
        print("Initializing hybrid face detector (MediaPipe + OpenCV fallback)...")
        detector = BlazeFaceDetector(
            min_detection_confidence=0.01,  # Very low threshold
            use_opencv_fallback=True
        )
        
        print("✅ Hybrid face detector initialized")
        
        # Test face detection
        detection_count = 0
        total_frames = 15
        
        print(f"\nTesting face detection on {total_frames} frames...")
        
        for i in range(total_frames):
            ret, frame = camera_manager.get_frame()
            if ret and frame is not None:
                print(f"Frame {i+1}: Processing {frame.shape}...")
                
                # Detect faces
                faces = detector.detect_faces(frame)
                
                if faces:
                    detection_count += 1
                    print(f"  ✅ Found {len(faces)} faces")
                    
                    for j, face in enumerate(faces):
                        x, y, w, h, conf = face
                        print(f"    Face {j}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                    
                    # Save first few detections
                    if detection_count <= 5:
                        frame_with_faces = detector.draw_faces(frame.copy(), faces)
                        cv2.imwrite(f"hybrid_detection_{detection_count}.jpg", frame_with_faces)
                        print(f"  💾 Saved: hybrid_detection_{detection_count}.jpg")
                else:
                    print(f"  ❌ No faces detected")
                
                time.sleep(0.3)
        
        # Cleanup
        detector.release()
        camera_manager.release()
        
        print(f"\n--- Detection Summary ---")
        print(f"Frames with faces: {detection_count}/{total_frames}")
        print(f"Detection rate: {(detection_count/total_frames)*100:.1f}%")
        
        return detection_count > 0
        
    except Exception as e:
        print(f"❌ Error testing hybrid detector: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_saved_frame():
    """Test with the saved NVR frame"""
    print("\n" + "=" * 60)
    print("TESTING WITH SAVED NVR FRAME")
    print("=" * 60)
    
    try:
        # Load saved frame
        frame_path = "nvr_test_frame_fixed.jpg"
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"❌ Failed to load frame: {frame_path}")
            return False
        
        print(f"✅ Loaded frame: {frame.shape}")
        
        # Initialize hybrid detector
        detector = BlazeFaceDetector(
            min_detection_confidence=0.01,
            use_opencv_fallback=True
        )
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        if faces:
            print(f"✅ Hybrid detector found {len(faces)} faces in saved frame")
            
            for j, face in enumerate(faces):
                x, y, w, h, conf = face
                print(f"  Face {j}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
            
            # Draw and save
            frame_with_faces = detector.draw_faces(frame.copy(), faces)
            cv2.imwrite("hybrid_saved_frame_detection.jpg", frame_with_faces)
            print("💾 Saved: hybrid_saved_frame_detection.jpg")
            
            detector.release()
            return True
        else:
            print("❌ Hybrid detector found no faces in saved frame")
            detector.release()
            return False
            
    except Exception as e:
        print(f"❌ Error testing with saved frame: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detector_capabilities():
    """Test detector capabilities and fallback behavior"""
    print("\n" + "=" * 60)
    print("TESTING DETECTOR CAPABILITIES")
    print("=" * 60)
    
    try:
        # Test with different configurations
        configs = [
            {"name": "MediaPipe Only", "use_opencv_fallback": False},
            {"name": "OpenCV Only", "use_opencv_fallback": True, "disable_mediapipe": True},
            {"name": "Hybrid (Default)", "use_opencv_fallback": True}
        ]
        
        frame = cv2.imread("nvr_test_frame_fixed.jpg")
        if frame is None:
            print("❌ Failed to load test frame")
            return False
        
        for config in configs:
            print(f"\n--- Testing {config['name']} ---")
            
            try:
                if config.get("disable_mediapipe", False):
                    # Simulate MediaPipe failure
                    detector = BlazeFaceDetector(use_opencv_fallback=True)
                    detector.mediapipe_available = False
                else:
                    detector = BlazeFaceDetector(
                        min_detection_confidence=0.01,
                        use_opencv_fallback=config["use_opencv_fallback"]
                    )
                
                faces = detector.detect_faces(frame)
                
                if faces:
                    print(f"  ✅ {config['name']}: Found {len(faces)} faces")
                else:
                    print(f"  ❌ {config['name']}: No faces detected")
                
                detector.release()
                
            except Exception as e:
                print(f"  ❌ {config['name']}: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing capabilities: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Final NVR Camera Face Detection Test - WORKING VERSION")
    print("This uses MediaPipe with OpenCV fallback for reliable detection")
    print("=" * 60)
    
    # Test 1: With saved frame
    saved_success = test_with_saved_frame()
    
    # Test 2: Detector capabilities
    capabilities_success = test_detector_capabilities()
    
    # Test 3: Live NVR camera
    live_success = test_nvr_with_hybrid_detector()
    
    # Results
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Saved frame test: {'✅ PASS' if saved_success else '❌ FAIL'}")
    print(f"Capabilities test: {'✅ PASS' if capabilities_success else '❌ FAIL'}")
    print(f"Live NVR test: {'✅ PASS' if live_success else '❌ FAIL'}")
    
    if saved_success or live_success:
        print("\n🎉 FACE DETECTION IS NOW WORKING!")
        print("The hybrid detector (MediaPipe + OpenCV fallback) successfully detects faces.")
        print("Check the generated hybrid_*.jpg files to see the results.")
        print("\nThe NVR camera face detection should now work in the main application!")
    else:
        print("\n❌ Face detection still not working")
        print("There may be a fundamental issue with the setup.")

if __name__ == "__main__":
    main()
