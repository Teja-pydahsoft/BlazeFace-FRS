"""
Test improved face detection with better filtering and accuracy
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

def test_improved_detection_with_saved_frame():
    """Test improved detection with the saved NVR frame"""
    print("=" * 60)
    print("TESTING IMPROVED FACE DETECTION WITH SAVED FRAME")
    print("=" * 60)
    
    try:
        # Load saved frame
        frame_path = "nvr_test_frame_fixed.jpg"
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"❌ Failed to load frame: {frame_path}")
            return False
        
        print(f"✅ Loaded frame: {frame.shape}")
        
        # Initialize improved detector
        detector = BlazeFaceDetector(
            min_detection_confidence=0.01,
            use_opencv_fallback=True
        )
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        if faces:
            print(f"✅ Improved detector found {len(faces)} faces")
            
            for j, face in enumerate(faces):
                x, y, w, h, conf = face
                print(f"  Face {j}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
            
            # Draw and save
            frame_with_faces = detector.draw_faces(frame.copy(), faces)
            cv2.imwrite("improved_detection_saved.jpg", frame_with_faces)
            print("💾 Saved: improved_detection_saved.jpg")
            
            detector.release()
            return True
        else:
            print("❌ Improved detector found no faces")
            detector.release()
            return False
            
    except Exception as e:
        print(f"❌ Error testing improved detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_improved_detection_with_live_camera():
    """Test improved detection with live NVR camera"""
    print("\n" + "=" * 60)
    print("TESTING IMPROVED FACE DETECTION WITH LIVE NVR CAMERA")
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
        
        # Initialize improved detector
        detector = BlazeFaceDetector(
            min_detection_confidence=0.01,
            use_opencv_fallback=True
        )
        
        # Test face detection
        detection_count = 0
        total_frames = 10
        
        print(f"\nTesting improved face detection on {total_frames} frames...")
        
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
                    if detection_count <= 3:
                        frame_with_faces = detector.draw_faces(frame.copy(), faces)
                        cv2.imwrite(f"improved_detection_live_{detection_count}.jpg", frame_with_faces)
                        print(f"  💾 Saved: improved_detection_live_{detection_count}.jpg")
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
        print(f"❌ Error testing improved detection with live camera: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detection_quality():
    """Test detection quality by analyzing the results"""
    print("\n" + "=" * 60)
    print("TESTING DETECTION QUALITY ANALYSIS")
    print("=" * 60)
    
    try:
        # Load saved frame
        frame = cv2.imread("nvr_test_frame_fixed.jpg")
        if frame is None:
            print("❌ Failed to load test frame")
            return False
        
        # Test with different confidence thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        detector = BlazeFaceDetector(
            min_detection_confidence=0.01,
            use_opencv_fallback=True
        )
        
        faces = detector.detect_faces(frame)
        
        if faces:
            print(f"Raw detections: {len(faces)} faces")
            
            # Analyze face sizes
            sizes = [w * h for x, y, w, h, conf in faces]
            print(f"Face sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.0f}")
            
            # Analyze confidence scores
            confidences = [conf for x, y, w, h, conf in faces]
            print(f"Confidence scores: min={min(confidences):.3f}, max={max(confidences):.3f}, avg={np.mean(confidences):.3f}")
            
            # Analyze aspect ratios
            aspect_ratios = [w/h for x, y, w, h, conf in faces]
            print(f"Aspect ratios: min={min(aspect_ratios):.2f}, max={max(aspect_ratios):.2f}, avg={np.mean(aspect_ratios):.2f}")
            
            # Filter by confidence
            for threshold in thresholds:
                filtered_faces = [face for face in faces if face[4] >= threshold]
                print(f"Confidence >= {threshold}: {len(filtered_faces)} faces")
            
            detector.release()
            return True
        else:
            print("❌ No faces detected for quality analysis")
            detector.release()
            return False
            
    except Exception as e:
        print(f"❌ Error in quality analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Improved Face Detection Test")
    print("This tests the enhanced face detection with better filtering")
    print("=" * 60)
    
    # Test 1: With saved frame
    saved_success = test_improved_detection_with_saved_frame()
    
    # Test 2: Quality analysis
    quality_success = test_detection_quality()
    
    # Test 3: Live camera
    live_success = test_improved_detection_with_live_camera()
    
    # Results
    print("\n" + "=" * 60)
    print("IMPROVED DETECTION TEST RESULTS")
    print("=" * 60)
    print(f"Saved frame test: {'✅ PASS' if saved_success else '❌ FAIL'}")
    print(f"Quality analysis: {'✅ PASS' if quality_success else '❌ FAIL'}")
    print(f"Live camera test: {'✅ PASS' if live_success else '❌ FAIL'}")
    
    if saved_success or live_success:
        print("\n🎉 IMPROVED FACE DETECTION IS WORKING!")
        print("The enhanced detector should now have fewer false positives.")
        print("Check the generated improved_*.jpg files to see the results.")
    else:
        print("\n❌ Improved detection still not working properly")

if __name__ == "__main__":
    main()
