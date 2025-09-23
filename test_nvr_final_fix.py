"""
Final comprehensive test for NVR camera face detection with all fixes
Tests all improvements and provides detailed diagnostics
"""

import sys
import os
import cv2
import time
import numpy as np
import logging
import json

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.camera_utils import CameraManager
from app.core.blazeface_detector import BlazeFaceDetector
from app.core.dual_pipeline import DualPipeline

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.json"""
    try:
        with open('app/config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def test_camera_connection():
    """Test NVR camera connection and basic functionality"""
    print("=" * 60)
    print("TESTING NVR CAMERA CONNECTION")
    print("=" * 60)
    
    nvr_url = "rtsp://admin:nvr@pydah@192.168.3.235:554/stream1"
    
    try:
        # Test basic camera connection - don't force resolution for NVR
        print(f"Testing camera connection: {nvr_url}")
        camera_manager = CameraManager(nvr_url)  # Let it use original resolution
        
        if not camera_manager.is_initialized:
            print("❌ Failed to initialize NVR camera")
            return False, None
        
        print("✅ NVR camera initialized successfully")
        
        # Get camera information
        camera_info = camera_manager.get_camera_info()
        print(f"Camera info: {camera_info}")
        
        # Test frame capture
        print("\n--- Testing Frame Capture ---")
        for i in range(5):
            ret, frame = camera_manager.get_frame()
            if ret and frame is not None:
                print(f"  Frame {i+1}: ✅ Captured {frame.shape}")
                if i == 0:  # Save first frame
                    cv2.imwrite("nvr_test_frame_fixed.jpg", frame)
                    print(f"  💾 Saved test frame: nvr_test_frame_fixed.jpg")
            else:
                print(f"  Frame {i+1}: ❌ Failed to capture")
            time.sleep(0.5)
        
        return True, camera_manager
        
    except Exception as e:
        print(f"❌ Error testing camera connection: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_improved_face_detection(camera_manager):
    """Test improved face detection with NVR-specific optimizations"""
    print("\n" + "=" * 60)
    print("TESTING IMPROVED FACE DETECTION")
    print("=" * 60)
    
    try:
        # Test with very low confidence threshold for NVR
        print("--- Testing with NVR-Optimized Detection ---")
        detector = BlazeFaceDetector(min_detection_confidence=0.01)  # Very low threshold
        
        detection_results = []
        
        for i in range(10):
            ret, frame = camera_manager.get_frame()
            if ret and frame is not None:
                print(f"Frame {i+1}: Processing {frame.shape}...")
                
                faces = detector.detect_faces(frame)
                
                if faces:
                    print(f"  ✅ Found {len(faces)} faces")
                    for j, face in enumerate(faces):
                        x, y, w, h, conf = face
                        print(f"    Face {j}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                    
                    detection_results.append((i+1, len(faces), faces))
                    
                    # Save first few frames with detections
                    if len(detection_results) <= 3:
                        frame_with_faces = detector.draw_faces(frame.copy(), faces)
                        cv2.imwrite(f"nvr_detection_{len(detection_results)}.jpg", frame_with_faces)
                        print(f"  💾 Saved detection frame: nvr_detection_{len(detection_results)}.jpg")
                else:
                    print(f"  ❌ No faces detected")
                
                time.sleep(0.3)
        
        detector.release()
        
        print(f"\n--- Detection Summary ---")
        print(f"Frames with faces: {len(detection_results)}/10")
        total_faces = sum(len(faces) for _, _, faces in detection_results)
        print(f"Total faces detected: {total_faces}")
        
        return len(detection_results) > 0
        
    except Exception as e:
        print(f"❌ Error testing face detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mediapipe_direct_nvr(camera_manager):
    """Test MediaPipe face detection directly with NVR optimizations"""
    print("\n" + "=" * 60)
    print("TESTING MEDIAPIPE DIRECT DETECTION (NVR OPTIMIZED)")
    print("=" * 60)
    
    try:
        import mediapipe as mp
        
        print(f"MediaPipe version: {mp.__version__}")
        
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        
        # Test with very low confidence and different models
        detection_count = 0
        
        for model_selection in [0, 1]:  # Try both short-range and full-range models
            print(f"\n--- Testing Model {model_selection} (0=short-range, 1=full-range) ---")
            
            with mp_face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=0.01  # Very low threshold
            ) as face_detection:
                
                for i in range(5):  # Test fewer frames per model
                    ret, frame = camera_manager.get_frame()
                    if ret and frame is not None:
                        print(f"MediaPipe Frame {i+1}: Processing {frame.shape}...")
                        
                        # Convert BGR to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Try different preprocessing for NVR
                        preprocessing_attempts = [
                            rgb_frame,
                            cv2.resize(rgb_frame, (640, 480)),
                            cv2.resize(rgb_frame, (1280, 720)),
                            cv2.convertScaleAbs(rgb_frame, alpha=1.2, beta=10)
                        ]
                        
                        best_detections = None
                        best_confidence = 0
                        
                        for j, processed_frame in enumerate(preprocessing_attempts):
                            try:
                                results = face_detection.process(processed_frame)
                                
                                if results.detections:
                                    current_confidence = sum(detection.score[0] for detection in results.detections)
                                    if current_confidence > best_confidence:
                                        best_detections = results.detections
                                        best_confidence = current_confidence
                                        
                            except Exception as e:
                                continue
                        
                        if best_detections:
                            detection_count += 1
                            print(f"  ✅ MediaPipe found {len(best_detections)} faces (confidence: {best_confidence:.3f})")
                            
                            # Convert detections to original frame coordinates
                            h, w, _ = frame.shape
                            faces = []
                            
                            for detection in best_detections:
                                bbox = detection.location_data.relative_bounding_box
                                
                                x = int(bbox.xmin * w)
                                y = int(bbox.ymin * h)
                                width = int(bbox.width * w)
                                height = int(bbox.height * h)
                                confidence = detection.score[0]
                                
                                print(f"    Face: bbox=({x},{y},{width},{height}), confidence={confidence:.3f}")
                                faces.append((x, y, width, height, confidence))
                            
                            # Save detection frame
                            if detection_count <= 2:
                                annotated_frame = frame.copy()
                                for (x, y, w, h, conf) in faces:
                                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    cv2.putText(annotated_frame, f"{conf:.2f}", (x, y - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                cv2.imwrite(f"mediapipe_nvr_{detection_count}.jpg", annotated_frame)
                                print(f"  💾 Saved MediaPipe frame: mediapipe_nvr_{detection_count}.jpg")
                        else:
                            print(f"  ❌ MediaPipe found no faces")
                        
                        time.sleep(0.3)
        
        print(f"\n--- MediaPipe Direct Detection Summary ---")
        print(f"MediaPipe detections: {detection_count}/10 frames")
        
        return detection_count > 0
        
    except Exception as e:
        print(f"❌ Error testing MediaPipe direct detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_detection_with_different_sizes(camera_manager):
    """Test face detection with different frame sizes"""
    print("\n" + "=" * 60)
    print("TESTING FACE DETECTION WITH DIFFERENT FRAME SIZES")
    print("=" * 60)
    
    try:
        detector = BlazeFaceDetector(min_detection_confidence=0.01)
        
        # Test different frame sizes
        test_sizes = [
            (640, 480),    # Standard face detection size
            (1280, 720),   # HD
            (1920, 1080),  # Full HD
        ]
        
        detection_count = 0
        
        for size_name, (target_w, target_h) in enumerate(test_sizes):
            print(f"\n--- Testing {target_w}x{target_h} ---")
            
            for i in range(3):  # Test 3 frames per size
                ret, frame = camera_manager.get_frame()
                if ret and frame is not None:
                    # Resize frame to target size
                    resized_frame = cv2.resize(frame, (target_w, target_h))
                    print(f"Frame {i+1}: Processing {resized_frame.shape}...")
                    
                    faces = detector.detect_faces(resized_frame)
                    
                    if faces:
                        detection_count += 1
                        print(f"  ✅ Found {len(faces)} faces in {target_w}x{target_h}")
                        for j, face in enumerate(faces):
                            x, y, w, h, conf = face
                            print(f"    Face {j}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                        
                        # Save detection
                        if detection_count <= 3:
                            frame_with_faces = detector.draw_faces(resized_frame.copy(), faces)
                            cv2.imwrite(f"size_test_{target_w}x{target_h}_{detection_count}.jpg", frame_with_faces)
                            print(f"  💾 Saved: size_test_{target_w}x{target_h}_{detection_count}.jpg")
                    else:
                        print(f"  ❌ No faces detected in {target_w}x{target_h}")
                    
                    time.sleep(0.3)
        
        detector.release()
        
        print(f"\n--- Size Test Summary ---")
        print(f"Total detections across all sizes: {detection_count}")
        
        return detection_count > 0
        
    except Exception as e:
        print(f"❌ Error testing different sizes: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Final NVR Camera Face Detection Test - FIXED VERSION")
    print("This script tests all NVR-specific improvements")
    print("=" * 60)
    
    # Test 1: Camera connection
    camera_success, camera_manager = test_camera_connection()
    if not camera_success:
        print("\n❌ Camera connection failed. Cannot proceed with face detection tests.")
        return
    
    # Test 2: Improved face detection
    detection_success = test_improved_face_detection(camera_manager)
    
    # Test 3: MediaPipe direct detection
    mediapipe_success = test_mediapipe_direct_nvr(camera_manager)
    
    # Test 4: Different frame sizes
    size_success = test_face_detection_with_different_sizes(camera_manager)
    
    # Cleanup
    if camera_manager:
        camera_manager.release()
    
    # Final results
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Camera Connection: {'✅ PASS' if camera_success else '❌ FAIL'}")
    print(f"Improved Detection: {'✅ PASS' if detection_success else '❌ FAIL'}")
    print(f"MediaPipe Direct: {'✅ PASS' if mediapipe_success else '❌ FAIL'}")
    print(f"Different Sizes: {'✅ PASS' if size_success else '❌ FAIL'}")
    
    if detection_success or mediapipe_success or size_success:
        print("\n🎉 FACE DETECTION IS WORKING!")
        print("Check the generated *_detection_*.jpg and mediapipe_nvr_*.jpg files to see the results.")
        print("The NVR camera face detection should now work in the main application.")
    else:
        print("\n❌ FACE DETECTION STILL NOT WORKING")
        print("There may be fundamental issues with the NVR camera stream or MediaPipe installation.")
        print("Consider checking:")
        print("1. NVR camera stream quality and format")
        print("2. MediaPipe installation and version")
        print("3. Network connectivity to the NVR camera")
        print("4. Camera permissions and authentication")
        print("5. Try testing with a different camera or video file")

if __name__ == "__main__":
    main()
