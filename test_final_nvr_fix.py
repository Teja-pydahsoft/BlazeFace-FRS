"""
Final comprehensive test for NVR camera face detection
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
        # Test basic camera connection
        print(f"Testing camera connection: {nvr_url}")
        camera_manager = CameraManager(nvr_url, width=1280, height=720)
        
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
                    cv2.imwrite("nvr_test_frame.jpg", frame)
                    print(f"  💾 Saved test frame: nvr_test_frame.jpg")
            else:
                print(f"  Frame {i+1}: ❌ Failed to capture")
            time.sleep(0.5)
        
        return True, camera_manager
        
    except Exception as e:
        print(f"❌ Error testing camera connection: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_face_detection_improvements(camera_manager):
    """Test improved face detection"""
    print("\n" + "=" * 60)
    print("TESTING IMPROVED FACE DETECTION")
    print("=" * 60)
    
    try:
        # Test with very low confidence threshold
        print("--- Testing with Very Low Confidence Threshold ---")
        detector = BlazeFaceDetector(min_detection_confidence=0.05)
        
        detection_results = []
        
        for i in range(10):
            ret, frame = camera_manager.get_frame()
            if ret and frame is not None:
                print(f"Frame {i+1}: Processing...")
                
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
                        cv2.imwrite(f"improved_detection_{len(detection_results)}.jpg", frame_with_faces)
                        print(f"  💾 Saved detection frame: improved_detection_{len(detection_results)}.jpg")
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

def test_pipeline_detection(camera_manager):
    """Test pipeline-based detection"""
    print("\n" + "=" * 60)
    print("TESTING PIPELINE-BASED DETECTION")
    print("=" * 60)
    
    try:
        # Load configuration
        config = load_config()
        
        # Use NVR-specific settings
        pipeline_config = {
            'camera_type': 'stream',
            'detection_confidence': config.get('nvr_settings', {}).get('detection_confidence', 0.05),
            'recognition_confidence': 0.85
        }
        
        print(f"Pipeline config: {pipeline_config}")
        
        # Initialize pipeline
        pipeline = DualPipeline(pipeline_config)
        pipeline.start_pipeline()
        
        print("✅ Pipeline started")
        
        # Test pipeline detection
        detection_count = 0
        for i in range(10):
            ret, frame = camera_manager.get_frame()
            if ret and frame is not None:
                print(f"Pipeline Frame {i+1}: Processing...")
                
                results = pipeline.process_frame(frame)
                faces = results.get('faces', [])
                
                if faces:
                    detection_count += 1
                    print(f"  ✅ Pipeline found {len(faces)} faces")
                    for j, face in enumerate(faces):
                        x, y, w, h, conf = face
                        print(f"    Face {j}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                    
                    # Save first few pipeline detections
                    if detection_count <= 3:
                        frame_with_faces = detector.draw_faces(frame.copy(), faces)
                        cv2.imwrite(f"pipeline_detection_{detection_count}.jpg", frame_with_faces)
                        print(f"  💾 Saved pipeline frame: pipeline_detection_{detection_count}.jpg")
                else:
                    print(f"  ❌ Pipeline found no faces")
                
                time.sleep(0.3)
        
        pipeline.stop_pipeline()
        
        print(f"\n--- Pipeline Detection Summary ---")
        print(f"Pipeline detections: {detection_count}/10 frames")
        
        return detection_count > 0
        
    except Exception as e:
        print(f"❌ Error testing pipeline detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mediapipe_direct(camera_manager):
    """Test MediaPipe face detection directly"""
    print("\n" + "=" * 60)
    print("TESTING MEDIAPIPE DIRECT DETECTION")
    print("=" * 60)
    
    try:
        import mediapipe as mp
        
        print(f"MediaPipe version: {mp.__version__}")
        
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        
        # Test with very low confidence
        with mp_face_detection.FaceDetection(
            model_selection=0,  # Short-range model
            min_detection_confidence=0.05
        ) as face_detection:
            
            detection_count = 0
            
            for i in range(10):
                ret, frame = camera_manager.get_frame()
                if ret and frame is not None:
                    print(f"MediaPipe Frame {i+1}: Processing...")
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process the frame
                    results = face_detection.process(rgb_frame)
                    
                    if results.detections:
                        detection_count += 1
                        print(f"  ✅ MediaPipe found {len(results.detections)} faces")
                        
                        for j, detection in enumerate(results.detections):
                            bbox = detection.location_data.relative_bounding_box
                            h, w, _ = frame.shape
                            
                            x = int(bbox.xmin * w)
                            y = int(bbox.ymin * h)
                            width = int(bbox.width * w)
                            height = int(bbox.height * h)
                            confidence = detection.score[0]
                            
                            print(f"    Face {j}: bbox=({x},{y},{width},{height}), confidence={confidence:.3f}")
                        
                        # Save first few MediaPipe detections
                        if detection_count <= 3:
                            annotated_frame = frame.copy()
                            for detection in results.detections:
                                mp_drawing.draw_detection(annotated_frame, detection)
                            cv2.imwrite(f"mediapipe_direct_{detection_count}.jpg", annotated_frame)
                            print(f"  💾 Saved MediaPipe frame: mediapipe_direct_{detection_count}.jpg")
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

def main():
    """Main test function"""
    print("Final NVR Camera Face Detection Test")
    print("This script comprehensively tests all face detection improvements")
    print("=" * 60)
    
    # Test 1: Camera connection
    camera_success, camera_manager = test_camera_connection()
    if not camera_success:
        print("\n❌ Camera connection failed. Cannot proceed with face detection tests.")
        return
    
    # Test 2: Improved face detection
    detection_success = test_face_detection_improvements(camera_manager)
    
    # Test 3: Pipeline detection
    pipeline_success = test_pipeline_detection(camera_manager)
    
    # Test 4: MediaPipe direct detection
    mediapipe_success = test_mediapipe_direct(camera_manager)
    
    # Cleanup
    if camera_manager:
        camera_manager.release()
    
    # Final results
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Camera Connection: {'✅ PASS' if camera_success else '❌ FAIL'}")
    print(f"Improved Detection: {'✅ PASS' if detection_success else '❌ FAIL'}")
    print(f"Pipeline Detection: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
    print(f"MediaPipe Direct: {'✅ PASS' if mediapipe_success else '❌ FAIL'}")
    
    if detection_success or pipeline_success or mediapipe_success:
        print("\n🎉 FACE DETECTION IS WORKING!")
        print("Check the generated *_detection_*.jpg files to see the results.")
        print("The NVR camera face detection should now work in the main application.")
    else:
        print("\n❌ FACE DETECTION STILL NOT WORKING")
        print("There may be fundamental issues with the NVR camera stream or MediaPipe installation.")
        print("Consider checking:")
        print("1. NVR camera stream quality and format")
        print("2. MediaPipe installation and version")
        print("3. Network connectivity to the NVR camera")
        print("4. Camera permissions and authentication")

if __name__ == "__main__":
    main()
