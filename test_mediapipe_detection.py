"""
Test MediaPipe face detection directly
"""

import cv2
import numpy as np
import mediapipe as mp
import sys
import os

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.camera_utils import CameraManager

def test_mediapipe_direct():
    """Test MediaPipe face detection directly"""
    print("=" * 60)
    print("TESTING MEDIAPIPE FACE DETECTION DIRECTLY")
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
        
        # Save original frame
        cv2.imwrite("mediapipe_test_original.jpg", frame)
        print("💾 Saved original frame: mediapipe_test_original.jpg")
        
        # Initialize MediaPipe face detection directly
        print("\n--- Testing MediaPipe Face Detection Directly ---")
        
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        
        # Test different confidence thresholds
        for conf_thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print(f"\n--- Testing confidence threshold: {conf_thresh} ---")
            
            with mp_face_detection.FaceDetection(
                model_selection=0,  # Short-range model
                min_detection_confidence=conf_thresh
            ) as face_detection:
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame
                results = face_detection.process(rgb_frame)
                
                if results.detections:
                    print(f"  ✅ Found {len(results.detections)} faces with confidence >= {conf_thresh}")
                    
                    # Draw detections
                    annotated_frame = frame.copy()
                    mp_drawing.draw_detection(annotated_frame, results.detections[0])
                    
                    # Save annotated frame
                    cv2.imwrite(f"mediapipe_test_conf_{conf_thresh}.jpg", annotated_frame)
                    print(f"  💾 Saved annotated frame: mediapipe_test_conf_{conf_thresh}.jpg")
                    
                    # Print detection details
                    for i, detection in enumerate(results.detections):
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        confidence = detection.score[0]
                        
                        print(f"    Face {i}: bbox=({x},{y},{width},{height}), confidence={confidence:.3f}")
                else:
                    print(f"  ❌ No faces found with confidence >= {conf_thresh}")
        
        # Test different model selections
        print(f"\n--- Testing different model selections ---")
        
        for model_sel in [0, 1]:
            print(f"\n--- Testing model selection: {model_sel} ---")
            
            with mp_face_detection.FaceDetection(
                model_selection=model_sel,
                min_detection_confidence=0.3
            ) as face_detection:
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)
                
                if results.detections:
                    print(f"  ✅ Model {model_sel}: Found {len(results.detections)} faces")
                    
                    # Draw detections
                    annotated_frame = frame.copy()
                    for detection in results.detections:
                        mp_drawing.draw_detection(annotated_frame, detection)
                    
                    cv2.imwrite(f"mediapipe_test_model_{model_sel}.jpg", annotated_frame)
                    print(f"  💾 Saved model frame: mediapipe_test_model_{model_sel}.jpg")
                    
                    # Print detection details
                    for i, detection in enumerate(results.detections):
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        confidence = detection.score[0]
                        
                        print(f"    Face {i}: bbox=({x},{y},{width},{height}), confidence={confidence:.3f}")
                else:
                    print(f"  ❌ Model {model_sel}: No faces found")
        
        # Test with different image sizes
        print(f"\n--- Testing different image sizes ---")
        
        sizes = [(320, 240), (640, 480), (1280, 720)]
        
        for width, height in sizes:
            print(f"\n--- Testing size: {width}x{height} ---")
            
            # Resize frame
            resized_frame = cv2.resize(frame, (width, height))
            
            with mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.3
            ) as face_detection:
                
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)
                
                if results.detections:
                    print(f"  ✅ Size {width}x{height}: Found {len(results.detections)} faces")
                    
                    # Draw detections
                    annotated_frame = resized_frame.copy()
                    for detection in results.detections:
                        mp_drawing.draw_detection(annotated_frame, detection)
                    
                    cv2.imwrite(f"mediapipe_test_size_{width}x{height}.jpg", annotated_frame)
                    print(f"  💾 Saved size frame: mediapipe_test_size_{width}x{height}.jpg")
                    
                    # Print detection details
                    for i, detection in enumerate(results.detections):
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = resized_frame.shape
                        
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        confidence = detection.score[0]
                        
                        print(f"    Face {i}: bbox=({x},{y},{width},{height}), confidence={confidence:.3f}")
                else:
                    print(f"  ❌ Size {width}x{height}: No faces found")
        
        # Cleanup
        camera_manager.release()
        
        print(f"\n✅ MediaPipe direct testing completed")
        print("Check the generated mediapipe_test_*.jpg files to see detection results")
        return True
        
    except Exception as e:
        print(f"❌ Error during MediaPipe testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("MediaPipe Face Detection Direct Test")
    print("This script tests MediaPipe face detection directly")
    
    success = test_mediapipe_direct()
    
    if success:
        print(f"\n{'='*60}")
        print("MEDIAPIPE TESTING COMPLETED")
        print(f"{'='*60}")
        print("Check the generated mediapipe_test_*.jpg files to analyze detection results")
    else:
        print(f"\n{'='*60}")
        print("MEDIAPIPE TESTING FAILED")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
