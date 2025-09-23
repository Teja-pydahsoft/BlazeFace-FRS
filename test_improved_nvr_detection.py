"""
Improved NVR camera face detection test
Uses enhanced detection algorithms and better stream handling
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
from app.core.dual_pipeline import DualPipeline

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_improved_nvr_detection():
    """Test improved NVR camera face detection"""
    print("=" * 60)
    print("TESTING IMPROVED NVR CAMERA FACE DETECTION")
    print("=" * 60)
    
    # NVR camera configuration
    nvr_url = "rtsp://admin:nvr@pydah@192.168.3.235:554/stream1"
    
    try:
        # Initialize camera with improved settings
        print(f"Initializing NVR camera: {nvr_url}")
        camera_manager = CameraManager(nvr_url, width=1280, height=720)  # Higher resolution
        
        if not camera_manager.is_initialized:
            print("❌ Failed to initialize NVR camera")
            return False
        
        print("✅ NVR camera initialized successfully")
        
        # Get camera info
        camera_info = camera_manager.get_camera_info()
        print(f"Camera info: {camera_info}")
        
        # Test multiple frames to ensure stable detection
        print(f"\n--- Testing Multiple Frames ---")
        
        # Initialize detector with very low confidence threshold
        face_detector = BlazeFaceDetector(min_detection_confidence=0.1)
        
        total_frames = 0
        frames_with_faces = 0
        total_faces = 0
        
        for i in range(20):  # Test 20 frames
            ret, frame = camera_manager.get_frame()
            if ret and frame is not None:
                total_frames += 1
                print(f"Frame {total_frames}: Processing...")
                
                # Detect faces with improved algorithm
                faces = face_detector.detect_faces(frame)
                
                if faces:
                    frames_with_faces += 1
                    total_faces += len(faces)
                    print(f"  ✅ Found {len(faces)} faces")
                    for j, face in enumerate(faces):
                        x, y, w, h, conf = face
                        print(f"    Face {j}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                    
                    # Draw and save sample frames
                    if frames_with_faces <= 3:  # Save first 3 frames with faces
                        frame_with_faces = face_detector.draw_faces(frame.copy(), faces)
                        cv2.imwrite(f"improved_nvr_detection_{frames_with_faces}.jpg", frame_with_faces)
                        print(f"  💾 Saved detection frame: improved_nvr_detection_{frames_with_faces}.jpg")
                else:
                    print(f"  ❌ No faces detected")
                
                time.sleep(0.5)  # Wait between frames
            else:
                print(f"  ❌ Failed to capture frame {i+1}")
        
        print(f"\n--- Detection Results ---")
        print(f"Total frames processed: {total_frames}")
        print(f"Frames with faces: {frames_with_faces}")
        print(f"Total faces detected: {total_faces}")
        print(f"Detection rate: {frames_with_faces/total_frames*100:.1f}%" if total_frames > 0 else "No frames processed")
        print(f"Average faces per frame: {total_faces/total_frames:.2f}" if total_frames > 0 else "No frames processed")
        
        # Test with different confidence thresholds
        print(f"\n--- Testing Different Confidence Thresholds ---")
        
        confidence_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        for conf_thresh in confidence_thresholds:
            print(f"\n--- Testing confidence threshold: {conf_thresh} ---")
            
            detector = BlazeFaceDetector(min_detection_confidence=conf_thresh)
            
            # Test on a few frames
            detection_count = 0
            for i in range(5):
                ret, frame = camera_manager.get_frame()
                if ret and frame is not None:
                    faces = detector.detect_faces(frame)
                    if faces:
                        detection_count += 1
                        print(f"  Frame {i+1}: Found {len(faces)} faces")
                    else:
                        print(f"  Frame {i+1}: No faces")
                time.sleep(0.2)
            
            print(f"  Detection rate with conf={conf_thresh}: {detection_count}/5 frames")
            detector.release()
        
        # Test with pipeline
        print(f"\n--- Testing Pipeline-Based Detection ---")
        
        config = {
            'camera_type': 'stream',
            'detection_confidence': 0.1,  # Very low threshold
            'recognition_confidence': 0.85
        }
        
        pipeline = DualPipeline(config)
        pipeline.start_pipeline()
        
        pipeline_detection_count = 0
        for i in range(10):
            ret, frame = camera_manager.get_frame()
            if ret and frame is not None:
                print(f"Pipeline Frame {i+1}: Processing...")
                
                results = pipeline.process_frame(frame)
                faces = results.get('faces', [])
                
                if faces:
                    pipeline_detection_count += 1
                    print(f"  ✅ Pipeline found {len(faces)} faces")
                    for j, face in enumerate(faces):
                        x, y, w, h, conf = face
                        print(f"    Face {j}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                else:
                    print(f"  ❌ Pipeline found no faces")
                
                time.sleep(0.3)
        
        pipeline.stop_pipeline()
        
        print(f"\n--- Pipeline Detection Results ---")
        print(f"Pipeline detections: {pipeline_detection_count}/10 frames")
        
        # Cleanup
        camera_manager.release()
        face_detector.release()
        
        print(f"\n✅ Improved NVR face detection test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error during improved NVR face detection test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Improved NVR Camera Face Detection Test")
    print("This script tests the enhanced face detection for NVR cameras")
    
    success = test_improved_nvr_detection()
    
    if success:
        print(f"\n{'='*60}")
        print("IMPROVED NVR DETECTION TEST COMPLETED")
        print(f"{'='*60}")
        print("Check the generated improved_nvr_detection_*.jpg files to see detection results")
    else:
        print(f"\n{'='*60}")
        print("IMPROVED NVR DETECTION TEST FAILED")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
