"""
Test script for NVR camera face detection
Tests the improved face detection pipeline for NVR cameras
"""

import sys
import os
import cv2
import time
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.camera_utils import CameraManager
from app.core.blazeface_detector import BlazeFaceDetector
from app.core.dual_pipeline import DualPipeline

def test_nvr_face_detection():
    """Test face detection with NVR camera"""
    print("=" * 60)
    print("TESTING NVR CAMERA FACE DETECTION")
    print("=" * 60)
    
    # NVR camera configuration
    nvr_url = "rtsp://admin:nvr@pydah@192.168.3.235:554/stream1"
    
    try:
        # Initialize camera manager
        print(f"Initializing NVR camera: {nvr_url}")
        camera_manager = CameraManager(nvr_url)
        
        if not camera_manager.is_initialized:
            print("❌ Failed to initialize NVR camera")
            return False
        
        print("✅ NVR camera initialized successfully")
        
        # Initialize face detector
        print("Initializing face detector...")
        face_detector = BlazeFaceDetector(min_detection_confidence=0.3)
        print("✅ Face detector initialized")
        
        # Test direct face detection
        print("\n--- Testing Direct Face Detection ---")
        frame_count = 0
        detection_count = 0
        
        for i in range(10):  # Test 10 frames
            ret, frame = camera_manager.get_frame()
            if ret and frame is not None:
                frame_count += 1
                print(f"Frame {frame_count}: Processing...")
                
                # Detect faces
                faces = face_detector.detect_faces(frame)
                
                if faces:
                    detection_count += 1
                    print(f"  ✅ Found {len(faces)} faces")
                    for j, face in enumerate(faces):
                        x, y, w, h, conf = face
                        print(f"    Face {j}: bbox=({x},{y},{w},{h}), confidence={conf:.3f}")
                    
                    # Draw faces on frame
                    frame_with_faces = face_detector.draw_faces(frame, faces)
                    
                    # Save sample frame with detections
                    if frame_count == 1:
                        cv2.imwrite("nvr_face_detection_sample.jpg", frame_with_faces)
                        print("  💾 Saved sample frame: nvr_face_detection_sample.jpg")
                else:
                    print(f"  ❌ No faces detected")
                
                time.sleep(0.5)  # Wait between frames
            else:
                print(f"  ❌ Failed to capture frame {i+1}")
        
        print(f"\n--- Direct Detection Results ---")
        print(f"Frames processed: {frame_count}")
        print(f"Frames with faces: {detection_count}")
        print(f"Detection rate: {detection_count/frame_count*100:.1f}%" if frame_count > 0 else "No frames processed")
        
        # Test pipeline-based detection
        print(f"\n--- Testing Pipeline-Based Detection ---")
        config = {
            'camera_type': 'stream',
            'detection_confidence': 0.3,
            'recognition_confidence': 0.85
        }
        
        pipeline = DualPipeline(config)
        pipeline.start_pipeline()
        
        pipeline_detection_count = 0
        for i in range(5):  # Test 5 frames with pipeline
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
                
                time.sleep(0.5)
        
        pipeline.stop_pipeline()
        
        print(f"\n--- Pipeline Detection Results ---")
        print(f"Pipeline detections: {pipeline_detection_count}/5 frames")
        
        # Cleanup
        camera_manager.release()
        face_detector.release()
        
        print(f"\n✅ NVR face detection test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error during NVR face detection test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("NVR Camera Face Detection Test")
    print("This script tests the improved face detection for NVR cameras")
    
    success = test_nvr_face_detection()
    
    if success:
        print(f"\n{'='*60}")
        print("TEST COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print("The NVR camera face detection should now work properly.")
        print("Check the main application to see face detection overlays.")
    else:
        print(f"\n{'='*60}")
        print("TEST FAILED")
        print(f"{'='*60}")
        print("There may be issues with the NVR camera connection or face detection.")

if __name__ == "__main__":
    main()
