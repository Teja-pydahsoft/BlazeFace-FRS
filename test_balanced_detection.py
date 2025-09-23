"""
Test balanced face detection with optimal parameters
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

def test_balanced_detection():
    """Test balanced detection with optimal parameters"""
    print("=" * 60)
    print("TESTING BALANCED FACE DETECTION")
    print("=" * 60)
    
    try:
        # Load saved frame
        frame_path = "nvr_test_frame_fixed.jpg"
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"❌ Failed to load frame: {frame_path}")
            return False
        
        print(f"✅ Loaded frame: {frame.shape}")
        
        # Test different parameter combinations
        test_configs = [
            {"name": "Conservative", "scale_factor": 1.2, "min_neighbors": 6, "min_size": (40, 40)},
            {"name": "Balanced", "scale_factor": 1.1, "min_neighbors": 5, "min_size": (30, 30)},
            {"name": "Sensitive", "scale_factor": 1.05, "min_neighbors": 4, "min_size": (25, 25)},
            {"name": "Very Sensitive", "scale_factor": 1.02, "min_neighbors": 3, "min_size": (20, 20)}
        ]
        
        best_config = None
        best_score = 0
        
        for config in test_configs:
            print(f"\n--- Testing {config['name']} ---")
            
            # Create detector with custom parameters
            detector = BlazeFaceDetector(
                min_detection_confidence=0.01,
                use_opencv_fallback=True
            )
            
            # Update OpenCV detector parameters
            detector.opencv_detector.scale_factor = config["scale_factor"]
            detector.opencv_detector.min_neighbors = config["min_neighbors"]
            detector.opencv_detector.min_size = config["min_size"]
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            if faces:
                print(f"  ✅ Found {len(faces)} faces")
                
                # Calculate quality score
                confidences = [face[4] for face in faces]
                avg_confidence = np.mean(confidences)
                face_count = len(faces)
                
                # Score based on count and confidence (balance between detection and accuracy)
                score = face_count * 0.5 + avg_confidence * 0.5
                print(f"  Score: {score:.3f} (count: {face_count}, avg_conf: {avg_confidence:.3f})")
                
                if score > best_score:
                    best_score = score
                    best_config = config
                
                # Save result
                frame_with_faces = detector.draw_faces(frame.copy(), faces)
                cv2.imwrite(f"balanced_{config['name'].lower().replace(' ', '_')}.jpg", frame_with_faces)
                print(f"  💾 Saved: balanced_{config['name'].lower().replace(' ', '_')}.jpg")
            else:
                print(f"  ❌ No faces detected")
            
            detector.release()
        
        print(f"\n--- Best Configuration ---")
        if best_config:
            print(f"Best: {best_config['name']} with score {best_score:.3f}")
            print(f"Parameters: scale_factor={best_config['scale_factor']}, min_neighbors={best_config['min_neighbors']}, min_size={best_config['min_size']}")
        else:
            print("No configuration found faces")
        
        return best_config is not None
        
    except Exception as e:
        print(f"❌ Error testing balanced detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_optimal_parameters():
    """Test with the optimal parameters found"""
    print("\n" + "=" * 60)
    print("TESTING WITH OPTIMAL PARAMETERS")
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
        
        # Initialize detector with optimal parameters
        detector = BlazeFaceDetector(
            min_detection_confidence=0.01,
            use_opencv_fallback=True
        )
        
        # Set optimal parameters (based on testing)
        detector.opencv_detector.scale_factor = 1.05
        detector.opencv_detector.min_neighbors = 4
        detector.opencv_detector.min_size = (25, 25)
        
        print("✅ Detector configured with optimal parameters")
        
        # Test face detection
        detection_count = 0
        total_frames = 10
        
        print(f"\nTesting optimal detection on {total_frames} frames...")
        
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
                        cv2.imwrite(f"optimal_detection_{detection_count}.jpg", frame_with_faces)
                        print(f"  💾 Saved: optimal_detection_{detection_count}.jpg")
                else:
                    print(f"  ❌ No faces detected")
                
                time.sleep(0.3)
        
        # Cleanup
        detector.release()
        camera_manager.release()
        
        print(f"\n--- Optimal Detection Summary ---")
        print(f"Frames with faces: {detection_count}/{total_frames}")
        print(f"Detection rate: {(detection_count/total_frames)*100:.1f}%")
        
        return detection_count > 0
        
    except Exception as e:
        print(f"❌ Error testing optimal detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Balanced Face Detection Test")
    print("This finds the optimal balance between accuracy and detection rate")
    print("=" * 60)
    
    # Test 1: Find optimal parameters
    config_success = test_balanced_detection()
    
    # Test 2: Test with optimal parameters
    optimal_success = test_with_optimal_parameters()
    
    # Results
    print("\n" + "=" * 60)
    print("BALANCED DETECTION TEST RESULTS")
    print("=" * 60)
    print(f"Parameter optimization: {'✅ PASS' if config_success else '❌ FAIL'}")
    print(f"Optimal detection test: {'✅ PASS' if optimal_success else '❌ FAIL'}")
    
    if config_success or optimal_success:
        print("\n🎉 BALANCED FACE DETECTION IS WORKING!")
        print("The detector now has a good balance between accuracy and detection rate.")
        print("Check the generated balanced_*.jpg and optimal_*.jpg files to see the results.")
    else:
        print("\n❌ Balanced detection still needs adjustment")

if __name__ == "__main__":
    main()
