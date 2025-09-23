"""
Test OpenCV face detection with NVR camera
This will be our working solution while we debug MediaPipe
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenCVFaceDetector:
    """OpenCV-based face detector as MediaPipe alternative"""
    
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load OpenCV face cascade")
        
        logger.info("OpenCV face detector initialized")
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            
            # Convert to our format (x, y, width, height, confidence)
            face_list = []
            for (x, y, w, h) in faces:
                # OpenCV doesn't provide confidence, so we'll use a default
                confidence = 0.8  # Default confidence for OpenCV
                face_list.append((x, y, w, h, confidence))
            
            return face_list
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def draw_faces(self, frame, faces):
        """Draw face bounding boxes on frame"""
        try:
            for (x, y, w, h, confidence) in faces:
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw confidence score
                label = f"Face: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing faces: {e}")
            return frame

def test_opencv_with_nvr():
    """Test OpenCV face detection with NVR camera"""
    print("=" * 60)
    print("TESTING OPENCV FACE DETECTION WITH NVR CAMERA")
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
        
        # Initialize face detector
        face_detector = OpenCVFaceDetector()
        
        # Test face detection
        detection_count = 0
        total_frames = 10
        
        print(f"\nTesting face detection on {total_frames} frames...")
        
        for i in range(total_frames):
            ret, frame = camera_manager.get_frame()
            if ret and frame is not None:
                print(f"Frame {i+1}: Processing {frame.shape}...")
                
                # Detect faces
                faces = face_detector.detect_faces(frame)
                
                if faces:
                    detection_count += 1
                    print(f"  ✅ Found {len(faces)} faces")
                    
                    for j, face in enumerate(faces):
                        x, y, w, h, conf = face
                        print(f"    Face {j}: bbox=({x},{y},{w},{h}), confidence={conf:.2f}")
                    
                    # Save first few detections
                    if detection_count <= 3:
                        frame_with_faces = face_detector.draw_faces(frame.copy(), faces)
                        cv2.imwrite(f"opencv_nvr_detection_{detection_count}.jpg", frame_with_faces)
                        print(f"  💾 Saved: opencv_nvr_detection_{detection_count}.jpg")
                else:
                    print(f"  ❌ No faces detected")
                
                time.sleep(0.3)
        
        # Cleanup
        camera_manager.release()
        
        print(f"\n--- Detection Summary ---")
        print(f"Frames with faces: {detection_count}/{total_frames}")
        print(f"Detection rate: {(detection_count/total_frames)*100:.1f}%")
        
        return detection_count > 0
        
    except Exception as e:
        print(f"❌ Error testing OpenCV with NVR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opencv_with_saved_frame():
    """Test OpenCV with the saved NVR frame"""
    print("\n" + "=" * 60)
    print("TESTING OPENCV WITH SAVED NVR FRAME")
    print("=" * 60)
    
    try:
        # Load saved frame
        frame_path = "nvr_test_frame_fixed.jpg"
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"❌ Failed to load frame: {frame_path}")
            return False
        
        print(f"✅ Loaded frame: {frame.shape}")
        
        # Initialize face detector
        face_detector = OpenCVFaceDetector()
        
        # Test different parameters
        test_params = [
            (1.1, 5, (30, 30), "Default"),
            (1.05, 3, (20, 20), "Sensitive"),
            (1.2, 7, (50, 50), "Conservative"),
            (1.1, 3, (40, 40), "Balanced")
        ]
        
        best_detections = 0
        best_params = None
        
        for scale_factor, min_neighbors, min_size, name in test_params:
            print(f"\nTesting {name} parameters...")
            
            # Create detector with these parameters
            detector = OpenCVFaceDetector(scale_factor, min_neighbors, min_size)
            faces = detector.detect_faces(frame)
            
            print(f"  {name}: {len(faces)} faces detected")
            
            if len(faces) > best_detections:
                best_detections = len(faces)
                best_params = (scale_factor, min_neighbors, min_size, name)
                
                # Save result
                if faces:
                    frame_with_faces = detector.draw_faces(frame.copy(), faces)
                    cv2.imwrite(f"opencv_best_{name.lower()}.jpg", frame_with_faces)
                    print(f"  💾 Saved: opencv_best_{name.lower()}.jpg")
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best detections: {best_detections}")
        
        return best_detections > 0
        
    except Exception as e:
        print(f"❌ Error testing with saved frame: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("OpenCV Face Detection Test for NVR Camera")
    print("This provides a working alternative to MediaPipe")
    print("=" * 60)
    
    # Test 1: With saved frame
    saved_success = test_opencv_with_saved_frame()
    
    # Test 2: With live NVR camera
    live_success = test_opencv_with_nvr()
    
    # Results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Saved frame test: {'✅ PASS' if saved_success else '❌ FAIL'}")
    print(f"Live NVR test: {'✅ PASS' if live_success else '❌ FAIL'}")
    
    if saved_success or live_success:
        print("\n🎉 OPENCV FACE DETECTION IS WORKING!")
        print("This can be used as a replacement for MediaPipe in the main application.")
        print("Check the generated opencv_*.jpg files to see the results.")
    else:
        print("\n❌ OpenCV face detection also failed")
        print("There may be a fundamental issue with the NVR stream or face detection setup.")

if __name__ == "__main__":
    main()
