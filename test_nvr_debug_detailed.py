"""
Detailed NVR face detection debugging
Tests specific issues with MediaPipe and NVR streams
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

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mediapipe_with_saved_frame():
    """Test MediaPipe with the saved NVR frame"""
    print("=" * 60)
    print("TESTING MEDIAPIPE WITH SAVED NVR FRAME")
    print("=" * 60)
    
    try:
        import mediapipe as mp
        
        # Load the saved frame
        frame_path = "nvr_test_frame_fixed.jpg"
        if not os.path.exists(frame_path):
            print(f"❌ Frame file not found: {frame_path}")
            return False
        
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"❌ Failed to load frame: {frame_path}")
            return False
        
        print(f"✅ Loaded frame: {frame.shape}")
        
        # Test different MediaPipe configurations
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        
        # Test 1: Very low confidence threshold
        print("\n--- Test 1: Very Low Confidence (0.001) ---")
        with mp_face_detection.FaceDetection(
            model_selection=0,  # Short-range model
            min_detection_confidence=0.001  # Extremely low threshold
        ) as face_detection:
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"RGB frame shape: {rgb_frame.shape}")
            
            # Test original frame
            results = face_detection.process(rgb_frame)
            print(f"Original frame detections: {len(results.detections) if results.detections else 0}")
            
            # Test different sizes
            for size in [(640, 480), (1280, 720), (960, 540)]:
                resized = cv2.resize(rgb_frame, size)
                results = face_detection.process(resized)
                print(f"Resized to {size}: {len(results.detections) if results.detections else 0} detections")
        
        # Test 2: Different preprocessing
        print("\n--- Test 2: Different Preprocessing ---")
        with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.01
        ) as face_detection:
            
            # Test different preprocessing approaches
            preprocessing_tests = [
                ("Original", rgb_frame),
                ("Resized 640x480", cv2.resize(rgb_frame, (640, 480))),
                ("Resized 1280x720", cv2.resize(rgb_frame, (1280, 720))),
                ("Enhanced contrast", cv2.convertScaleAbs(rgb_frame, alpha=1.5, beta=20)),
                ("Histogram equalized", cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)), cv2.COLOR_GRAY2RGB)),
                ("Gaussian blur + sharpen", cv2.addWeighted(rgb_frame, 1.5, cv2.GaussianBlur(rgb_frame, (3, 3), 0), -0.5, 0)),
                ("CLAHE", cv2.cvtColor(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)), cv2.COLOR_GRAY2RGB))
            ]
            
            for name, processed_frame in preprocessing_tests:
                try:
                    results = face_detection.process(processed_frame)
                    detections = len(results.detections) if results.detections else 0
                    print(f"{name}: {detections} detections")
                    
                    if detections > 0:
                        print(f"  ✅ Found faces with {name}!")
                        # Save the result
                        annotated_frame = frame.copy()
                        for detection in results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            h, w, _ = frame.shape
                            x = int(bbox.xmin * w)
                            y = int(bbox.ymin * h)
                            width = int(bbox.width * w)
                            height = int(bbox.height * h)
                            confidence = detection.score[0]
                            
                            cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"{confidence:.3f}", (x, y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        cv2.imwrite(f"debug_detection_{name.replace(' ', '_')}.jpg", annotated_frame)
                        print(f"  💾 Saved: debug_detection_{name.replace(' ', '_')}.jpg")
                        
                except Exception as e:
                    print(f"{name}: Error - {e}")
        
        # Test 3: Both models
        print("\n--- Test 3: Both MediaPipe Models ---")
        for model_selection in [0, 1]:
            print(f"\nModel {model_selection} (0=short-range, 1=full-range):")
            with mp_face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=0.01
            ) as face_detection:
                
                # Test with best preprocessing
                best_frame = cv2.resize(rgb_frame, (640, 480))
                results = face_detection.process(best_frame)
                detections = len(results.detections) if results.detections else 0
                print(f"  Detections: {detections}")
                
                if detections > 0:
                    print(f"  ✅ Model {model_selection} found faces!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing MediaPipe: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opencv_face_detection():
    """Test OpenCV's built-in face detection as comparison"""
    print("\n" + "=" * 60)
    print("TESTING OPENCV FACE DETECTION (COMPARISON)")
    print("=" * 60)
    
    try:
        # Load the saved frame
        frame_path = "nvr_test_frame_fixed.jpg"
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"❌ Failed to load frame: {frame_path}")
            return False
        
        # Load OpenCV face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            print("❌ Failed to load OpenCV face cascade")
            return False
        
        print("✅ OpenCV face cascade loaded")
        
        # Test different scales and sizes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for scale_factor in [1.1, 1.2, 1.3]:
            for min_neighbors in [3, 5, 7]:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(30, 30)
                )
                
                print(f"Scale {scale_factor}, Neighbors {min_neighbors}: {len(faces)} faces")
                
                if len(faces) > 0:
                    print(f"  ✅ OpenCV found {len(faces)} faces!")
                    
                    # Draw and save
                    annotated_frame = frame.copy()
                    for (x, y, w, h) in faces:
                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, "OpenCV", (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    cv2.imwrite(f"opencv_detection_{scale_factor}_{min_neighbors}.jpg", annotated_frame)
                    print(f"  💾 Saved: opencv_detection_{scale_factor}_{min_neighbors}.jpg")
                    return True
        
        print("❌ OpenCV also found no faces")
        return False
        
    except Exception as e:
        print(f"❌ Error testing OpenCV: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_frame_quality():
    """Test frame quality and characteristics"""
    print("\n" + "=" * 60)
    print("TESTING FRAME QUALITY AND CHARACTERISTICS")
    print("=" * 60)
    
    try:
        frame_path = "nvr_test_frame_fixed.jpg"
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"❌ Failed to load frame: {frame_path}")
            return False
        
        print(f"Frame shape: {frame.shape}")
        print(f"Frame dtype: {frame.dtype}")
        print(f"Frame min/max values: {frame.min()}/{frame.max()}")
        print(f"Frame mean: {frame.mean():.2f}")
        print(f"Frame std: {frame.std():.2f}")
        
        # Check for potential issues
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check brightness
        brightness = gray.mean()
        print(f"Brightness (0-255): {brightness:.2f}")
        
        # Check contrast
        contrast = gray.std()
        print(f"Contrast (std): {contrast:.2f}")
        
        # Check for motion blur
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Sharpness (Laplacian variance): {laplacian_var:.2f}")
        
        # Check for compression artifacts
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        print(f"Edge density: {edge_density:.4f}")
        
        # Save analysis
        analysis_frame = frame.copy()
        cv2.putText(analysis_frame, f"Brightness: {brightness:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(analysis_frame, f"Contrast: {contrast:.1f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(analysis_frame, f"Sharpness: {laplacian_var:.1f}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(analysis_frame, f"Edge density: {edge_density:.4f}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite("frame_analysis.jpg", analysis_frame)
        print("💾 Saved frame analysis: frame_analysis.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing frame: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debugging function"""
    print("NVR Face Detection Detailed Debugging")
    print("This script performs detailed analysis of the NVR stream")
    print("=" * 60)
    
    # Test 1: MediaPipe with saved frame
    mediapipe_success = test_mediapipe_with_saved_frame()
    
    # Test 2: OpenCV comparison
    opencv_success = test_opencv_face_detection()
    
    # Test 3: Frame quality analysis
    quality_success = test_frame_quality()
    
    # Results
    print("\n" + "=" * 60)
    print("DEBUGGING RESULTS")
    print("=" * 60)
    print(f"MediaPipe with saved frame: {'✅ SUCCESS' if mediapipe_success else '❌ FAILED'}")
    print(f"OpenCV face detection: {'✅ SUCCESS' if opencv_success else '❌ FAILED'}")
    print(f"Frame quality analysis: {'✅ SUCCESS' if quality_success else '❌ FAILED'}")
    
    if mediapipe_success or opencv_success:
        print("\n🎉 FACE DETECTION IS WORKING!")
        print("Check the generated debug_*.jpg and opencv_*.jpg files.")
    else:
        print("\n❌ FACE DETECTION STILL NOT WORKING")
        print("This suggests a fundamental issue with the frame content or MediaPipe setup.")
        print("Check the frame_analysis.jpg for frame quality metrics.")

if __name__ == "__main__":
    main()
