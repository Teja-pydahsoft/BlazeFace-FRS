"""
Basic MediaPipe test to verify installation and functionality
"""

import cv2
import numpy as np
import mediapipe as mp
import time

def test_mediapipe_basic():
    """Test MediaPipe with a simple synthetic image"""
    print("Testing MediaPipe with synthetic face image...")
    
    try:
        # Create a simple synthetic face image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a simple face
        # Face outline
        cv2.ellipse(img, (320, 240), (100, 120), 0, 0, 360, (200, 180, 160), -1)
        
        # Eyes
        cv2.circle(img, (290, 200), 15, (0, 0, 0), -1)
        cv2.circle(img, (350, 200), 15, (0, 0, 0), -1)
        
        # Nose
        cv2.ellipse(img, (320, 240), (10, 20), 0, 0, 360, (150, 130, 110), -1)
        
        # Mouth
        cv2.ellipse(img, (320, 280), (30, 15), 0, 0, 180, (100, 50, 50), -1)
        
        # Save synthetic image
        cv2.imwrite("synthetic_face.jpg", img)
        print("✅ Created synthetic face image: synthetic_face.jpg")
        
        # Test MediaPipe
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        
        with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.1
        ) as face_detection:
            
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process
            results = face_detection.process(rgb_img)
            
            if results.detections:
                print(f"✅ MediaPipe detected {len(results.detections)} faces in synthetic image")
                
                # Draw detections
                annotated_img = img.copy()
                for detection in results.detections:
                    mp_drawing.draw_detection(annotated_img, detection)
                
                cv2.imwrite("synthetic_face_detected.jpg", annotated_img)
                print("💾 Saved detection result: synthetic_face_detected.jpg")
                return True
            else:
                print("❌ MediaPipe found no faces in synthetic image")
                return False
                
    except Exception as e:
        print(f"❌ Error testing MediaPipe: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mediapipe_with_webcam():
    """Test MediaPipe with webcam if available"""
    print("\nTesting MediaPipe with webcam...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No webcam available")
            return False
        
        print("✅ Webcam opened")
        
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        
        with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.3
        ) as face_detection:
            
            for i in range(5):  # Test 5 frames
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process
                results = face_detection.process(rgb_frame)
                
                if results.detections:
                    print(f"✅ MediaPipe detected {len(results.detections)} faces in webcam frame {i+1}")
                    
                    # Draw and save
                    annotated_frame = frame.copy()
                    for detection in results.detections:
                        mp_drawing.draw_detection(annotated_frame, detection)
                    
                    cv2.imwrite(f"webcam_detection_{i+1}.jpg", annotated_frame)
                    print(f"💾 Saved webcam detection: webcam_detection_{i+1}.jpg")
                    cap.release()
                    return True
                else:
                    print(f"Frame {i+1}: No faces detected")
                
                time.sleep(0.5)
        
        cap.release()
        print("❌ No faces detected in webcam frames")
        return False
        
    except Exception as e:
        print(f"❌ Error testing webcam: {e}")
        return False

if __name__ == "__main__":
    print("MediaPipe Basic Functionality Test")
    print("=" * 40)
    
    # Test 1: Synthetic image
    synthetic_success = test_mediapipe_basic()
    
    # Test 2: Webcam
    webcam_success = test_mediapipe_with_webcam()
    
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"Synthetic image test: {'✅ PASS' if synthetic_success else '❌ FAIL'}")
    print(f"Webcam test: {'✅ PASS' if webcam_success else '❌ FAIL'}")
    
    if synthetic_success or webcam_success:
        print("\n🎉 MediaPipe is working correctly!")
    else:
        print("\n❌ MediaPipe has issues - check installation")