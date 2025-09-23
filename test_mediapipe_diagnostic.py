"""
Comprehensive MediaPipe diagnostic test
"""

import cv2
import numpy as np
import mediapipe as mp
import time

def test_mediapipe_installation():
    """Test MediaPipe installation and basic functionality"""
    print("Testing MediaPipe installation...")
    
    try:
        print(f"MediaPipe version: {mp.__version__}")
        
        # Test basic imports
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        
        print("✅ MediaPipe modules imported successfully")
        
        # Test creating face detection instance
        face_detection = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.1
        )
        
        print("✅ Face detection instance created successfully")
        
        # Test with a simple image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray image
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = face_detection.process(rgb_img)
        print(f"✅ MediaPipe processed image successfully (detections: {len(results.detections) if results.detections else 0})")
        
        face_detection.close()
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe installation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_face_image():
    """Test with a real face image"""
    print("\nTesting with real face image...")
    
    try:
        # Create a more realistic face image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Face skin color
        cv2.ellipse(img, (320, 240), (120, 150), 0, 0, 360, (220, 200, 180), -1)
        
        # Eyes
        cv2.circle(img, (280, 200), 20, (255, 255, 255), -1)  # White
        cv2.circle(img, (360, 200), 20, (255, 255, 255), -1)  # White
        cv2.circle(img, (280, 200), 12, (0, 0, 0), -1)  # Black pupil
        cv2.circle(img, (360, 200), 12, (0, 0, 0), -1)  # Black pupil
        
        # Nose
        cv2.ellipse(img, (320, 250), (8, 25), 0, 0, 360, (200, 180, 160), -1)
        
        # Mouth
        cv2.ellipse(img, (320, 300), (40, 20), 0, 0, 180, (150, 100, 100), -1)
        
        # Save the image
        cv2.imwrite("realistic_face.jpg", img)
        print("✅ Created realistic face image: realistic_face.jpg")
        
        # Test with MediaPipe
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        
        with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.01  # Very low threshold
        ) as face_detection:
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_img)
            
            if results.detections:
                print(f"✅ MediaPipe detected {len(results.detections)} faces!")
                
                # Draw and save
                annotated_img = img.copy()
                for detection in results.detections:
                    mp_drawing.draw_detection(annotated_img, detection)
                
                cv2.imwrite("realistic_face_detected.jpg", annotated_img)
                print("💾 Saved detection result: realistic_face_detected.jpg")
                return True
            else:
                print("❌ MediaPipe found no faces in realistic image")
                return False
                
    except Exception as e:
        print(f"❌ Error testing realistic face: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_nvr_frame():
    """Test with the actual NVR frame"""
    print("\nTesting with NVR frame...")
    
    try:
        # Load the NVR frame
        frame_path = "nvr_test_frame_fixed.jpg"
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"❌ Failed to load NVR frame: {frame_path}")
            return False
        
        print(f"✅ Loaded NVR frame: {frame.shape}")
        
        # Test different approaches
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        
        # Test 1: Original resolution
        print("Testing original resolution...")
        with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.001  # Extremely low threshold
        ) as face_detection:
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                print(f"✅ Original resolution: {len(results.detections)} faces detected!")
                return True
            else:
                print("❌ Original resolution: No faces detected")
        
        # Test 2: Resized to 640x480
        print("Testing resized to 640x480...")
        with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.001
        ) as face_detection:
            
            resized = cv2.resize(frame, (640, 480))
            rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_resized)
            
            if results.detections:
                print(f"✅ Resized 640x480: {len(results.detections)} faces detected!")
                
                # Draw and save
                annotated = resized.copy()
                for detection in results.detections:
                    mp_drawing.draw_detection(annotated, detection)
                cv2.imwrite("nvr_detected_640x480.jpg", annotated)
                print("💾 Saved: nvr_detected_640x480.jpg")
                return True
            else:
                print("❌ Resized 640x480: No faces detected")
        
        # Test 3: Different preprocessing
        print("Testing with preprocessing...")
        with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.001
        ) as face_detection:
            
            # Try enhanced contrast
            enhanced = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
            resized = cv2.resize(enhanced, (640, 480))
            rgb_enhanced = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            results = face_detection.process(rgb_enhanced)
            
            if results.detections:
                print(f"✅ Enhanced contrast: {len(results.detections)} faces detected!")
                
                # Draw and save
                annotated = resized.copy()
                for detection in results.detections:
                    mp_drawing.draw_detection(annotated, detection)
                cv2.imwrite("nvr_detected_enhanced.jpg", annotated)
                print("💾 Saved: nvr_detected_enhanced.jpg")
                return True
            else:
                print("❌ Enhanced contrast: No faces detected")
        
        return False
        
    except Exception as e:
        print(f"❌ Error testing NVR frame: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opencv_comparison():
    """Test OpenCV face detection for comparison"""
    print("\nTesting OpenCV face detection for comparison...")
    
    try:
        # Load NVR frame
        frame = cv2.imread("nvr_test_frame_fixed.jpg")
        if frame is None:
            print("❌ Failed to load NVR frame")
            return False
        
        # Load OpenCV face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            print("❌ Failed to load OpenCV face cascade")
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            print(f"✅ OpenCV detected {len(faces)} faces!")
            
            # Draw and save
            annotated = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated, "OpenCV", (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imwrite("nvr_opencv_detected.jpg", annotated)
            print("💾 Saved: nvr_opencv_detected.jpg")
            return True
        else:
            print("❌ OpenCV found no faces")
            return False
            
    except Exception as e:
        print(f"❌ Error testing OpenCV: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("MediaPipe Comprehensive Diagnostic Test")
    print("=" * 50)
    
    # Test 1: Installation
    install_success = test_mediapipe_installation()
    
    # Test 2: Realistic face
    realistic_success = test_with_real_face_image()
    
    # Test 3: NVR frame
    nvr_success = test_with_nvr_frame()
    
    # Test 4: OpenCV comparison
    opencv_success = test_opencv_comparison()
    
    # Results
    print("\n" + "=" * 50)
    print("DIAGNOSTIC RESULTS")
    print("=" * 50)
    print(f"MediaPipe installation: {'✅ PASS' if install_success else '❌ FAIL'}")
    print(f"Realistic face test: {'✅ PASS' if realistic_success else '❌ FAIL'}")
    print(f"NVR frame test: {'✅ PASS' if nvr_success else '❌ FAIL'}")
    print(f"OpenCV comparison: {'✅ PASS' if opencv_success else '❌ FAIL'}")
    
    if nvr_success:
        print("\n🎉 MediaPipe is working with NVR frames!")
    elif opencv_success:
        print("\n⚠️ OpenCV works but MediaPipe doesn't - MediaPipe may have issues")
    else:
        print("\n❌ Both MediaPipe and OpenCV failed - frame quality issue")
