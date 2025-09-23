"""
Simple test for face detection
"""

import sys
import os
import cv2
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.blazeface_detector import BlazeFaceDetector

def test_simple_detection():
    """Test simple face detection"""
    try:
        print("Testing Simple Face Detection...")
        
        # Initialize detector
        detector = BlazeFaceDetector(min_detection_confidence=0.5)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera opened successfully. Press 'q' to quit")
        
        frame_count = 0
        while frame_count < 50:  # Test for 50 frames
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            if faces:
                print(f"Frame {frame_count}: Detected {len(faces)} faces")
                for i, face_box in enumerate(faces):
                    x, y, w, h, confidence = face_box
                    print(f"  Face {i}: x={x}, y={y}, w={w}, h={h}, confidence={confidence:.2f}")
            else:
                if frame_count % 10 == 0:  # Print every 10th frame
                    print(f"Frame {frame_count}: No faces detected")
            
            # Draw faces
            frame_with_faces = detector.draw_faces(frame, faces)
            
            # Display frame
            cv2.imshow('Simple Face Detection Test', frame_with_faces)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        
        print("Simple face detection test completed")
        
    except Exception as e:
        print(f"Error in simple face detection test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_detection()
