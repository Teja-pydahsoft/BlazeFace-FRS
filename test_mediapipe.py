"""
Test MediaPipe face detection directly
"""

import cv2
import numpy as np
import mediapipe as mp

def test_mediapipe_detection():
    """Test MediaPipe face detection directly"""
    try:
        print("Testing MediaPipe Face Detection...")
        
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        
        face_detection = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
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
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = face_detection.process(rgb_frame)
            
            faces = []
            if results.detections:
                h, w, _ = frame.shape
                
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coordinates to absolute
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Get confidence score
                    confidence = detection.score[0]
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    if width > 0 and height > 0:
                        faces.append((x, y, width, height, confidence))
            
            if faces:
                print(f"Frame {frame_count}: Detected {len(faces)} faces")
                for i, face_box in enumerate(faces):
                    x, y, w, h, confidence = face_box
                    print(f"  Face {i}: x={x}, y={y}, w={w}, h={h}, confidence={confidence:.2f}")
            else:
                if frame_count % 10 == 0:  # Print every 10th frame
                    print(f"Frame {frame_count}: No faces detected")
            
            # Draw faces
            for (x, y, w, h, confidence) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"Face: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('MediaPipe Face Detection Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        del face_detection
        
        print("MediaPipe face detection test completed")
        
    except Exception as e:
        print(f"Error in MediaPipe face detection test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mediapipe_detection()
