"""
Test face detection functionality
"""

import sys
import os
import cv2
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.blazeface_detector import BlazeFaceDetector
from app.core.facenet_embedder import FaceNetEmbedder

def test_face_detection():
    """Test face detection with camera"""
    try:
        print("Testing Face Detection...")
        
        # Initialize detector
        detector = BlazeFaceDetector(min_detection_confidence=0.5)
        embedder = FaceNetEmbedder()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera opened successfully. Press 'q' to quit, 's' to save face")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Detect faces
            faces = detector.detect_faces(frame)
            print(f"Frame {frame_count}: Detected {len(faces)} faces")
            
            # Draw faces
            frame_with_faces = detector.draw_faces(frame, faces)
            
            # Get embeddings for detected faces
            if faces:
                for i, face_box in enumerate(faces):
                    x, y, w, h, confidence = face_box
                    face_region = detector.extract_face_region(frame, (x, y, w, h))
                    if face_region is not None:
                        embedding = embedder.get_embedding(face_region)
                        if embedding is not None:
                            print(f"  Face {i}: Confidence={confidence:.2f}, Embedding shape={embedding.shape}")
                        else:
                            print(f"  Face {i}: Confidence={confidence:.2f}, Failed to get embedding")
                    else:
                        print(f"  Face {i}: Confidence={confidence:.2f}, Failed to extract face region")
            
            # Display frame
            cv2.imshow('Face Detection Test', frame_with_faces)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and faces:
                # Save first detected face
                face_box = faces[0]
                x, y, w, h, confidence = face_box
                face_region = detector.extract_face_region(frame, (x, y, w, h))
                if face_region is not None:
                    cv2.imwrite('test_face.jpg', face_region)
                    print(f"Saved face to test_face.jpg with confidence {confidence:.2f}")
            
            frame_count += 1
            
            # Limit frame count for testing
            if frame_count > 100:
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        embedder.release()
        
        print("Face detection test completed")
        
    except Exception as e:
        print(f"Error in face detection test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_face_detection()
