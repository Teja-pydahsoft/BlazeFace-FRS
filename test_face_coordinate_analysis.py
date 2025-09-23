"""
Analyze face detection coordinates and implement face indexing
"""

import sys
import os
import cv2
import time
import numpy as np
import logging
from typing import List, Tuple, Dict

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.camera_utils import CameraManager
from app.core.blazeface_detector import BlazeFaceDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceIndexer:
    """Face indexing and tracking system"""
    
    def __init__(self, max_distance: float = 50.0):
        self.max_distance = max_distance
        self.face_tracks = {}  # face_id -> (x, y, w, h, confidence, frame_count)
        self.next_face_id = 1
        self.frame_count = 0
    
    def update_faces(self, faces: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int, float, int]]:
        """
        Update face tracking and assign IDs
        
        Args:
            faces: List of face detections (x, y, w, h, confidence)
            
        Returns:
            List of face detections with IDs (x, y, w, h, confidence, face_id)
        """
        self.frame_count += 1
        faces_with_ids = []
        
        for face in faces:
            x, y, w, h, conf = face
            center_x, center_y = x + w//2, y + h//2
            
            # Find closest existing face
            best_match_id = None
            best_distance = float('inf')
            
            for face_id, track in self.face_tracks.items():
                track_x, track_y, track_w, track_h, track_conf, track_frames = track
                track_center_x = track_x + track_w//2
                track_center_y = track_y + track_h//2
                
                distance = np.sqrt((center_x - track_center_x)**2 + (center_y - track_center_y)**2)
                
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match_id = face_id
            
            if best_match_id is not None:
                # Update existing face
                self.face_tracks[best_match_id] = (x, y, w, h, conf, self.frame_count)
                faces_with_ids.append((x, y, w, h, conf, best_match_id))
            else:
                # Create new face track
                face_id = self.next_face_id
                self.next_face_id += 1
                self.face_tracks[face_id] = (x, y, w, h, conf, self.frame_count)
                faces_with_ids.append((x, y, w, h, conf, face_id))
        
        # Remove old faces (not seen for 5 frames)
        old_faces = [face_id for face_id, track in self.face_tracks.items() 
                    if self.frame_count - track[5] > 5]
        for face_id in old_faces:
            del self.face_tracks[face_id]
        
        return faces_with_ids

def analyze_face_coordinates():
    """Analyze face detection coordinates from saved frame"""
    print("=" * 60)
    print("FACE COORDINATE ANALYSIS")
    print("=" * 60)
    
    try:
        # Load saved frame
        frame_path = "nvr_test_frame_fixed.jpg"
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"❌ Failed to load frame: {frame_path}")
            return False
        
        print(f"✅ Loaded frame: {frame.shape}")
        
        # Initialize detector
        detector = BlazeFaceDetector(
            min_detection_confidence=0.01,
            use_opencv_fallback=True
        )
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        if faces:
            print(f"✅ Found {len(faces)} faces")
            print("\n--- Face Coordinate Analysis ---")
            
            for i, face in enumerate(faces):
                x, y, w, h, conf = face
                center_x, center_y = x + w//2, y + h//2
                print(f"Face {i+1}:")
                print(f"  Bounding box: ({x}, {y}, {w}, {h})")
                print(f"  Center: ({center_x}, {center_y})")
                print(f"  Confidence: {conf:.3f}")
                print(f"  Area: {w * h} pixels")
                print(f"  Aspect ratio: {w/h:.2f}")
                print()
            
            # Initialize face indexer
            indexer = FaceIndexer()
            faces_with_ids = indexer.update_faces(faces)
            
            print("--- Face Indexing Results ---")
            for face in faces_with_ids:
                x, y, w, h, conf, face_id = face
                print(f"Face ID {face_id}: bbox=({x},{y},{w},{h}), conf={conf:.3f}")
            
            # Draw faces with IDs
            frame_with_faces = frame.copy()
            for face in faces_with_ids:
                x, y, w, h, conf, face_id = face
                cv2.rectangle(frame_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_with_faces, f"ID:{face_id}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_with_faces, f"{conf:.2f}", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imwrite("face_coordinate_analysis.jpg", frame_with_faces)
            print("💾 Saved: face_coordinate_analysis.jpg")
            
            detector.release()
            return True
        else:
            print("❌ No faces detected")
            detector.release()
            return False
            
    except Exception as e:
        print(f"❌ Error in coordinate analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Face Coordinate Analysis and Indexing Test")
    print("This analyzes face detection coordinates and implements indexing")
    print("=" * 60)
    
    success = analyze_face_coordinates()
    
    if success:
        print("\n🎉 FACE COORDINATE ANALYSIS COMPLETE!")
        print("✅ Face coordinates analyzed")
        print("✅ Face indexing implemented")
        print("✅ Check face_coordinate_analysis.jpg for results")
    else:
        print("\n❌ Face coordinate analysis failed")

if __name__ == "__main__":
    main()
