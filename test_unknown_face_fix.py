"""
Test script to verify unknown faces are properly rejected
"""

import sys
import os
import cv2
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import DatabaseManager
from app.core.simple_face_embedder import SimpleFaceEmbedder

def test_unknown_face_rejection():
    """Test that unknown faces are properly rejected"""
    print("=" * 60)
    print("TESTING UNKNOWN FACE REJECTION")
    print("=" * 60)
    
    try:
        # Initialize components
        db_manager = DatabaseManager("database/blazeface_frs.db")
        embedder = SimpleFaceEmbedder()
        
        # Get stored encodings
        encodings = db_manager.get_face_encodings()
        print(f"Loaded {len(encodings)} face encodings from database")
        
        if not encodings:
            print("❌ No face encodings found in database!")
            print("Please register some students first.")
            return
        
        # Create student encodings dictionary
        student_encodings = {}
        for student_id, encoding, encoding_type in encodings:
            if student_id not in student_encodings:
                student_encodings[student_id] = []
            student_encodings[student_id].append(encoding)
        
        print(f"Students in database: {list(student_encodings.keys())}")
        
        # Create a completely random "unknown" face encoding
        print("\n--- Creating Unknown Face Encoding ---")
        unknown_encoding = np.random.randn(128).astype(np.float32)
        unknown_encoding = unknown_encoding / np.linalg.norm(unknown_encoding)  # Normalize
        
        print(f"Unknown face encoding shape: {unknown_encoding.shape}")
        print(f"Unknown face encoding norm: {np.linalg.norm(unknown_encoding):.4f}")
        
        # Test with old logic (low threshold)
        print("\n--- Testing OLD Logic (threshold=0.6) ---")
        old_threshold = 0.6
        old_best_similarity = 0.0
        old_best_student = None
        
        for student_id, encodings_list in student_encodings.items():
            for i, encoding in enumerate(encodings_list):
                is_same, similarity = embedder.compare_faces(unknown_encoding, encoding, old_threshold)
                if similarity > old_best_similarity:
                    old_best_similarity = similarity
                    old_best_student = student_id
        
        print(f"OLD LOGIC RESULT:")
        print(f"  Best match: {old_best_student} with similarity {old_best_similarity:.4f}")
        if old_best_similarity > old_threshold:
            print(f"  ❌ WRONG: Would recognize as '{old_best_student}' (false positive!)")
        else:
            print(f"  ✅ Correct: Would reject as unknown")
        
        # Test with new logic (high threshold + confidence gap)
        print("\n--- Testing NEW Logic (threshold=0.90 + gap=0.05) ---")
        new_threshold = 0.90
        min_gap = 0.05
        
        new_best_similarity = 0.0
        new_best_student = None
        all_similarities = []
        
        for student_id, encodings_list in student_encodings.items():
            for i, encoding in enumerate(encodings_list):
                is_same, similarity = embedder.compare_faces(unknown_encoding, encoding, 0.01)
                all_similarities.append((student_id, i, similarity))
                if similarity > new_best_similarity:
                    new_best_similarity = similarity
                    new_best_student = student_id
        
        # Sort similarities to find second best
        sorted_similarities = sorted(all_similarities, key=lambda x: x[2], reverse=True)
        second_best = sorted_similarities[1][2] if len(sorted_similarities) > 1 else 0.0
        gap = new_best_similarity - second_best
        
        print(f"NEW LOGIC RESULT:")
        print(f"  Best match: {new_best_student} with similarity {new_best_similarity:.4f}")
        print(f"  Second best: {sorted_similarities[1][0] if len(sorted_similarities) > 1 else 'N/A'} with similarity {second_best:.4f}")
        print(f"  Gap: {gap:.4f}")
        
        # Check both conditions
        if new_best_similarity > new_threshold and gap >= min_gap:
            print(f"  ❌ WRONG: Would recognize as '{new_best_student}' (false positive!)")
        else:
            print(f"  ✅ CORRECT: Would reject as unknown")
            if new_best_similarity <= new_threshold:
                print(f"    Reason: Similarity {new_best_similarity:.4f} below threshold {new_threshold:.2f}")
            if gap < min_gap:
                print(f"    Reason: Gap {gap:.4f} below minimum {min_gap:.2f}")
        
        # Show top 5 similarities for debugging
        print(f"\n--- Top 5 Similarities ---")
        for i, (student_id, enc_idx, sim) in enumerate(sorted_similarities[:5]):
            print(f"  {i+1}. {student_id}[{enc_idx}]: {sim:.4f}")
        
        print(f"\n--- Summary ---")
        if old_best_similarity > old_threshold and new_best_similarity <= new_threshold:
            print("✅ FIX WORKING: Old logic would have false positive, new logic correctly rejects")
        elif old_best_similarity > old_threshold and gap < min_gap:
            print("✅ FIX WORKING: Old logic would have false positive, new logic rejects due to small gap")
        elif old_best_similarity <= old_threshold:
            print("ℹ️  Both old and new logic would reject (this unknown face has low similarity to all students)")
        else:
            print("⚠️  Both old and new logic would accept (this unknown face is very similar to a student)")
    
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_with_real_camera():
    """Test with real camera to see the behavior"""
    print("\n" + "=" * 60)
    print("REAL CAMERA TEST")
    print("=" * 60)
    print("To test with real camera:")
    print("1. Run the main application: python main.py")
    print("2. Go to Attendance Marking")
    print("3. Start attendance")
    print("4. Point camera at unknown people")
    print("5. Check if they show as 'Unknown Face' (red box) or get assigned names")
    print("\nExpected behavior:")
    print("- Known students: Green box with name")
    print("- Unknown people: Red box with 'Unknown Face'")
    print("- No more false name assignments!")

if __name__ == "__main__":
    test_unknown_face_rejection()
    test_with_real_camera()
