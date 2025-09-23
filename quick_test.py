"""
Quick test of the encoding comparison fix
"""

import sys
import os
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import DatabaseManager
from app.core.simple_face_embedder import SimpleFaceEmbedder

def quick_test():
    """Quick test of encoding comparison"""
    try:
        print("Quick Test of Encoding Comparison...")
        
        # Initialize components
        db_manager = DatabaseManager("database/blazeface_frs.db")
        embedder = SimpleFaceEmbedder()
        
        # Get stored encodings
        encodings = db_manager.get_face_encodings()
        print(f"Loaded {len(encodings)} face encodings")
        
        if not encodings:
            print("No encodings found!")
            return
        
        # Test with first encoding
        student_id, encoding, encoding_type = encodings[0]
        print(f"Testing with student {student_id}")
        
        # Test 1: Same encoding vs itself
        is_same, similarity = embedder.compare_faces(encoding, encoding, 0.9)
        print(f"Same encoding vs itself: is_same={is_same}, similarity={similarity:.4f}")
        
        # Test 2: Same encoding vs random
        random_encoding = np.random.rand(128).astype(np.float32)
        random_encoding = random_encoding / np.linalg.norm(random_encoding)
        is_same2, similarity2 = embedder.compare_faces(encoding, random_encoding, 0.9)
        print(f"Same encoding vs random: is_same={is_same2}, similarity={similarity2:.4f}")
        
        print("Expected results:")
        print("- Same vs same: is_same=True, similarity=1.0000")
        print("- Same vs random: is_same=False, similarity<0.9")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
