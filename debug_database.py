"""
Debug database contents to check face encodings
"""

import sys
import os
import sqlite3
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import DatabaseManager

def debug_database():
    """Debug database contents"""
    try:
        print("Debugging Database Contents...")
        
        # Initialize database manager
        db_manager = DatabaseManager("database/blazeface_frs.db")
        
        # Get all students
        students = db_manager.get_all_students()
        print(f"\nStudents in database: {len(students)}")
        for student in students:
            print(f"  - ID: {student['student_id']}, Name: {student['name']}")
        
        # Get all face encodings
        encodings = db_manager.get_face_encodings()
        print(f"\nFace encodings in database: {len(encodings)}")
        for i, (student_id, encoding, encoding_type) in enumerate(encodings):
            print(f"  - {i+1}: Student ID: {student_id}, Type: {encoding_type}, Shape: {encoding.shape}, Dtype: {encoding.dtype}")
            print(f"    Sample values: {encoding[:5]}...")
        
        # Test face comparison
        if len(encodings) >= 2:
            print(f"\nTesting face comparison...")
            from app.core.simple_face_embedder import SimpleFaceEmbedder
            embedder = SimpleFaceEmbedder()
            
            encoding1 = encodings[0][1]
            encoding2 = encodings[0][1]  # Same encoding
            
            is_same, similarity = embedder.compare_faces(encoding1, encoding2, 0.6)
            print(f"Same encoding comparison: is_same={is_same}, similarity={similarity:.4f}")
            
            if len(encodings) >= 2:
                encoding2 = encodings[1][1]  # Different encoding
                is_same, similarity = embedder.compare_faces(encoding1, encoding2, 0.6)
                print(f"Different encoding comparison: is_same={is_same}, similarity={similarity:.4f}")
        
        # Get statistics
        stats = db_manager.get_statistics()
        print(f"\nDatabase Statistics:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
    except Exception as e:
        print(f"Error debugging database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_database()
