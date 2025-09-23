"""
Test script to validate the face recognition system fixes
Tests for false positive prevention and accurate recognition
"""

import sys
import os
import cv2
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import DatabaseManager
from app.core.facenet_embedder import FaceNetEmbedder
from app.core.enhanced_face_embedder import EnhancedFaceEmbedder
from app.core.simple_face_embedder import SimpleFaceEmbedder
from app.core.blazeface_detector import BlazeFaceDetector

def test_recognition_accuracy():
    """Test recognition accuracy with different thresholds"""
    print("=" * 60)
    print("TESTING FACE RECOGNITION SYSTEM FIXES")
    print("=" * 60)
    
    try:
        # Initialize components
        db_manager = DatabaseManager("database/blazeface_frs.db")
        
        # Test different embedders
        embedders = [
            ("FaceNet", FaceNetEmbedder()),
            ("Enhanced", EnhancedFaceEmbedder()),
            ("Simple", SimpleFaceEmbedder())
        ]
        
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
        
        # Test with different thresholds
        thresholds = [0.6, 0.7, 0.8, 0.85, 0.90, 0.95]
        
        print("\n" + "=" * 60)
        print("THRESHOLD TESTING")
        print("=" * 60)
        
        for embedder_name, embedder in embedders:
            print(f"\n--- Testing {embedder_name} Embedder ---")
            
            for threshold in thresholds:
                print(f"\nThreshold: {threshold:.2f}")
                
                # Test with each student's own encoding (should match)
                true_positives = 0
                false_positives = 0
                
                for student_id, encodings_list in student_encodings.items():
                    for i, encoding in enumerate(encodings_list):
                        # Test with same person (should match)
                        is_same, similarity = embedder.compare_faces(encoding, encoding, threshold)
                        if is_same:
                            true_positives += 1
                        else:
                            print(f"  ❌ False negative: {student_id}[{i}] similarity={similarity:.4f}")
                        
                        # Test with different person (should not match)
                        for other_student_id, other_encodings in student_encodings.items():
                            if other_student_id != student_id:
                                for j, other_encoding in enumerate(other_encodings):
                                    is_same, similarity = embedder.compare_faces(encoding, other_encoding, threshold)
                                    if is_same:
                                        false_positives += 1
                                        print(f"  ❌ False positive: {student_id}[{i}] vs {other_student_id}[{j}] similarity={similarity:.4f}")
                
                total_tests = len(encodings) * len(encodings)
                accuracy = (true_positives / len(encodings)) * 100
                false_positive_rate = (false_positives / (total_tests - len(encodings))) * 100
                
                print(f"  True Positives: {true_positives}/{len(encodings)} ({accuracy:.1f}%)")
                print(f"  False Positives: {false_positives} ({false_positive_rate:.1f}%)")
                
                # Recommend threshold
                if false_positive_rate < 5 and accuracy > 90:
                    print(f"  ✅ GOOD: Low false positives, high accuracy")
                elif false_positive_rate < 10:
                    print(f"  ⚠️  ACCEPTABLE: Some false positives")
                else:
                    print(f"  ❌ POOR: Too many false positives")
    
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_new_matching_logic():
    """Test the new matching logic with confidence gap"""
    print("\n" + "=" * 60)
    print("TESTING NEW MATCHING LOGIC")
    print("=" * 60)
    
    try:
        db_manager = DatabaseManager("database/blazeface_frs.db")
        embedder = SimpleFaceEmbedder()  # Use simple embedder for testing
        
        # Get encodings
        encodings = db_manager.get_face_encodings()
        if not encodings:
            print("❌ No encodings found")
            return
        
        # Create test scenarios
        student_encodings = {}
        for student_id, encoding, encoding_type in encodings:
            if student_id not in student_encodings:
                student_encodings[student_id] = []
            student_encodings[student_id].append(encoding)
        
        print("Testing new matching logic with confidence gap...")
        
        # Test with each student
        for student_id, encodings_list in student_encodings.items():
            print(f"\n--- Testing {student_id} ---")
            
            for i, query_encoding in enumerate(encodings_list):
                print(f"  Query encoding {i}:")
                
                # Find best match using new logic
                best_similarity = 0.0
                best_student_id = None
                all_similarities = []
                
                recognition_threshold = 0.85  # New higher threshold
                
                for other_student_id, other_encodings in student_encodings.items():
                    student_max_similarity = 0.0
                    
                    for j, other_encoding in enumerate(other_encodings):
                        is_same, similarity = embedder.compare_faces(query_encoding, other_encoding, 0.01)
                        all_similarities.append((other_student_id, j, similarity))
                        
                        if similarity > student_max_similarity:
                            student_max_similarity = similarity
                    
                    if student_max_similarity > recognition_threshold and student_max_similarity > best_similarity:
                        best_similarity = student_max_similarity
                        best_student_id = other_student_id
                
                # Check confidence gap
                if best_similarity > recognition_threshold:
                    # Find second best
                    sorted_similarities = sorted(all_similarities, key=lambda x: x[2], reverse=True)
                    second_best = 0.0
                    for _, _, sim in sorted_similarities[1:3]:
                        if sim > second_best:
                            second_best = sim
                    
                    min_gap = 0.05
                    gap = best_similarity - second_best
                    
                    if gap >= min_gap:
                        print(f"    ✅ CLEAR MATCH: {best_student_id} (similarity: {best_similarity:.4f}, gap: {gap:.4f})")
                    else:
                        print(f"    ⚠️  AMBIGUOUS: {best_student_id} (similarity: {best_similarity:.4f}, gap: {gap:.4f}) - TOO CLOSE")
                else:
                    print(f"    ❌ NO MATCH: Best similarity {best_similarity:.4f} below threshold {recognition_threshold:.2f}")
    
    except Exception as e:
        print(f"❌ Error during matching logic test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("Face Recognition System Fix Validation")
    print("This script tests the fixes for false positive prevention")
    
    # Test recognition accuracy
    test_recognition_accuracy()
    
    # Test new matching logic
    test_new_matching_logic()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("1. Use recognition_confidence: 0.90 or higher")
    print("2. Use min_confidence_gap: 0.05 or higher")
    print("3. FaceNet embedder is most accurate (if available)")
    print("4. Enhanced embedder is good fallback")
    print("5. Simple embedder works but may have more false positives")
    print("=" * 60)

if __name__ == "__main__":
    main()
