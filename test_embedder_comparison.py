"""
Compare SimpleFaceEmbedder vs FacialFeatureEmbedder
"""

import sys
import os
import numpy as np
import cv2

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder
from app.core.facial_feature_embedder import FacialFeatureEmbedder

def test_embedder_comparison():
    """Compare the two embedders"""
    print("Comparing SimpleFaceEmbedder vs FacialFeatureEmbedder...")
    
    # Initialize embedders
    simple_embedder = SimpleFaceEmbedder()
    facial_embedder = FacialFeatureEmbedder()
    
    # Create test faces
    np.random.seed(1)
    face1 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)  # Person A
    np.random.seed(2)
    face2 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)  # Person B
    
    print(f"Face 1 shape: {face1.shape}")
    print(f"Face 2 shape: {face2.shape}")
    
    # Test SimpleFaceEmbedder
    print("\n=== SimpleFaceEmbedder ===")
    simple_emb1 = simple_embedder.get_embedding(face1)
    simple_emb2 = simple_embedder.get_embedding(face2)
    
    if simple_emb1 is not None and simple_emb2 is not None:
        print(f"Embedding 1 shape: {simple_emb1.shape}, norm: {np.linalg.norm(simple_emb1):.4f}")
        print(f"Embedding 2 shape: {simple_emb2.shape}, norm: {np.linalg.norm(simple_emb2):.4f}")
        
        # Test consistency (same face)
        simple_emb1_again = simple_embedder.get_embedding(face1)
        is_same, similarity = simple_embedder.compare_faces(simple_emb1, simple_emb1_again, 0.5)
        print(f"Same face consistency: is_same={is_same}, similarity={similarity:.4f}")
        
        # Test discrimination (different faces)
        is_same, similarity = simple_embedder.compare_faces(simple_emb1, simple_emb2, 0.5)
        print(f"Different faces: is_same={is_same}, similarity={similarity:.4f}")
        
        # Test with high threshold
        is_same, similarity = simple_embedder.compare_faces(simple_emb1, simple_emb2, 0.95)
        print(f"Different faces (0.95 threshold): is_same={is_same}, similarity={similarity:.4f}")
    else:
        print("❌ SimpleFaceEmbedder failed to generate embeddings")
    
    # Test FacialFeatureEmbedder
    print("\n=== FacialFeatureEmbedder ===")
    facial_emb1 = facial_embedder.get_embedding(face1)
    facial_emb2 = facial_embedder.get_embedding(face2)
    
    if facial_emb1 is not None and facial_emb2 is not None:
        print(f"Embedding 1 shape: {facial_emb1.shape}, norm: {np.linalg.norm(facial_emb1):.4f}")
        print(f"Embedding 2 shape: {facial_emb2.shape}, norm: {np.linalg.norm(facial_emb2):.4f}")
        
        # Test consistency (same face)
        facial_emb1_again = facial_embedder.get_embedding(face1)
        is_same, similarity = facial_embedder.compare_faces(facial_emb1, facial_emb1_again, 0.5)
        print(f"Same face consistency: is_same={is_same}, similarity={similarity:.4f}")
        
        # Test discrimination (different faces)
        is_same, similarity = facial_embedder.compare_faces(facial_emb1, facial_emb2, 0.5)
        print(f"Different faces: is_same={is_same}, similarity={similarity:.4f}")
        
        # Test with high threshold
        is_same, similarity = facial_embedder.compare_faces(facial_emb1, facial_emb2, 0.95)
        print(f"Different faces (0.95 threshold): is_same={is_same}, similarity={similarity:.4f}")
    else:
        print("❌ FacialFeatureEmbedder failed to generate embeddings")
    
    # Test with different thresholds
    print("\n=== Threshold Comparison ===")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
    
    print("Threshold | Simple (same) | Simple (diff) | Facial (same) | Facial (diff)")
    print("-" * 70)
    
    for threshold in thresholds:
        # Simple embedder
        if simple_emb1 is not None and simple_emb2 is not None:
            _, sim_simple_same = simple_embedder.compare_faces(simple_emb1, simple_emb1_again, threshold)
            _, sim_simple_diff = simple_embedder.compare_faces(simple_emb1, simple_emb2, threshold)
        else:
            sim_simple_same = sim_simple_diff = 0.0
        
        # Facial embedder
        if facial_emb1 is not None and facial_emb2 is not None:
            _, sim_facial_same = facial_embedder.compare_faces(facial_emb1, facial_emb1_again, threshold)
            _, sim_facial_diff = facial_embedder.compare_faces(facial_emb1, facial_emb2, threshold)
        else:
            sim_facial_same = sim_facial_diff = 0.0
        
        print(f"{threshold:8.2f} | {sim_simple_same:12.4f} | {sim_simple_diff:13.4f} | {sim_facial_same:12.4f} | {sim_facial_diff:12.4f}")
    
    print("\n=== Analysis ===")
    if simple_emb1 is not None and simple_emb2 is not None:
        _, sim_simple = simple_embedder.compare_faces(simple_emb1, simple_emb2, 0.5)
        if sim_simple > 0.9:
            print("❌ SimpleFaceEmbedder: Different faces have high similarity - POOR DISCRIMINATION")
        else:
            print("✅ SimpleFaceEmbedder: Different faces have low similarity - GOOD DISCRIMINATION")
    
    if facial_emb1 is not None and facial_emb2 is not None:
        _, sim_facial = facial_embedder.compare_faces(facial_emb1, facial_emb2, 0.5)
        if sim_facial > 0.9:
            print("❌ FacialFeatureEmbedder: Different faces have high similarity - POOR DISCRIMINATION")
        else:
            print("✅ FacialFeatureEmbedder: Different faces have low similarity - GOOD DISCRIMINATION")

if __name__ == "__main__":
    test_embedder_comparison()
