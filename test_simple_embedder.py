"""
Test the simple face embedder
"""

import sys
import os
import cv2
import numpy as np

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.simple_face_embedder import SimpleFaceEmbedder

def test_simple_embedder():
    """Test the simple face embedder"""
    try:
        print("Testing Simple Face Embedder...")
        
        # Initialize embedder
        embedder = SimpleFaceEmbedder()
        
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Get embedding
        embedding = embedder.get_embedding(test_image)
        if embedding is not None:
            print(f"✓ Embedding generated successfully")
            print(f"  - Shape: {embedding.shape}")
            print(f"  - Type: {embedding.dtype}")
            print(f"  - Sample values: {embedding[:5]}")
            print(f"  - Norm: {np.linalg.norm(embedding):.4f}")
        else:
            print("✗ Failed to generate embedding")
            return
        
        # Test consistency - same image should give same embedding
        embedding2 = embedder.get_embedding(test_image)
        if embedding2 is not None:
            is_same, similarity = embedder.compare_faces(embedding, embedding2, 0.9)
            print(f"✓ Consistency test: is_same={is_same}, similarity={similarity:.4f}")
            
            if is_same and similarity > 0.99:
                print("✓ Embeddings are consistent")
            else:
                print("✗ Embeddings are not consistent")
        else:
            print("✗ Failed to generate second embedding")
        
        # Test with different image
        test_image2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        embedding3 = embedder.get_embedding(test_image2)
        if embedding3 is not None:
            is_same, similarity = embedder.compare_faces(embedding, embedding3, 0.9)
            print(f"✓ Different image test: is_same={is_same}, similarity={similarity:.4f}")
            
            if not is_same:
                print("✓ Different images correctly identified as different")
            else:
                print("✗ Different images incorrectly identified as same")
        else:
            print("✗ Failed to generate third embedding")
        
        print("Simple face embedder test completed")
        
    except Exception as e:
        print(f"Error in simple embedder test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_embedder()
