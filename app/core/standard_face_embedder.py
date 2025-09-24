"""
Standard Face Embedder using face_recognition library (dlib's ResNet model)
Industry-standard 128-dimensional face embeddings with proven accuracy
"""

import cv2
import numpy as np
import face_recognition
from typing import Optional, Tuple, List
import logging
import os

class StandardFaceEmbedder:
    """
    Standard face embedder using face_recognition library
    Provides industry-standard 128-dimensional face embeddings
    """
    
    def __init__(self, model: str = 'large'):
        """
        Initialize standard face embedder
        
        Args:
            model: Face recognition model ('small' or 'large')
                  'small': Faster, less accurate (5 face landmarks)
                  'large': Slower, more accurate (68 face landmarks)
        """
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.embedding_size = 128  # face_recognition always returns 128-dimensional embeddings
        
        # Verify face_recognition is available
        try:
            # Test with a dummy image to ensure library works
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            face_recognition.face_encodings(test_image, model=model)
            self.logger.info(f"StandardFaceEmbedder initialized with {model} model")
        except Exception as e:
            self.logger.error(f"Failed to initialize face_recognition: {e}")
            raise RuntimeError(f"face_recognition library not available: {e}")
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for face_recognition
        
        Args:
            face_image: Input face image (BGR format)
            
        Returns:
            Preprocessed face image (RGB format)
        """
        try:
            # Convert BGR to RGB (face_recognition expects RGB)
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image
            
            # Ensure minimum size for face_recognition
            h, w = face_rgb.shape[:2]
            if h < 40 or w < 40:
                # Resize if too small
                scale = max(40 / h, 40 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                face_rgb = cv2.resize(face_rgb, (new_w, new_h))
            
            return face_rgb
            
        except Exception as e:
            self.logger.error(f"Error preprocessing face: {str(e)}")
            return None
    
    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding from face image using face_recognition
        
        Args:
            face_image: Input face image (BGR format)
            
        Returns:
            Face embedding vector (128-dimensional) or None if failed
        """
        try:
            # Preprocess face
            processed_face = self.preprocess_face(face_image)
            if processed_face is None:
                return None
            
            # Get face encodings using face_recognition
            encodings = face_recognition.face_encodings(
                processed_face, 
                model=self.model
            )
            
            if not encodings:
                self.logger.warning("No face encodings found in image")
                return None
            
            # Return the first (and usually only) encoding
            embedding = encodings[0]
            
            # Verify embedding dimensions
            if len(embedding) != self.embedding_size:
                self.logger.warning(f"Unexpected embedding size: {len(embedding)} (expected {self.embedding_size})")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error getting face embedding: {str(e)}")
            return None
    
    def get_embeddings_batch(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for multiple face images
        
        Args:
            face_images: List of face images
            
        Returns:
            List of face embeddings
        """
        embeddings = []
        for face_image in face_images:
            embedding = self.get_embedding(face_image)
            embeddings.append(embedding)
        return embeddings
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                     threshold: float = 0.6) -> Tuple[bool, float]:
        """
        Compare two face embeddings using face_recognition's distance function
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Distance threshold (lower = more strict)
                      face_recognition uses Euclidean distance, not cosine similarity
                      Typical values: 0.4-0.6 (lower = more strict)
            
        Returns:
            Tuple of (is_same_person, distance_score)
        """
        try:
            if embedding1 is None or embedding2 is None:
                return False, float('inf')
            
            # face_recognition uses Euclidean distance
            distance = face_recognition.face_distance([embedding1], embedding2)[0]
            
            # Lower distance means more similar
            is_same = distance <= threshold
            
            return is_same, distance
            
        except Exception as e:
            self.logger.error(f"Error comparing faces: {str(e)}")
            return False, float('inf')
    
    def find_best_match(self, query_embedding: np.ndarray, 
                       reference_embeddings: List[np.ndarray],
                       reference_labels: List[str],
                       threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """
        Find best matching face from reference embeddings
        
        Args:
            query_embedding: Query face embedding
            reference_embeddings: List of reference embeddings
            reference_labels: List of reference labels
            threshold: Distance threshold
            
        Returns:
            Tuple of (best_match_label, best_distance)
        """
        try:
            if not reference_embeddings or not reference_labels:
                return None, float('inf')
            
            best_distance = float('inf')
            best_label = None
            
            for embedding, label in zip(reference_embeddings, reference_labels):
                is_same, distance = self.compare_faces(query_embedding, embedding, threshold)
                if is_same and distance < best_distance:
                    best_distance = distance
                    best_label = label
            
            return best_label, best_distance
            
        except Exception as e:
            self.logger.error(f"Error finding best match: {str(e)}")
            return None, float('inf')
    
    def detect_and_encode_faces(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect faces in image and return their encodings
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of tuples (face_location, face_encoding)
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image, model=self.model)
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations, model=self.model)
            
            # Return pairs of (location, encoding)
            return list(zip(face_locations, face_encodings))
            
        except Exception as e:
            self.logger.error(f"Error detecting and encoding faces: {str(e)}")
            return []
    
    def save_embedding(self, embedding: np.ndarray, filepath: str):
        """
        Save face embedding to file
        
        Args:
            embedding: Face embedding vector
            filepath: Path to save the embedding
        """
        try:
            np.save(filepath, embedding)
            self.logger.info(f"Embedding saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving embedding: {str(e)}")
    
    def load_embedding(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load face embedding from file
        
        Args:
            filepath: Path to the embedding file
            
        Returns:
            Face embedding vector or None if failed
        """
        try:
            embedding = np.load(filepath)
            self.logger.info(f"Embedding loaded from {filepath}")
            return embedding
        except Exception as e:
            self.logger.error(f"Error loading embedding: {str(e)}")
            return None
    
    def get_embedding_info(self) -> dict:
        """
        Get information about this embedder
        
        Returns:
            Dictionary with embedder information
        """
        return {
            'name': 'StandardFaceEmbedder',
            'library': 'face_recognition',
            'model': self.model,
            'embedding_size': self.embedding_size,
            'distance_metric': 'euclidean',
            'description': f'Industry-standard face embeddings using dlib\'s ResNet model ({self.model})'
        }
    
    def release(self):
        """Release resources"""
        # face_recognition doesn't require explicit cleanup
        pass
    
    def __del__(self):
        """Destructor"""
        self.release()

