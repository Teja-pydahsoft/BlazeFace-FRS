"""
InsightFace Embedder for high-accuracy face recognition
Provides state-of-the-art face embeddings with superior accuracy
"""

import cv2
import numpy as np
import insightface
from typing import Optional, Tuple, List
import logging
import os

class InsightFaceEmbedder:
    """
    InsightFace embedder for high-accuracy face recognition
    Provides state-of-the-art face embeddings with superior accuracy
    """
    
    def __init__(self, model_name: str = 'buffalo_l', providers: List[str] = None):
        """
        Initialize InsightFace embedder
        
        Args:
            model_name: InsightFace model name
                       'buffalo_l': Large model (best accuracy)
                       'buffalo_m': Medium model (balanced)
                       'buffalo_s': Small model (fastest)
            providers: ONNX runtime providers (e.g., ['CPUExecutionProvider'])
        """
        self.model_name = model_name
        self.providers = providers or ['CPUExecutionProvider']
        self.logger = logging.getLogger(__name__)
        self.embedding_size = 512  # InsightFace typically uses 512-dimensional embeddings
        
        # Initialize InsightFace app
        try:
            self.app = insightface.app.FaceAnalysis(
                name=model_name,
                providers=providers
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info(f"InsightFace embedder initialized with {model_name} model")
        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace: {e}")
            raise RuntimeError(f"InsightFace not available: {e}")
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for InsightFace
        
        Args:
            face_image: Input face image (BGR format)
            
        Returns:
            Preprocessed face image
        """
        try:
            # InsightFace expects BGR format, so no conversion needed
            # Ensure minimum size
            h, w = face_image.shape[:2]
            if h < 40 or w < 40:
                # Resize if too small
                scale = max(40 / h, 40 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                face_image = cv2.resize(face_image, (new_w, new_h))
            
            return face_image
            
        except Exception as e:
            self.logger.error(f"Error preprocessing face: {str(e)}")
            return None
    
    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding from face image using InsightFace
        
        Args:
            face_image: Input face image (BGR format)
            
        Returns:
            Face embedding vector (512-dimensional) or None if failed
        """
        try:
            # Preprocess face
            processed_face = self.preprocess_face(face_image)
            if processed_face is None:
                return None
            
            # Get face analysis (detection + embedding)
            faces = self.app.get(processed_face)
            
            if not faces:
                self.logger.warning("No faces detected in image")
                return None
            
            # Get the first (and usually only) face embedding
            face = faces[0]
            embedding = face.embedding
            
            # Verify embedding dimensions
            if len(embedding) != self.embedding_size:
                self.logger.warning(f"Unexpected embedding size: {len(embedding)} (expected {self.embedding_size})")
                self.embedding_size = len(embedding)  # Update to actual size
            
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
        Compare two face embeddings using cosine similarity
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold (higher = more strict)
                      Typical values: 0.5-0.7 (higher = more strict)
            
        Returns:
            Tuple of (is_same_person, similarity_score)
        """
        try:
            if embedding1 is None or embedding2 is None:
                return False, 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return False, 0.0
            
            similarity = dot_product / (norm1 * norm2)
            is_same = similarity >= threshold
            
            return is_same, similarity
            
        except Exception as e:
            self.logger.error(f"Error comparing faces: {str(e)}")
            return False, 0.0
    
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
            threshold: Similarity threshold
            
        Returns:
            Tuple of (best_match_label, best_similarity)
        """
        try:
            if not reference_embeddings or not reference_labels:
                return None, 0.0
            
            best_similarity = 0.0
            best_label = None
            
            for embedding, label in zip(reference_embeddings, reference_labels):
                is_same, similarity = self.compare_faces(query_embedding, embedding, threshold)
                if is_same and similarity > best_similarity:
                    best_similarity = similarity
                    best_label = label
            
            return best_label, best_similarity
            
        except Exception as e:
            self.logger.error(f"Error finding best match: {str(e)}")
            return None, 0.0
    
    def detect_and_encode_faces(self, image: np.ndarray) -> List[Tuple[dict, np.ndarray]]:
        """
        Detect faces in image and return their encodings with metadata
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of tuples (face_info, face_encoding)
        """
        try:
            # Get face analysis
            faces = self.app.get(image)
            
            results = []
            for face in faces:
                # Extract face information
                face_info = {
                    'bbox': face.bbox,  # [x1, y1, x2, y2]
                    'kps': face.kps,    # 5 keypoints
                    'det_score': face.det_score,  # Detection confidence
                    'age': getattr(face, 'age', None),
                    'gender': getattr(face, 'gender', None)
                }
                
                results.append((face_info, face.embedding))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error detecting and encoding faces: {str(e)}")
            return []
    
    def get_face_analysis(self, image: np.ndarray) -> List[dict]:
        """
        Get comprehensive face analysis including detection, landmarks, age, gender
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face analysis dictionaries
        """
        try:
            faces = self.app.get(image)
            
            results = []
            for face in faces:
                face_info = {
                    'bbox': face.bbox.tolist(),  # [x1, y1, x2, y2]
                    'kps': face.kps.tolist(),    # 5 keypoints
                    'det_score': float(face.det_score),  # Detection confidence
                    'embedding': face.embedding.tolist(),  # Face embedding
                    'age': getattr(face, 'age', None),
                    'gender': getattr(face, 'gender', None)
                }
                results.append(face_info)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting face analysis: {str(e)}")
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
            'name': 'InsightFaceEmbedder',
            'library': 'insightface',
            'model': self.model_name,
            'embedding_size': self.embedding_size,
            'distance_metric': 'cosine',
            'description': f'State-of-the-art face embeddings using InsightFace {self.model_name} model',
            'features': ['face_detection', 'face_embedding', 'age_estimation', 'gender_classification', 'facial_landmarks']
        }
    
    def release(self):
        """Release resources"""
        try:
            if hasattr(self, 'app'):
                del self.app
        except Exception as e:
            self.logger.error(f"Error releasing InsightFace embedder: {str(e)}")
    
    def __del__(self):
        """Destructor"""
        self.release()

