"""
FaceNet embedding model implementation
Efficient face recognition using FaceNet embeddings
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import List, Optional, Tuple
import logging
import os

class FaceNetEmbedder:
    def __init__(self, model_path: str = None, input_size: Tuple[int, int] = (160, 160)):
        """
        Initialize FaceNet embedder
        
        Args:
            model_path: Path to FaceNet model file
            input_size: Input image size for the model
        """
        self.model_path = model_path
        self.input_size = input_size
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._create_default_model()
    
    def _create_default_model(self):
        """Create a default FaceNet-like model for demonstration"""
        try:
            # Set random seed for consistent model initialization
            tf.random.set_seed(42)
            np.random.seed(42)
            
            # Create a simple CNN model for face embeddings
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(*self.input_size, 3)),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(2),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(2),
                tf.keras.layers.Conv2D(128, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(2),
                tf.keras.layers.Conv2D(256, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation='linear')  # 128-dimensional embedding
            ])
            
            # Compile the model to initialize weights
            model.compile(optimizer='adam', loss='mse')
            
            self.model = model
            self.logger.info("Default FaceNet model created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating default model: {str(e)}")
            self.model = None
    
    def load_model(self, model_path: str):
        """
        Load FaceNet model from file
        
        Args:
            model_path: Path to the model file
        """
        try:
            if model_path.endswith('.h5'):
                self.model = tf.keras.models.load_model(model_path)
            elif model_path.endswith('.tflite'):
                # Load TFLite model
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                self.model = interpreter
            else:
                self.logger.warning(f"Unsupported model format: {model_path}")
                self._create_default_model()
                return
            
            self.logger.info(f"FaceNet model loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self._create_default_model()
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for FaceNet model
        
        Args:
            face_image: Input face image (BGR format)
            
        Returns:
            Preprocessed face image
        """
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            face_resized = cv2.resize(face_rgb, self.input_size)
            
            # Normalize pixel values to [0, 1]
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            return face_batch
            
        except Exception as e:
            self.logger.error(f"Error preprocessing face: {str(e)}")
            return None
    
    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding from face image
        
        Args:
            face_image: Input face image (BGR format)
            
        Returns:
            Face embedding vector or None if failed
        """
        try:
            if self.model is None:
                self.logger.error("Model not loaded")
                return None
            
            # Preprocess face
            processed_face = self.preprocess_face(face_image)
            if processed_face is None:
                return None
            
            # Get embedding
            if isinstance(self.model, tf.keras.Model):
                embedding = self.model.predict(processed_face, verbose=0)
                return embedding.flatten()
            else:
                # TFLite model
                input_details = self.model.get_input_details()
                output_details = self.model.get_output_details()
                
                self.model.set_tensor(input_details[0]['index'], processed_face)
                self.model.invoke()
                
                embedding = self.model.get_tensor(output_details[0]['index'])
                return embedding.flatten()
                
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
        Compare two face embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold
            
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
    
    def release(self):
        """Release resources"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
        except Exception as e:
            self.logger.error(f"Error releasing FaceNet embedder: {str(e)}")
    
    def __del__(self):
        """Destructor"""
        self.release()
