"""
Simple Face Embedder using OpenCV and basic feature extraction
This provides consistent embeddings for face recognition
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

class SimpleFaceEmbedder:
    def __init__(self):
        """Initialize simple face embedder"""
        self.logger = logging.getLogger(__name__)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.embedding_size = 128
        
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for embedding
        
        Args:
            face_image: Input face image (BGR format)
            
        Returns:
            Preprocessed face image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            resized = cv2.resize(gray, (128, 128))
            
            # Apply histogram equalization for better contrast
            equalized = cv2.equalizeHist(resized)
            
            # Normalize
            normalized = equalized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error preprocessing face: {str(e)}")
            return None
    
    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding from face image using simple feature extraction
        
        Args:
            face_image: Input face image (BGR format)
            
        Returns:
            Face embedding vector or None if failed
        """
        try:
            # Preprocess face
            processed_face = self.preprocess_face(face_image)
            if processed_face is None:
                return None
            
            # Extract features using multiple methods
            features = []
            
            # 1. Histogram features (more bins for better discrimination)
            hist = cv2.calcHist([processed_face], [0], None, [64], [0, 1])
            features.extend(hist.flatten())
            
            # 2. LBP-like features (simplified)
            lbp_features = self._extract_lbp_features(processed_face)
            features.extend(lbp_features)
            
            # 3. Gradient features
            grad_x = cv2.Sobel(processed_face, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(processed_face, cv2.CV_64F, 0, 1, ksize=3)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            grad_features = self._extract_region_features(grad_magnitude)
            features.extend(grad_features)
            
            # 4. Texture features
            texture_features = self._extract_texture_features(processed_face)
            features.extend(texture_features)
            
            # 5. Edge features
            edges = cv2.Canny((processed_face * 255).astype(np.uint8), 50, 150)
            edge_features = self._extract_region_features(edges.astype(np.float32) / 255.0)
            features.extend(edge_features)
            
            # 6. Local features (keypoint-like)
            local_features = self._extract_local_features(processed_face)
            features.extend(local_features)
            
            # 7. Frequency domain features (FFT)
            fft_features = self._extract_fft_features(processed_face)
            features.extend(fft_features)
            
            # 8. Geometric features
            geom_features = self._extract_geometric_features(processed_face)
            features.extend(geom_features)
            
            # Convert to numpy array and pad/truncate to embedding_size
            embedding = np.array(features, dtype=np.float32)
            
            # Pad or truncate to desired size
            if len(embedding) < self.embedding_size:
                padding = np.zeros(self.embedding_size - len(embedding))
                embedding = np.concatenate([embedding, padding])
            elif len(embedding) > self.embedding_size:
                embedding = embedding[:self.embedding_size]
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error getting face embedding: {str(e)}")
            return None
    
    def _extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Local Binary Pattern-like features with better discrimination"""
        try:
            # Enhanced LBP implementation
            features = []
            h, w = image.shape
            
            # Sample points in a grid with overlap for better coverage
            for i in range(0, h-8, 4):  # Overlapping regions
                for j in range(0, w-8, 4):
                    region = image[i:i+8, j:j+8]
                    if region.size == 64:  # 8x8 region
                        # Calculate local statistics
                        mean_val = np.mean(region)
                        std_val = np.std(region)
                        min_val = np.min(region)
                        max_val = np.max(region)
                        # Add texture measure
                        texture = np.var(region)
                        features.extend([mean_val, std_val, min_val, max_val, texture])
            
            return np.array(features[:40])  # Limit to 40 features
            
        except Exception as e:
            self.logger.error(f"Error extracting LBP features: {str(e)}")
            return np.zeros(40)
    
    def _extract_region_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from image regions"""
        try:
            features = []
            h, w = image.shape
            
            # Divide image into 4x4 grid
            for i in range(0, h, h//4):
                for j in range(0, w, w//4):
                    region = image[i:i+h//4, j:j+w//4]
                    if region.size > 0:
                        mean_val = np.mean(region)
                        std_val = np.std(region)
                        features.extend([mean_val, std_val])
            
            return np.array(features[:32])  # Limit to 32 features
            
        except Exception as e:
            self.logger.error(f"Error extracting region features: {str(e)}")
            return np.zeros(32)
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features"""
        try:
            features = []
            
            # Calculate co-occurrence matrix features (simplified)
            # Convert to integer values for co-occurrence
            img_int = (image * 255).astype(np.uint8)
            
            # Calculate horizontal and vertical differences
            diff_h = np.diff(img_int, axis=1)
            diff_v = np.diff(img_int, axis=0)
            
            # Calculate statistics
            features.extend([
                np.mean(diff_h),
                np.std(diff_h),
                np.mean(diff_v),
                np.std(diff_v)
            ])
            
            # Add more texture features
            features.extend([
                np.mean(image),
                np.std(image),
                np.min(image),
                np.max(image)
            ])
            
            return np.array(features[:32])  # Limit to 32 features
            
        except Exception as e:
            self.logger.error(f"Error extracting texture features: {str(e)}")
            return np.zeros(32)
    
    def _extract_local_features(self, image: np.ndarray) -> np.ndarray:
        """Extract local features (keypoint-like)"""
        try:
            features = []
            h, w = image.shape
            
            # Sample keypoints in a grid pattern
            keypoints = []
            for i in range(0, h, 16):  # Every 16 pixels
                for j in range(0, w, 16):
                    if i + 8 < h and j + 8 < w:
                        keypoints.append((i + 8, j + 8))
            
            # Extract features around each keypoint
            for y, x in keypoints[:16]:  # Limit to 16 keypoints
                # Extract 8x8 patch around keypoint
                patch = image[max(0, y-4):min(h, y+4), max(0, x-4):min(w, x+4)]
                if patch.size > 0:
                    # Calculate patch statistics
                    features.extend([
                        np.mean(patch),
                        np.std(patch),
                        np.var(patch)
                    ])
                else:
                    features.extend([0, 0, 0])
            
            return np.array(features[:48])  # Limit to 48 features
            
        except Exception as e:
            self.logger.error(f"Error extracting local features: {str(e)}")
            return np.zeros(48)
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                     threshold: float = 0.8) -> Tuple[bool, float]:
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
    
    def _extract_fft_features(self, image: np.ndarray) -> np.ndarray:
        """Extract frequency domain features using FFT"""
        try:
            # Apply FFT
            fft = np.fft.fft2(image)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Extract features from frequency domain
            features = []
            
            # Low frequency components (center region)
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            low_freq = magnitude[center_h-16:center_h+16, center_w-16:center_w+16]
            features.extend([np.mean(low_freq), np.std(low_freq), np.max(low_freq)])
            
            # High frequency components (corners)
            corners = [
                magnitude[:16, :16],  # Top-left
                magnitude[:16, -16:],  # Top-right
                magnitude[-16:, :16],  # Bottom-left
                magnitude[-16:, -16:]  # Bottom-right
            ]
            for corner in corners:
                features.extend([np.mean(corner), np.std(corner)])
            
            return np.array(features[:20])  # Limit to 20 features
            
        except Exception as e:
            self.logger.error(f"Error extracting FFT features: {str(e)}")
            return np.zeros(20)
    
    def _extract_geometric_features(self, image: np.ndarray) -> np.ndarray:
        """Extract geometric features"""
        try:
            features = []
            
            # Image moments
            moments = cv2.moments(image)
            if moments['m00'] != 0:
                # Central moments
                features.extend([
                    moments['m10'] / moments['m00'],  # Centroid x
                    moments['m01'] / moments['m00'],  # Centroid y
                    moments['mu20'] / moments['m00'],  # Variance x
                    moments['mu02'] / moments['m00'],  # Variance y
                    moments['mu11'] / moments['m00'],  # Covariance
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
            
            # Contour features
            contours, _ = cv2.findContours((image * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    features.append(circularity)
                else:
                    features.append(0)
            else:
                features.append(0)
            
            return np.array(features[:6])  # Limit to 6 features
            
        except Exception as e:
            self.logger.error(f"Error extracting geometric features: {str(e)}")
            return np.zeros(6)
    
    def release(self):
        """Release resources"""
        pass
    
    def __del__(self):
        """Destructor"""
        self.release()
