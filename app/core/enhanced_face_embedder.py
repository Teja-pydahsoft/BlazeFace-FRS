"""
Enhanced Face Embedder with Better Discrimination
Focuses on specific facial characteristics for better person differentiation
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List

class EnhancedFaceEmbedder:
    def __init__(self):
        """Initialize enhanced face embedder"""
        self.logger = logging.getLogger(__name__)
        self.embedding_size = 128
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for feature extraction"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            resized = cv2.resize(gray, (200, 200))
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(resized)
            
            # Normalize
            normalized = equalized.astype(np.float32) / 255.0
            
            return normalized
        except Exception as e:
            self.logger.error(f"Error preprocessing face: {str(e)}")
            return None
    
    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding based on enhanced facial features"""
        try:
            processed_face = self.preprocess_face(face_image)
            if processed_face is None:
                return None
            
            features = []
            
            # 1. Face shape and proportions
            shape_features = self._extract_face_shape_features(processed_face)
            features.extend(shape_features)
            
            # 2. Eye region characteristics
            eye_features = self._extract_enhanced_eye_features(processed_face)
            features.extend(eye_features)
            
            # 3. Nose region characteristics
            nose_features = self._extract_enhanced_nose_features(processed_face)
            features.extend(nose_features)
            
            # 4. Mouth region characteristics
            mouth_features = self._extract_enhanced_mouth_features(processed_face)
            features.extend(mouth_features)
            
            # 5. Jawline and chin characteristics
            jawline_features = self._extract_enhanced_jawline_features(processed_face)
            features.extend(jawline_features)
            
            # 6. Forehead characteristics
            forehead_features = self._extract_enhanced_forehead_features(processed_face)
            features.extend(forehead_features)
            
            # 7. Cheek characteristics
            cheek_features = self._extract_enhanced_cheek_features(processed_face)
            features.extend(cheek_features)
            
            # 8. Facial symmetry analysis
            symmetry_features = self._extract_enhanced_symmetry_features(processed_face)
            features.extend(symmetry_features)
            
            # 9. Texture and skin characteristics
            texture_features = self._extract_enhanced_texture_features(processed_face)
            features.extend(texture_features)
            
            # 10. Edge and contour characteristics
            edge_features = self._extract_enhanced_edge_features(processed_face)
            features.extend(edge_features)
            
            # Convert to embedding
            embedding = np.array(features, dtype=np.float32)
            
            # Pad or truncate to desired size
            if len(embedding) < self.embedding_size:
                padding = np.zeros(self.embedding_size - len(embedding))
                embedding = np.concatenate([embedding, padding])
            elif len(embedding) > self.embedding_size:
                embedding = embedding[:self.embedding_size]
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error getting face embedding: {str(e)}")
            return None
    
    def _extract_face_shape_features(self, face: np.ndarray) -> np.ndarray:
        """Extract face shape and proportion features"""
        try:
            features = []
            h, w = face.shape
            
            # Face aspect ratio
            aspect_ratio = w / h
            features.append(aspect_ratio)
            
            # Face area
            face_area = np.sum(face > 0.1) / (w * h)
            features.append(face_area)
            
            # Face moments
            moments = cv2.moments(face)
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
            
            # Face contour features
            contours, _ = cv2.findContours((face * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            
            return np.array(features[:8])  # Limit to 8 features
            
        except Exception as e:
            self.logger.error(f"Error extracting face shape features: {str(e)}")
            return np.zeros(8)
    
    def _extract_enhanced_eye_features(self, face: np.ndarray) -> np.ndarray:
        """Extract enhanced eye region features"""
        try:
            features = []
            h, w = face.shape
            
            # Left and right eye regions
            left_eye_region = face[int(h*0.2):int(h*0.4), int(w*0.2):int(w*0.45)]
            right_eye_region = face[int(h*0.2):int(h*0.4), int(w*0.55):int(w*0.8)]
            
            for eye_region in [left_eye_region, right_eye_region]:
                if eye_region.size > 0:
                    # Eye shape features
                    features.extend([
                        np.mean(eye_region),
                        np.std(eye_region),
                        np.var(eye_region),
                        np.min(eye_region),
                        np.max(eye_region)
                    ])
                    
                    # Eye gradient features
                    grad_x = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(eye_region, cv2.CV_64F, 0, 1, ksize=3)
                    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                    
                    features.extend([
                        np.mean(grad_magnitude),
                        np.std(grad_magnitude),
                        np.max(grad_magnitude)
                    ])
                    
                    # Eye edge features
                    edges = cv2.Canny((eye_region * 255).astype(np.uint8), 50, 150)
                    features.extend([
                        np.mean(edges),
                        np.std(edges),
                        np.sum(edges > 0) / edges.size
                    ])
                else:
                    features.extend([0] * 11)  # Pad with zeros
            
            return np.array(features[:22])  # Limit to 22 features
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced eye features: {str(e)}")
            return np.zeros(22)
    
    def _extract_enhanced_nose_features(self, face: np.ndarray) -> np.ndarray:
        """Extract enhanced nose region features"""
        try:
            features = []
            h, w = face.shape
            
            # Nose region
            nose_region = face[int(h*0.35):int(h*0.65), int(w*0.4):int(w*0.6)]
            
            if nose_region.size > 0:
                # Nose shape features
                features.extend([
                    np.mean(nose_region),
                    np.std(nose_region),
                    np.var(nose_region),
                    np.min(nose_region),
                    np.max(nose_region)
                ])
                
                # Nose gradient features
                grad_x = cv2.Sobel(nose_region, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(nose_region, cv2.CV_64F, 0, 1, ksize=3)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                features.extend([
                    np.mean(grad_magnitude),
                    np.std(grad_magnitude),
                    np.max(grad_magnitude)
                ])
                
                # Nose edge features
                edges = cv2.Canny((nose_region * 255).astype(np.uint8), 50, 150)
                features.extend([
                    np.mean(edges),
                    np.std(edges),
                    np.sum(edges > 0) / edges.size
                ])
            else:
                features.extend([0] * 11)
            
            return np.array(features[:11])  # Limit to 11 features
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced nose features: {str(e)}")
            return np.zeros(11)
    
    def _extract_enhanced_mouth_features(self, face: np.ndarray) -> np.ndarray:
        """Extract enhanced mouth region features"""
        try:
            features = []
            h, w = face.shape
            
            # Mouth region
            mouth_region = face[int(h*0.6):int(h*0.8), int(w*0.3):int(w*0.7)]
            
            if mouth_region.size > 0:
                # Mouth shape features
                features.extend([
                    np.mean(mouth_region),
                    np.std(mouth_region),
                    np.var(mouth_region),
                    np.min(mouth_region),
                    np.max(mouth_region)
                ])
                
                # Mouth gradient features
                grad_x = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(mouth_region, cv2.CV_64F, 0, 1, ksize=3)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                features.extend([
                    np.mean(grad_magnitude),
                    np.std(grad_magnitude),
                    np.max(grad_magnitude)
                ])
                
                # Mouth edge features
                edges = cv2.Canny((mouth_region * 255).astype(np.uint8), 50, 150)
                features.extend([
                    np.mean(edges),
                    np.std(edges),
                    np.sum(edges > 0) / edges.size
                ])
            else:
                features.extend([0] * 11)
            
            return np.array(features[:11])  # Limit to 11 features
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced mouth features: {str(e)}")
            return np.zeros(11)
    
    def _extract_enhanced_jawline_features(self, face: np.ndarray) -> np.ndarray:
        """Extract enhanced jawline region features"""
        try:
            features = []
            h, w = face.shape
            
            # Jawline region
            jawline_region = face[int(h*0.7):int(h*0.9), int(w*0.2):int(w*0.8)]
            
            if jawline_region.size > 0:
                # Jawline shape features
                features.extend([
                    np.mean(jawline_region),
                    np.std(jawline_region),
                    np.var(jawline_region),
                    np.min(jawline_region),
                    np.max(jawline_region)
                ])
                
                # Jawline edge features
                edges = cv2.Canny((jawline_region * 255).astype(np.uint8), 50, 150)
                features.extend([
                    np.mean(edges),
                    np.std(edges),
                    np.sum(edges > 0) / edges.size
                ])
            else:
                features.extend([0] * 8)
            
            return np.array(features[:8])  # Limit to 8 features
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced jawline features: {str(e)}")
            return np.zeros(8)
    
    def _extract_enhanced_forehead_features(self, face: np.ndarray) -> np.ndarray:
        """Extract enhanced forehead region features"""
        try:
            features = []
            h, w = face.shape
            
            # Forehead region
            forehead_region = face[int(h*0.1):int(h*0.3), int(w*0.2):int(w*0.8)]
            
            if forehead_region.size > 0:
                # Forehead shape features
                features.extend([
                    np.mean(forehead_region),
                    np.std(forehead_region),
                    np.var(forehead_region),
                    np.min(forehead_region),
                    np.max(forehead_region)
                ])
                
                # Forehead texture features
                features.extend([
                    np.mean(forehead_region),
                    np.std(forehead_region),
                    np.var(forehead_region)
                ])
            else:
                features.extend([0] * 8)
            
            return np.array(features[:8])  # Limit to 8 features
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced forehead features: {str(e)}")
            return np.zeros(8)
    
    def _extract_enhanced_cheek_features(self, face: np.ndarray) -> np.ndarray:
        """Extract enhanced cheek region features"""
        try:
            features = []
            h, w = face.shape
            
            # Left and right cheek regions
            left_cheek = face[int(h*0.3):int(h*0.6), int(w*0.1):int(w*0.3)]
            right_cheek = face[int(h*0.3):int(h*0.6), int(w*0.7):int(w*0.9)]
            
            for cheek_region in [left_cheek, right_cheek]:
                if cheek_region.size > 0:
                    # Cheek shape features
                    features.extend([
                        np.mean(cheek_region),
                        np.std(cheek_region),
                        np.var(cheek_region),
                        np.min(cheek_region),
                        np.max(cheek_region)
                    ])
                else:
                    features.extend([0] * 5)
            
            return np.array(features[:10])  # Limit to 10 features
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced cheek features: {str(e)}")
            return np.zeros(10)
    
    def _extract_enhanced_symmetry_features(self, face: np.ndarray) -> np.ndarray:
        """Extract enhanced facial symmetry features"""
        try:
            features = []
            h, w = face.shape
            
            # Split face into left and right halves
            left_half = face[:, :w//2]
            right_half = face[:, w//2:]
            
            if left_half.size > 0 and right_half.size > 0:
                # Flip right half to compare with left
                right_half_flipped = np.fliplr(right_half)
                
                # Resize to same dimensions
                min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                left_half = left_half[:, :min_width]
                right_half_flipped = right_half_flipped[:, :min_width]
                
                # Calculate symmetry
                diff = np.abs(left_half - right_half_flipped)
                symmetry_score = 1.0 - np.mean(diff)
                features.append(symmetry_score)
                
                # Additional symmetry features
                features.extend([
                    np.std(diff),
                    np.max(diff),
                    np.mean(diff)
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            return np.array(features[:4])  # Limit to 4 features
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced symmetry features: {str(e)}")
            return np.zeros(4)
    
    def _extract_enhanced_texture_features(self, face: np.ndarray) -> np.ndarray:
        """Extract enhanced texture features"""
        try:
            features = []
            
            # Local Binary Pattern-like features
            lbp_features = self._extract_lbp_features(face)
            features.extend(lbp_features)
            
            # Gabor filter features
            gabor_features = self._extract_gabor_features(face)
            features.extend(gabor_features)
            
            return np.array(features[:16])  # Limit to 16 features
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced texture features: {str(e)}")
            return np.zeros(16)
    
    def _extract_enhanced_edge_features(self, face: np.ndarray) -> np.ndarray:
        """Extract enhanced edge features"""
        try:
            features = []
            
            # Canny edges
            edges = cv2.Canny((face * 255).astype(np.uint8), 50, 150)
            features.extend([
                np.mean(edges),
                np.std(edges),
                np.sum(edges > 0) / edges.size
            ])
            
            # Laplacian edges
            laplacian = cv2.Laplacian(face, cv2.CV_64F)
            features.extend([
                np.mean(laplacian),
                np.std(laplacian),
                np.var(laplacian)
            ])
            
            return np.array(features[:6])  # Limit to 6 features
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced edge features: {str(e)}")
            return np.zeros(6)
    
    def _extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Local Binary Pattern-like features"""
        try:
            features = []
            h, w = image.shape
            
            # Sample points in a grid
            for i in range(0, h-8, 8):
                for j in range(0, w-8, 8):
                    region = image[i:i+8, j:j+8]
                    if region.size == 64:  # 8x8 region
                        # Calculate local statistics
                        mean_val = np.mean(region)
                        std_val = np.std(region)
                        min_val = np.min(region)
                        max_val = np.max(region)
                        texture = np.var(region)
                        features.extend([mean_val, std_val, min_val, max_val, texture])
            
            return np.array(features[:10])  # Limit to 10 features
            
        except Exception as e:
            self.logger.error(f"Error extracting LBP features: {str(e)}")
            return np.zeros(10)
    
    def _extract_gabor_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Gabor filter features"""
        try:
            features = []
            
            # Apply Gabor filters with different orientations
            for theta in [0, 45, 90, 135]:
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 10, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                features.extend([
                    np.mean(filtered),
                    np.std(filtered)
                ])
            
            return np.array(features[:8])  # Limit to 8 features
            
        except Exception as e:
            self.logger.error(f"Error extracting Gabor features: {str(e)}")
            return np.zeros(8)
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                     threshold: float = 0.8) -> Tuple[bool, float]:
        """Compare two face embeddings using cosine similarity"""
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
    
    def release(self):
        """Release resources"""
        pass
    
    def __del__(self):
        self.release()
