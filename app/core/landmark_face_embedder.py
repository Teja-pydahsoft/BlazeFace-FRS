"""
Facial Landmark-Based Face Embedder
Uses OpenCV's facial landmark detection to extract specific facial features
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List

class LandmarkFaceEmbedder:
    def __init__(self):
        """Initialize landmark-based face embedder"""
        self.logger = logging.getLogger(__name__)
        self.embedding_size = 128
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize facial landmark detector (using MediaPipe if available)
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            self.use_mediapipe = True
            self.logger.info("Using MediaPipe for facial landmarks")
        except ImportError:
            self.use_mediapipe = False
            self.logger.warning("MediaPipe not available, using basic feature extraction")
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for landmark detection"""
        try:
            # Convert to RGB for MediaPipe
            if self.use_mediapipe:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                return rgb_image
            else:
                # Convert to grayscale for basic processing
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                return gray
        except Exception as e:
            self.logger.error(f"Error preprocessing face: {str(e)}")
            return None
    
    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding based on facial landmarks"""
        try:
            processed_face = self.preprocess_face(face_image)
            if processed_face is None:
                return None
            
            if self.use_mediapipe:
                return self._get_mediapipe_embedding(processed_face)
            else:
                return self._get_basic_embedding(processed_face)
                
        except Exception as e:
            self.logger.error(f"Error getting face embedding: {str(e)}")
            return None
    
    def _get_mediapipe_embedding(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """Get embedding using MediaPipe facial landmarks"""
        try:
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
            
            # Get the first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract landmark coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Convert to numpy array
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # Extract specific facial features
            features = []
            
            # Eye features
            eye_features = self._extract_eye_landmarks(landmarks)
            features.extend(eye_features)
            
            # Nose features
            nose_features = self._extract_nose_landmarks(landmarks)
            features.extend(nose_features)
            
            # Mouth features
            mouth_features = self._extract_mouth_landmarks(landmarks)
            features.extend(mouth_features)
            
            # Jawline features
            jawline_features = self._extract_jawline_landmarks(landmarks)
            features.extend(jawline_features)
            
            # Eyebrow features
            eyebrow_features = self._extract_eyebrow_landmarks(landmarks)
            features.extend(eyebrow_features)
            
            # Overall face shape
            shape_features = self._extract_face_shape_landmarks(landmarks)
            features.extend(shape_features)
            
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
            self.logger.error(f"Error getting MediaPipe embedding: {str(e)}")
            return None
    
    def _get_basic_embedding(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """Get embedding using basic feature extraction"""
        try:
            features = []
            
            # Detect face
            faces = self.face_cascade.detectMultiScale(gray_image, 1.1, 4)
            if len(faces) == 0:
                return None
            
            x, y, w, h = faces[0]
            face_roi = gray_image[y:y+h, x:x+w]
            
            # Extract basic features
            # 1. Face aspect ratio
            aspect_ratio = w / h
            features.append(aspect_ratio)
            
            # 2. Face area
            face_area = np.sum(face_roi > 0) / (w * h)
            features.append(face_area)
            
            # 3. Eye region features (approximate)
            eye_region = face_roi[int(h*0.2):int(h*0.4), :]
            if eye_region.size > 0:
                features.extend([
                    np.mean(eye_region),
                    np.std(eye_region),
                    np.var(eye_region)
                ])
            else:
                features.extend([0, 0, 0])
            
            # 4. Nose region features (approximate)
            nose_region = face_roi[int(h*0.35):int(h*0.65), int(w*0.4):int(w*0.6)]
            if nose_region.size > 0:
                features.extend([
                    np.mean(nose_region),
                    np.std(nose_region),
                    np.var(nose_region)
                ])
            else:
                features.extend([0, 0, 0])
            
            # 5. Mouth region features (approximate)
            mouth_region = face_roi[int(h*0.6):int(h*0.8), int(w*0.3):int(w*0.7)]
            if mouth_region.size > 0:
                features.extend([
                    np.mean(mouth_region),
                    np.std(mouth_region),
                    np.var(mouth_region)
                ])
            else:
                features.extend([0, 0, 0])
            
            # 6. Edge features
            edges = cv2.Canny(face_roi, 50, 150)
            features.extend([
                np.mean(edges),
                np.std(edges),
                np.sum(edges > 0) / edges.size
            ])
            
            # 7. Texture features
            texture = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            features.append(texture)
            
            # 8. Histogram features
            hist = cv2.calcHist([face_roi], [0], None, [16], [0, 256])
            features.extend(hist.flatten())
            
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
            self.logger.error(f"Error getting basic embedding: {str(e)}")
            return None
    
    def _extract_eye_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract eye landmark features"""
        try:
            # MediaPipe eye landmark indices (simplified)
            # Left eye: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
            # Right eye: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
            
            # For now, use basic features from landmark coordinates
            features = []
            
            # Eye region statistics (using first 20 landmarks as eye region approximation)
            eye_landmarks = landmarks[:60]  # First 20 landmarks (x, y, z)
            
            if len(eye_landmarks) >= 6:  # At least 2 landmarks
                # Extract x, y, z coordinates
                x_coords = eye_landmarks[::3]
                y_coords = eye_landmarks[1::3]
                z_coords = eye_landmarks[2::3]
                
                # Eye shape features
                features.extend([
                    np.mean(x_coords),
                    np.std(x_coords),
                    np.mean(y_coords),
                    np.std(y_coords),
                    np.mean(z_coords),
                    np.std(z_coords)
                ])
                
                # Eye distance and angle features
                if len(x_coords) >= 2:
                    eye_width = np.max(x_coords) - np.min(x_coords)
                    eye_height = np.max(y_coords) - np.min(y_coords)
                    features.extend([eye_width, eye_height])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0] * 8)
            
            return np.array(features[:8])
            
        except Exception as e:
            self.logger.error(f"Error extracting eye landmarks: {str(e)}")
            return np.zeros(8)
    
    def _extract_nose_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract nose landmark features"""
        try:
            # Use landmarks 20-40 as nose region approximation
            nose_landmarks = landmarks[60:120]  # Next 20 landmarks
            
            features = []
            
            if len(nose_landmarks) >= 6:
                x_coords = nose_landmarks[::3]
                y_coords = nose_landmarks[1::3]
                z_coords = nose_landmarks[2::3]
                
                features.extend([
                    np.mean(x_coords),
                    np.std(x_coords),
                    np.mean(y_coords),
                    np.std(y_coords),
                    np.mean(z_coords),
                    np.std(z_coords)
                ])
                
                if len(x_coords) >= 2:
                    nose_width = np.max(x_coords) - np.min(x_coords)
                    nose_height = np.max(y_coords) - np.min(y_coords)
                    features.extend([nose_width, nose_height])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0] * 8)
            
            return np.array(features[:8])
            
        except Exception as e:
            self.logger.error(f"Error extracting nose landmarks: {str(e)}")
            return np.zeros(8)
    
    def _extract_mouth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract mouth landmark features"""
        try:
            # Use landmarks 40-60 as mouth region approximation
            mouth_landmarks = landmarks[120:180]  # Next 20 landmarks
            
            features = []
            
            if len(mouth_landmarks) >= 6:
                x_coords = mouth_landmarks[::3]
                y_coords = mouth_landmarks[1::3]
                z_coords = mouth_landmarks[2::3]
                
                features.extend([
                    np.mean(x_coords),
                    np.std(x_coords),
                    np.mean(y_coords),
                    np.std(y_coords),
                    np.mean(z_coords),
                    np.std(z_coords)
                ])
                
                if len(x_coords) >= 2:
                    mouth_width = np.max(x_coords) - np.min(x_coords)
                    mouth_height = np.max(y_coords) - np.min(y_coords)
                    features.extend([mouth_width, mouth_height])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0] * 8)
            
            return np.array(features[:8])
            
        except Exception as e:
            self.logger.error(f"Error extracting mouth landmarks: {str(e)}")
            return np.zeros(8)
    
    def _extract_jawline_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract jawline landmark features"""
        try:
            # Use landmarks 60-80 as jawline region approximation
            jawline_landmarks = landmarks[180:240]  # Next 20 landmarks
            
            features = []
            
            if len(jawline_landmarks) >= 6:
                x_coords = jawline_landmarks[::3]
                y_coords = jawline_landmarks[1::3]
                z_coords = jawline_landmarks[2::3]
                
                features.extend([
                    np.mean(x_coords),
                    np.std(x_coords),
                    np.mean(y_coords),
                    np.std(y_coords),
                    np.mean(z_coords),
                    np.std(z_coords)
                ])
                
                if len(x_coords) >= 2:
                    jawline_width = np.max(x_coords) - np.min(x_coords)
                    jawline_height = np.max(y_coords) - np.min(y_coords)
                    features.extend([jawline_width, jawline_height])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0] * 8)
            
            return np.array(features[:8])
            
        except Exception as e:
            self.logger.error(f"Error extracting jawline landmarks: {str(e)}")
            return np.zeros(8)
    
    def _extract_eyebrow_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract eyebrow landmark features"""
        try:
            # Use landmarks 80-100 as eyebrow region approximation
            eyebrow_landmarks = landmarks[240:300]  # Next 20 landmarks
            
            features = []
            
            if len(eyebrow_landmarks) >= 6:
                x_coords = eyebrow_landmarks[::3]
                y_coords = eyebrow_landmarks[1::3]
                z_coords = eyebrow_landmarks[2::3]
                
                features.extend([
                    np.mean(x_coords),
                    np.std(x_coords),
                    np.mean(y_coords),
                    np.std(y_coords),
                    np.mean(z_coords),
                    np.std(z_coords)
                ])
                
                if len(x_coords) >= 2:
                    eyebrow_width = np.max(x_coords) - np.min(x_coords)
                    eyebrow_height = np.max(y_coords) - np.min(y_coords)
                    features.extend([eyebrow_width, eyebrow_height])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0] * 8)
            
            return np.array(features[:8])
            
        except Exception as e:
            self.logger.error(f"Error extracting eyebrow landmarks: {str(e)}")
            return np.zeros(8)
    
    def _extract_face_shape_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract overall face shape features"""
        try:
            features = []
            
            # Use all landmarks for face shape
            x_coords = landmarks[::3]
            y_coords = landmarks[1::3]
            z_coords = landmarks[2::3]
            
            # Face dimensions
            face_width = np.max(x_coords) - np.min(x_coords)
            face_height = np.max(y_coords) - np.min(y_coords)
            face_depth = np.max(z_coords) - np.min(z_coords)
            
            features.extend([face_width, face_height, face_depth])
            
            # Face aspect ratios
            if face_height > 0:
                features.append(face_width / face_height)
            else:
                features.append(0)
            
            if face_depth > 0:
                features.append(face_width / face_depth)
            else:
                features.append(0)
            
            # Face center
            features.extend([
                np.mean(x_coords),
                np.mean(y_coords),
                np.mean(z_coords)
            ])
            
            return np.array(features[:8])
            
        except Exception as e:
            self.logger.error(f"Error extracting face shape landmarks: {str(e)}")
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
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
    
    def __del__(self):
        self.release()
