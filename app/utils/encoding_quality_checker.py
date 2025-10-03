"""
Encoding quality checker for registration process
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

class EncodingQualityChecker:
    """Checks encoding quality during registration to prevent issues"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.min_similarity_with_existing = 0.70  # Minimum similarity with existing encodings of same person
        self.max_similarity_with_others = 0.60    # Maximum similarity with encodings of different people
        self.duplicate_threshold = 0.85           # Threshold to detect potential duplicates
        self.min_encoding_quality = 0.60          # Minimum quality score for encoding
        self.max_encodings_per_person = 3         # Maximum encodings per person
    
    def check_new_encoding_quality(self, new_encoding: np.ndarray, 
                                 existing_encodings: List[np.ndarray],
                                 other_people_encodings: List[Tuple[str, np.ndarray]],
                                 person_id: str) -> Tuple[bool, str, dict]:
        """
        Check if a new encoding meets quality standards
        
        Args:
            new_encoding: The new encoding to check
            existing_encodings: Existing encodings for the same person
            other_people_encodings: Encodings for other people as a list of (student_id, encoding)
            person_id: ID of the person being registered
            
        Returns:
            Tuple of (is_acceptable, reason, quality_metrics)
        """
        try:
            quality_metrics = {}
            
            # Check encoding basic quality
            norm = np.linalg.norm(new_encoding)
            mean_val = np.mean(new_encoding)
            std_val = np.std(new_encoding)
            
            quality_metrics['norm'] = norm
            quality_metrics['mean'] = mean_val
            quality_metrics['std'] = std_val
            
            # Basic quality checks
            if norm < 0.1 or norm > 10:
                return False, f"Encoding norm out of range: {norm:.4f}", quality_metrics
            
            if abs(mean_val) > 1.0:
                return False, f"Encoding mean too high: {mean_val:.4f}", quality_metrics
            
            if std_val < 0.01 or std_val > 2.0:
                return False, f"Encoding std out of range: {std_val:.4f}", quality_metrics
            
            # Check similarity with existing encodings of same person
            if existing_encodings:
                min_similarity = float('inf')
                for existing_encoding in existing_encodings:
                    similarity = self._calculate_similarity(new_encoding, existing_encoding)
                    min_similarity = min(min_similarity, similarity)
                
                quality_metrics['min_similarity_with_self'] = min_similarity
                
                if min_similarity < self.min_similarity_with_existing:
                    return False, f"Too different from existing encodings: {min_similarity:.4f} < {self.min_similarity_with_existing}", quality_metrics
            
            # Check similarity with other people's encodings
            if other_people_encodings:
                max_similarity = 0.0
                most_similar_student = None
                for other_student_id, other_encoding in other_people_encodings:
                    similarity = self._calculate_similarity(new_encoding, other_encoding)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_student = other_student_id
                
                quality_metrics['max_similarity_with_others'] = max_similarity
                
                if max_similarity > self.duplicate_threshold:
                    return False, f"Potential duplicate of student {most_similar_student} (similarity: {max_similarity:.2f})", quality_metrics
                
                if max_similarity > self.max_similarity_with_others:
                    return False, f"Too similar to other people: {max_similarity:.4f} > {self.max_similarity_with_others}", quality_metrics
            
            # Check if person already has too many encodings
            if len(existing_encodings) >= self.max_encodings_per_person:
                return False, f"Person already has maximum encodings: {len(existing_encodings)} >= {self.max_encodings_per_person}", quality_metrics
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(new_encoding, existing_encodings, other_people_encodings)
            quality_metrics['quality_score'] = quality_score
            
            if quality_score < self.min_encoding_quality:
                return False, f"Encoding quality too low: {quality_score:.4f} < {self.min_encoding_quality}", quality_metrics
            
            return True, "Encoding meets quality standards", quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error checking encoding quality: {str(e)}")
            return False, f"Error checking quality: {str(e)}", {}
    
    def _calculate_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """Calculate cosine similarity between two encodings"""
        try:
            dot_product = np.dot(encoding1, encoding2)
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def _calculate_quality_score(self, new_encoding: np.ndarray,
                               existing_encodings: List[np.ndarray],
                               other_people_encodings: List[Tuple[str, np.ndarray]]) -> float:
        """Calculate overall quality score for an encoding"""
        try:
            score = 1.0
            
            # Penalize if too similar to other people
            if other_people_encodings:
                max_similarity = 0.0
                for _, other_encoding in other_people_encodings:
                    similarity = self._calculate_similarity(new_encoding, other_encoding)
                    max_similarity = max(max_similarity, similarity)
                
                # Penalty for high similarity with others
                if max_similarity > 0.7:
                    score -= (max_similarity - 0.7) * 2  # Strong penalty
            
            # Reward good similarity with self (if existing encodings)
            if existing_encodings:
                avg_similarity = 0.0
                for existing_encoding in existing_encodings:
                    similarity = self._calculate_similarity(new_encoding, existing_encoding)
                    avg_similarity += similarity
                avg_similarity /= len(existing_encodings)
                
                # Reward for good similarity with self
                if avg_similarity > 0.8:
                    score += (avg_similarity - 0.8) * 0.5  # Small reward
            
            # Check encoding properties
            norm = np.linalg.norm(new_encoding)
            if norm < 0.5 or norm > 2.0:
                score -= 0.2  # Penalty for unusual norm
            
            std_val = np.std(new_encoding)
            if std_val < 0.05 or std_val > 0.5:
                score -= 0.1  # Penalty for unusual std
            
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {str(e)}")
            return 0.0
    
    def recommend_registration_settings(self, person_id: str, 
                                      existing_encodings: List[np.ndarray]) -> dict:
        """
        Recommend registration settings based on existing data
        
        Args:
            person_id: ID of the person
            existing_encodings: Existing encodings for this person
            
        Returns:
            Dictionary with recommendations
        """
        try:
            recommendations = {
                'can_register': True,
                'recommended_encodings': 3,
                'quality_threshold': 0.60,
                'warnings': [],
                'suggestions': []
            }
            
            if not existing_encodings:
                recommendations['suggestions'].append("First registration - ensure good lighting and clear face")
                return recommendations
            
            # Check existing encoding quality
            if len(existing_encodings) >= self.max_encodings_per_person:
                recommendations['can_register'] = False
                recommendations['warnings'].append(f"Person already has maximum encodings ({len(existing_encodings)})")
                return recommendations
            
            # Check internal consistency
            if len(existing_encodings) > 1:
                min_similarity = float('inf')
                for i in range(len(existing_encodings)):
                    for j in range(i + 1, len(existing_encodings)):
                        similarity = self._calculate_similarity(existing_encodings[i], existing_encodings[j])
                        min_similarity = min(min_similarity, similarity)
                
                if min_similarity < 0.7:
                    recommendations['warnings'].append(f"Low internal consistency: {min_similarity:.4f}")
                    recommendations['suggestions'].append("Consider re-registering with better face detection")
                elif min_similarity > 0.95:
                    recommendations['warnings'].append(f"Very high internal consistency: {min_similarity:.4f}")
                    recommendations['suggestions'].append("Encodings may be too similar - try different angles/lighting")
            
            # Calculate remaining slots
            remaining_slots = self.max_encodings_per_person - len(existing_encodings)
            recommendations['recommended_encodings'] = min(remaining_slots, 2)  # Max 2 new encodings
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return {'can_register': False, 'error': str(e)}