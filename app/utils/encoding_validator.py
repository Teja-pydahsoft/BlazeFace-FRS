"""
Face encoding validation utilities
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

class FaceEncodingValidator:
    """Validates face encodings for quality and uniqueness"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.min_internal_similarity = 0.70  # Minimum similarity within same person
        self.max_cross_similarity = 0.85     # Maximum similarity between different people
        self.min_encoding_norm = 0.1         # Minimum encoding norm
        self.max_encoding_norm = 10.0        # Maximum encoding norm
        self.max_encodings_per_person = 5    # Maximum encodings per person
    
    def validate_encoding_quality(self, encoding: np.ndarray) -> Tuple[bool, str]:
        """
        Validate the quality of a single encoding
        
        Args:
            encoding: Face encoding to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check encoding properties
            norm = np.linalg.norm(encoding)
            mean_val = np.mean(encoding)
            std_val = np.std(encoding)
            
            # Check norm
            if norm < self.min_encoding_norm:
                return False, f"Encoding norm too low: {norm:.4f} < {self.min_encoding_norm}"
            if norm > self.max_encoding_norm:
                return False, f"Encoding norm too high: {norm:.4f} > {self.max_encoding_norm}"
            
            # Check for unusual patterns
            if abs(mean_val) > 1.0:
                return False, f"Encoding mean too high: {mean_val:.4f}"
            if std_val < 0.01:
                return False, f"Encoding std too low: {std_val:.4f}"
            if std_val > 2.0:
                return False, f"Encoding std too high: {std_val:.4f}"
            
            # Check for all zeros or all same values
            if np.all(encoding == 0):
                return False, "Encoding is all zeros"
            if np.all(encoding == encoding[0]):
                return False, "Encoding has all same values"
            
            return True, "Valid encoding"
            
        except Exception as e:
            return False, f"Error validating encoding: {str(e)}"
    
    def validate_person_encodings(self, encodings: List[np.ndarray]) -> Tuple[bool, str]:
        """
        Validate encodings for a single person
        
        Args:
            encodings: List of encodings for one person
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not encodings:
                return False, "No encodings provided"
            
            if len(encodings) > self.max_encodings_per_person:
                return False, f"Too many encodings: {len(encodings)} > {self.max_encodings_per_person}"
            
            # Validate each encoding
            for i, encoding in enumerate(encodings):
                is_valid, error_msg = self.validate_encoding_quality(encoding)
                if not is_valid:
                    return False, f"Encoding {i}: {error_msg}"
            
            # Check internal consistency
            if len(encodings) > 1:
                min_similarity = float('inf')
                for i in range(len(encodings)):
                    for j in range(i + 1, len(encodings)):
                        similarity = self._calculate_similarity(encodings[i], encodings[j])
                        min_similarity = min(min_similarity, similarity)
                
                if min_similarity < self.min_internal_similarity:
                    return False, f"Low internal similarity: {min_similarity:.4f} < {self.min_internal_similarity}"
            
            return True, "Valid person encodings"
            
        except Exception as e:
            return False, f"Error validating person encodings: {str(e)}"
    
    def validate_cross_person_similarity(self, person_encodings: Dict[str, List[np.ndarray]]) -> Tuple[bool, str, List[Tuple[str, str, float]]]:
        """
        Validate that different people have sufficiently different encodings
        
        Args:
            person_encodings: Dictionary mapping person_id to list of encodings
            
        Returns:
            Tuple of (is_valid, error_message, problematic_pairs)
        """
        try:
            person_ids = list(person_encodings.keys())
            problematic_pairs = []
            
            for i in range(len(person_ids)):
                for j in range(i + 1, len(person_ids)):
                    person1_id = person_ids[i]
                    person2_id = person_ids[j]
                    
                    enc1_list = person_encodings[person1_id]
                    enc2_list = person_encodings[person2_id]
                    
                    # Find maximum similarity between any encodings of these two people
                    max_similarity = 0.0
                    for enc1 in enc1_list:
                        for enc2 in enc2_list:
                            similarity = self._calculate_similarity(enc1, enc2)
                            max_similarity = max(max_similarity, similarity)
                    
                    if max_similarity > self.max_cross_similarity:
                        problematic_pairs.append((person1_id, person2_id, max_similarity))
            
            if problematic_pairs:
                error_msg = f"Found {len(problematic_pairs)} problematic pairs with similarity > {self.max_cross_similarity}"
                return False, error_msg, problematic_pairs
            
            return True, "No problematic cross-person similarities", []
            
        except Exception as e:
            return False, f"Error validating cross-person similarity: {str(e)}", []
    
    def recommend_encoding_cleanup(self, person_encodings: Dict[str, List[np.ndarray]]) -> Dict[str, List[int]]:
        """
        Recommend which encodings to keep for each person
        
        Args:
            person_encodings: Dictionary mapping person_id to list of encodings
            
        Returns:
            Dictionary mapping person_id to list of recommended encoding indices
        """
        try:
            recommendations = {}
            
            for person_id, encodings in person_encodings.items():
                if len(encodings) <= 3:
                    # Keep all if 3 or fewer
                    recommendations[person_id] = list(range(len(encodings)))
                else:
                    # Calculate average similarity for each encoding
                    encoding_scores = []
                    
                    for i, encoding in enumerate(encodings):
                        total_similarity = 0
                        count = 0
                        
                        for j, other_encoding in enumerate(encodings):
                            if i != j:
                                similarity = self._calculate_similarity(encoding, other_encoding)
                                total_similarity += similarity
                                count += 1
                        
                        avg_similarity = total_similarity / count if count > 0 else 0
                        encoding_scores.append((i, avg_similarity))
                    
                    # Sort by average similarity and keep top 3
                    encoding_scores.sort(key=lambda x: x[1], reverse=True)
                    recommendations[person_id] = [idx for idx, _ in encoding_scores[:3]]
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating cleanup recommendations: {str(e)}")
            return {}
    
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
    
    def validate_database_encodings(self, encodings: List[Tuple[str, np.ndarray, str]]) -> Dict[str, any]:
        """
        Comprehensive validation of all database encodings
        
        Args:
            encodings: List of (student_id, encoding, encoding_type) tuples
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Group encodings by student
            person_encodings = {}
            for student_id, encoding, encoding_type in encodings:
                if student_id not in person_encodings:
                    person_encodings[student_id] = []
                person_encodings[student_id].append(encoding)
            
            results = {
                'total_encodings': len(encodings),
                'total_students': len(person_encodings),
                'person_validation': {},
                'cross_person_validation': {},
                'recommendations': {}
            }
            
            # Validate each person's encodings
            for person_id, enc_list in person_encodings.items():
                is_valid, error_msg = self.validate_person_encodings(enc_list)
                results['person_validation'][person_id] = {
                    'is_valid': is_valid,
                    'error_message': error_msg,
                    'encoding_count': len(enc_list)
                }
            
            # Validate cross-person similarity
            is_valid, error_msg, problematic_pairs = self.validate_cross_person_similarity(person_encodings)
            results['cross_person_validation'] = {
                'is_valid': is_valid,
                'error_message': error_msg,
                'problematic_pairs': problematic_pairs
            }
            
            # Generate cleanup recommendations
            results['recommendations'] = self.recommend_encoding_cleanup(person_encodings)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error validating database encodings: {str(e)}")
            return {'error': str(e)}
