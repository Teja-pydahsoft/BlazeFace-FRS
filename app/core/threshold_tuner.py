"""
Threshold Tuning System for Face Recognition
Provides dynamic threshold adjustment and validation capabilities
"""

import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime
import sqlite3

class ThresholdTuner:
    """
    Dynamic threshold tuning system for face recognition
    Validates and optimizes recognition thresholds based on performance metrics
    """
    
    def __init__(self, config: Dict[str, Any], database_manager=None):
        """
        Initialize threshold tuner
        
        Args:
            config: Configuration dictionary
            database_manager: Database manager instance
        """
        self.config = config
        self.database_manager = database_manager
        self.logger = logging.getLogger(__name__)
        
        # Get threshold configuration
        self.threshold_config = config.get('threshold_tuning', {})
        self.auto_tune_enabled = self.threshold_config.get('auto_tune_enabled', False)
        self.validation_dataset_path = self.threshold_config.get('validation_dataset_path', 'validation_data/')
        
        # Embedder-specific thresholds
        self.embedder_thresholds = {
            'insightface': self.threshold_config.get('insightface_threshold', 0.5),
            'standard_face': self.threshold_config.get('dlib_threshold', 0.6),
            'facenet': self.threshold_config.get('facenet_threshold', 0.6),
            'simple': self.threshold_config.get('simple_threshold', 0.8)
        }
        
        # Performance tracking
        self.performance_history = []
        
    def get_threshold_for_embedder(self, embedder_type: str) -> float:
        """
        Get appropriate threshold for embedder type
        
        Args:
            embedder_type: Type of embedder ('insightface', 'standard_face', etc.)
            
        Returns:
            Threshold value for the embedder
        """
        return self.embedder_thresholds.get(embedder_type.lower(), 0.6)
    
    def validate_thresholds_with_dataset(self, validation_dataset: List[Dict[str, Any]], 
                                       embedder_type: str) -> Dict[str, Any]:
        """
        Validate thresholds using a test dataset
        
        Args:
            validation_dataset: List of test cases with ground truth
            embedder_type: Type of embedder to test
            
        Returns:
            Performance metrics dictionary
        """
        try:
            self.logger.info(f"Validating thresholds for {embedder_type} with {len(validation_dataset)} test cases")
            
            # Test different threshold values
            threshold_range = np.arange(0.3, 0.9, 0.05)
            best_metrics = None
            best_threshold = self.embedder_thresholds[embedder_type]
            
            for threshold in threshold_range:
                metrics = self._evaluate_threshold(validation_dataset, threshold, embedder_type)
                
                if best_metrics is None or metrics['f1_score'] > best_metrics['f1_score']:
                    best_metrics = metrics
                    best_threshold = threshold
            
            # Update configuration if auto-tuning is enabled
            if self.auto_tune_enabled and best_metrics:
                self.embedder_thresholds[embedder_type] = best_threshold
                self._save_threshold_config()
                
                # Log performance metrics to database
                if self.database_manager:
                    self._log_performance_metrics(embedder_type, best_threshold, best_metrics, len(validation_dataset))
            
            return {
                'embedder_type': embedder_type,
                'best_threshold': best_threshold,
                'metrics': best_metrics,
                'test_date': datetime.now().strftime('%Y-%m-%d'),
                'dataset_size': len(validation_dataset)
            }
            
        except Exception as e:
            self.logger.error(f"Error validating thresholds: {str(e)}")
            return {}
    
    def _evaluate_threshold(self, test_cases: List[Dict[str, Any]], 
                          threshold: float, embedder_type: str) -> Dict[str, float]:
        """
        Evaluate a specific threshold value
        
        Args:
            test_cases: List of test cases
            threshold: Threshold value to test
            embedder_type: Type of embedder
            
        Returns:
            Performance metrics (accuracy, precision, recall, f1)
        """
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for test_case in test_cases:
            query_embedding = test_case.get('query_embedding')
            reference_embedding = test_case.get('reference_embedding')
            ground_truth = test_case.get('is_same_person', False)
            
            if query_embedding is None or reference_embedding is None:
                continue
            
            # Calculate similarity (this would use the actual embedder)
            # For now, we'll simulate the comparison
            similarity = self._calculate_similarity(query_embedding, reference_embedding, embedder_type)
            predicted_same = similarity >= threshold
            
            # Update confusion matrix
            if ground_truth and predicted_same:
                true_positives += 1
            elif not ground_truth and predicted_same:
                false_positives += 1
            elif not ground_truth and not predicted_same:
                true_negatives += 1
            elif ground_truth and not predicted_same:
                false_negatives += 1
        
        # Calculate metrics
        total = true_positives + false_positives + true_negatives + false_negatives
        if total == 0:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
        
        accuracy = (true_positives + true_negatives) / total
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                            embedder_type: str) -> float:
        """
        Calculate similarity between two embeddings based on embedder type
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            embedder_type: Type of embedder
            
        Returns:
            Similarity score
        """
        try:
            if embedder_type == 'insightface':
                # InsightFace uses cosine similarity
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
            else:
                # Other embedders use Euclidean distance converted to similarity
                distance = np.linalg.norm(embedding1 - embedding2)
                return max(0.0, 1.0 - distance)  # Convert distance to similarity
                
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def _save_threshold_config(self):
        """Save updated threshold configuration"""
        try:
            # Update config dictionary
            for embedder_type, threshold in self.embedder_thresholds.items():
                config_key = f"{embedder_type}_threshold"
                if embedder_type == 'standard_face':
                    config_key = 'dlib_threshold'
                
                self.threshold_config[config_key] = threshold
            
            # Save to file
            config_file = 'app/config.json'
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                config_data['threshold_tuning'].update(self.threshold_config)
                
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=4)
                
                self.logger.info("Threshold configuration updated")
                
        except Exception as e:
            self.logger.error(f"Error saving threshold config: {str(e)}")
    
    def _log_performance_metrics(self, embedder_type: str, threshold: float, 
                               metrics: Dict[str, float], dataset_size: int):
        """Log performance metrics to database"""
        try:
            if self.database_manager and hasattr(self.database_manager, 'log_performance_metrics'):
                # This would be implemented in database manager
                pass
        except Exception as e:
            self.logger.error(f"Error logging performance metrics: {str(e)}")
    
    def create_validation_dataset_from_logs(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Create validation dataset from recognition logs
        
        Args:
            days_back: Number of days back to include in dataset
            
        Returns:
            List of validation test cases
        """
        try:
            if not self.database_manager:
                return []
            
            # Get recognition logs from database
            logs = self.database_manager.get_recognition_logs(
                date_from=(datetime.now().date().replace(day=datetime.now().day - days_back)).strftime('%Y-%m-%d')
            )
            
            validation_cases = []
            for log in logs:
                # This would require additional data structure to store ground truth
                # For now, return empty list
                pass
            
            return validation_cases
            
        except Exception as e:
            self.logger.error(f"Error creating validation dataset: {str(e)}")
            return []
    
    def get_performance_summary(self, embedder_type: str = None) -> Dict[str, Any]:
        """
        Get performance summary for embedders
        
        Args:
            embedder_type: Specific embedder type (optional)
            
        Returns:
            Performance summary dictionary
        """
        try:
            summary = {
                'current_thresholds': self.embedder_thresholds.copy(),
                'auto_tune_enabled': self.auto_tune_enabled,
                'last_validation': None,
                'recommendations': []
            }
            
            # Add recommendations based on current thresholds
            for embedder, threshold in self.embedder_thresholds.items():
                if threshold < 0.4:
                    summary['recommendations'].append(f"{embedder}: Threshold may be too low, consider increasing")
                elif threshold > 0.8:
                    summary['recommendations'].append(f"{embedder}: Threshold may be too high, consider decreasing")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {str(e)}")
            return {}
    
    def adjust_threshold_dynamically(self, embedder_type: str, 
                                   recent_accuracy: float, target_accuracy: float = 0.95):
        """
        Dynamically adjust threshold based on recent performance
        
        Args:
            embedder_type: Type of embedder
            recent_accuracy: Recent accuracy score
            target_accuracy: Target accuracy to maintain
        """
        try:
            if not self.auto_tune_enabled:
                return
            
            current_threshold = self.embedder_thresholds[embedder_type]
            
            # Simple adjustment logic
            if recent_accuracy < target_accuracy - 0.05:  # Accuracy too low
                # Increase threshold to be more strict
                new_threshold = min(0.9, current_threshold + 0.05)
            elif recent_accuracy > target_accuracy + 0.05:  # Accuracy too high
                # Decrease threshold to be more lenient
                new_threshold = max(0.3, current_threshold - 0.05)
            else:
                return  # No adjustment needed
            
            self.embedder_thresholds[embedder_type] = new_threshold
            self._save_threshold_config()
            
            self.logger.info(f"Adjusted {embedder_type} threshold from {current_threshold:.3f} to {new_threshold:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error adjusting threshold dynamically: {str(e)}")
