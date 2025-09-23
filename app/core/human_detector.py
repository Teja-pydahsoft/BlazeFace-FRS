"""
Human detection using MobileNet SSD
Optimized for real-time human detection
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional
import logging
import os

class HumanDetector:
    def __init__(self, 
                 model_path: str = None,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initialize human detector
        
        Args:
            model_path: Path to the detection model
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.logger = logging.getLogger(__name__)
        
        # COCO class names (person is class 0)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._create_default_detector()
    
    def _create_default_detector(self):
        """Create a default human detector using OpenCV DNN"""
        try:
            # Use OpenCV's DNN module with a pre-trained model
            # This is a fallback when no specific model is provided
            self.logger.info("Creating default human detector using OpenCV DNN")
            # Note: In a real implementation, you would load a pre-trained model here
            self.net = None  # Placeholder for actual model loading
            
        except Exception as e:
            self.logger.error(f"Error creating default detector: {str(e)}")
            self.net = None
    
    def load_model(self, model_path: str):
        """
        Load human detection model
        
        Args:
            model_path: Path to the model file
        """
        try:
            if model_path.endswith('.tflite'):
                # Load TFLite model
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.model_type = 'tflite'
                
            elif model_path.endswith('.pb') or model_path.endswith('.pbtxt'):
                # Load TensorFlow model
                self.net = cv2.dnn.readNetFromTensorflow(model_path)
                self.model_type = 'opencv_dnn'
                
            else:
                self.logger.warning(f"Unsupported model format: {model_path}")
                self._create_default_detector()
                return
            
            self.logger.info(f"Human detection model loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self._create_default_detector()
    
    def detect_humans(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect humans in the given frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of human bounding boxes as (x, y, width, height, confidence)
        """
        try:
            if self.net is None and not hasattr(self, 'interpreter'):
                return []
            
            h, w = frame.shape[:2]
            
            if hasattr(self, 'interpreter') and self.model_type == 'tflite':
                return self._detect_tflite(frame, h, w)
            elif self.net is not None and self.model_type == 'opencv_dnn':
                return self._detect_opencv_dnn(frame, h, w)
            else:
                return self._detect_simple(frame, h, w)
                
        except Exception as e:
            self.logger.error(f"Error in human detection: {str(e)}")
            return []
    
    def _detect_tflite(self, frame: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int, float]]:
        """Detect humans using TFLite model"""
        try:
            # Preprocess frame
            input_shape = self.input_details[0]['shape']
            input_height, input_width = input_shape[1], input_shape[2]
            
            frame_resized = cv2.resize(frame, (input_width, input_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frame_batch = np.expand_dims(frame_normalized, axis=0)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], frame_batch)
            self.interpreter.invoke()
            
            # Get outputs
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
            scores = self.interpreter.get_tensor(self.output_details[1]['index'])
            classes = self.interpreter.get_tensor(self.output_details[2]['index'])
            
            humans = []
            for i in range(len(scores[0])):
                if classes[0][i] == 0 and scores[0][i] > self.confidence_threshold:  # person class
                    y1, x1, y2, x2 = boxes[0][i]
                    x1 = int(x1 * w)
                    y1 = int(y1 * h)
                    x2 = int(x2 * w)
                    y2 = int(y2 * h)
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width > 0 and height > 0:
                        humans.append((x1, y1, width, height, scores[0][i]))
            
            return humans
            
        except Exception as e:
            self.logger.error(f"Error in TFLite detection: {str(e)}")
            return []
    
    def _detect_opencv_dnn(self, frame: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int, float]]:
        """Detect humans using OpenCV DNN"""
        try:
            # Create blob from frame
            blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (300, 300), (0, 0, 0), True, crop=False)
            
            # Set input
            self.net.setInput(blob)
            
            # Run inference
            detections = self.net.forward()
            
            humans = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > self.confidence_threshold:
                    class_id = int(detections[0, 0, i, 1])
                    
                    if class_id == 0:  # person class
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        
                        width = x2 - x1
                        height = y2 - y1
                        
                        if width > 0 and height > 0:
                            humans.append((x1, y1, width, height, confidence))
            
            return humans
            
        except Exception as e:
            self.logger.error(f"Error in OpenCV DNN detection: {str(e)}")
            return []
    
    def _detect_simple(self, frame: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int, float]]:
        """Simple human detection using basic computer vision techniques"""
        try:
            # This is a placeholder implementation
            # In a real scenario, you would implement a more sophisticated method
            humans = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply HOG descriptor for human detection
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Detect humans
            boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
            
            for i, (x, y, w, h) in enumerate(boxes):
                if weights[i] > self.confidence_threshold:
                    humans.append((x, y, w, h, weights[i]))
            
            return humans
            
        except Exception as e:
            self.logger.error(f"Error in simple detection: {str(e)}")
            return []
    
    def draw_humans(self, frame: np.ndarray, humans: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        Draw human bounding boxes on the frame
        
        Args:
            frame: Input image frame
            humans: List of human bounding boxes
            
        Returns:
            Frame with drawn human bounding boxes
        """
        try:
            for (x, y, w, h, confidence) in humans:
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Draw confidence score
                label = f"Human: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), (255, 0, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error drawing humans: {str(e)}")
            return frame
    
    def extract_human_region(self, frame: np.ndarray, human_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract human region from frame
        
        Args:
            frame: Input image frame
            human_box: Human bounding box (x, y, width, height)
            
        Returns:
            Extracted human region or None if extraction fails
        """
        try:
            x, y, w, h = human_box
            human_region = frame[y:y+h, x:x+w]
            
            if human_region.size > 0:
                return human_region
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting human region: {str(e)}")
            return None
    
    def release(self):
        """Release resources"""
        try:
            if hasattr(self, 'net') and self.net is not None:
                del self.net
            if hasattr(self, 'interpreter'):
                del self.interpreter
        except Exception as e:
            self.logger.error(f"Error releasing human detector: {str(e)}")
    
    def __del__(self):
        """Destructor"""
        self.release()
