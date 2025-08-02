import numpy as np
import cv2
from typing import List, Dict, Tuple
from collections import deque, Counter
import time

class AccuracyEnhancer:
    """Advanced accuracy enhancement system for object detection."""
    
    def __init__(self, frame_buffer_size: int = 10):
        self.frame_buffer_size = frame_buffer_size
        self.detection_history = {}  # {track_id: deque of detections}
        self.confidence_history = {}  # {track_id: deque of confidence scores}
        self.class_stability = {}    # {track_id: class voting history}
        self.bbox_smoothing = {}     # {track_id: bbox history for smoothing}
        
        # Accuracy metrics
        self.detection_stats = {
            'total_detections': 0,
            'stable_detections': 0,
            'enhanced_detections': 0,
            'filtered_false_positives': 0
        }
    
    def enhance_detections(self, detections: List[Dict], frame_number: int) -> Tuple[List[Dict], Dict]:
        """
        Enhance detection accuracy using multiple validation techniques.
        
        Returns:
            Tuple of (enhanced_detections, accuracy_metrics)
        """
        enhanced_detections = []
        
        for detection in detections:
            track_id = detection.get('track_id', -1)
            
            if track_id == -1:
                # No tracking ID, use as-is but with lower confidence
                detection['original_confidence'] = detection['confidence']
                detection['confidence'] *= 0.8  # Reduce confidence for untracked
                enhanced_detections.append(detection)
                continue
            
            # Apply multi-frame validation
            enhanced_detection = self._apply_multi_frame_validation(detection, frame_number)
            
            if enhanced_detection:
                enhanced_detections.append(enhanced_detection)
                self.detection_stats['enhanced_detections'] += 1
            else:
                self.detection_stats['filtered_false_positives'] += 1
        
        self.detection_stats['total_detections'] += len(detections)
        
        # Calculate current accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics()
        
        return enhanced_detections, accuracy_metrics
    
    def _apply_multi_frame_validation(self, detection: Dict, frame_number: int) -> Dict:
        """Apply temporal validation across multiple frames."""
        track_id = detection['track_id']
        
        # Initialize history for new tracks
        if track_id not in self.detection_history:
            self.detection_history[track_id] = deque(maxlen=self.frame_buffer_size)
            self.confidence_history[track_id] = deque(maxlen=self.frame_buffer_size)
            self.class_stability[track_id] = deque(maxlen=self.frame_buffer_size)
            self.bbox_smoothing[track_id] = deque(maxlen=self.frame_buffer_size)
        
        # Add current detection to history
        self.detection_history[track_id].append(detection.copy())
        self.confidence_history[track_id].append(detection['confidence'])
        self.class_stability[track_id].append(detection['class'])
        self.bbox_smoothing[track_id].append(detection['bbox'])
        
        # Skip validation if insufficient history
        if len(self.detection_history[track_id]) < 3:
            return detection
        
        # 1. Confidence smoothing
        smoothed_confidence = self._smooth_confidence(track_id)
        
        # 2. Class stability check  
        stable_class = self._get_stable_class(track_id)
        
        # 3. Bbox smoothing
        smoothed_bbox = self._smooth_bbox(track_id)
        
        # 4. Temporal consistency check
        if not self._check_temporal_consistency(track_id):
            return None  # Filter out inconsistent detection
        
        # Create enhanced detection
        enhanced_detection = detection.copy()
        enhanced_detection['original_confidence'] = detection['confidence']
        enhanced_detection['confidence'] = smoothed_confidence
        enhanced_detection['class'] = stable_class
        enhanced_detection['bbox'] = smoothed_bbox
        enhanced_detection['validation_score'] = self._calculate_validation_score(track_id)
        
        self.detection_stats['stable_detections'] += 1
        return enhanced_detection
    
    def _smooth_confidence(self, track_id: int) -> float:
        """Apply confidence smoothing using weighted average."""
        confidences = list(self.confidence_history[track_id])
        
        # Weight recent frames more heavily
        weights = np.exp(np.linspace(-1, 0, len(confidences)))
        weights /= weights.sum()
        
        smoothed = np.average(confidences, weights=weights)
        
        # Boost confidence if consistently detected
        if len(confidences) >= 5 and min(confidences) > 0.3:
            smoothed = min(smoothed * 1.1, 0.95)  # Cap at 95%
        
        return float(smoothed)
    
    def _get_stable_class(self, track_id: int) -> str:
        """Get most stable class prediction using voting."""
        class_votes = Counter(self.class_stability[track_id])
        most_common_class, vote_count = class_votes.most_common(1)[0]
        
        # Require majority vote for class stability
        total_votes = len(self.class_stability[track_id])
        if vote_count / total_votes >= 0.6:  # 60% consensus required
            return most_common_class
        else:
            # Return most recent if no consensus
            return list(self.class_stability[track_id])[-1]
    
    def _smooth_bbox(self, track_id: int) -> List[float]:
        """Apply bounding box smoothing to reduce jitter."""
        bboxes = np.array(list(self.bbox_smoothing[track_id]))
        
        # Use weighted average with more weight on recent frames
        weights = np.exp(np.linspace(-0.5, 0, len(bboxes)))
        weights /= weights.sum()
        
        smoothed_bbox = np.average(bboxes, axis=0, weights=weights)
        return smoothed_bbox.tolist()
    
    def _check_temporal_consistency(self, track_id: int) -> bool:
        """Check if detection is temporally consistent."""
        confidences = list(self.confidence_history[track_id])
        
        # Filter out detections with too low confidence
        if confidences[-1] < 0.1:  # Current detection too weak
            return False
        
        # Check for confidence trend
        if len(confidences) >= 5:
            recent_avg = np.mean(confidences[-3:])
            older_avg = np.mean(confidences[-5:-2])
            
            # Filter out detections with declining confidence
            if recent_avg < older_avg * 0.7:  # 30% drop
                return False
        
        return True
    
    def _calculate_validation_score(self, track_id: int) -> float:
        """Calculate overall validation score for detection."""
        scores = []
        
        # Confidence stability score
        confidences = list(self.confidence_history[track_id])
        if len(confidences) > 1:
            conf_std = np.std(confidences)
            conf_stability = max(0, 1 - conf_std)  # Lower std = higher stability
            scores.append(conf_stability)
        
        # Class stability score
        class_votes = Counter(self.class_stability[track_id])
        most_common_count = class_votes.most_common(1)[0][1]
        class_stability = most_common_count / len(self.class_stability[track_id])
        scores.append(class_stability)
        
        # Temporal consistency score
        temporal_score = len(self.detection_history[track_id]) / self.frame_buffer_size
        scores.append(temporal_score)
        
        return float(np.mean(scores))
    
    def _calculate_accuracy_metrics(self) -> Dict:
        """Calculate current accuracy metrics."""
        total = self.detection_stats['total_detections']
        if total == 0:
            return {'accuracy_rate': 0.0, 'enhancement_rate': 0.0, 'filtering_rate': 0.0}
        
        stable_rate = self.detection_stats['stable_detections'] / total
        enhancement_rate = self.detection_stats['enhanced_detections'] / total
        filtering_rate = self.detection_stats['filtered_false_positives'] / total
        
        return {
            'accuracy_rate': stable_rate,
            'enhancement_rate': enhancement_rate, 
            'filtering_rate': filtering_rate,
            'total_processed': total
        }
    
    def get_accuracy_display_text(self) -> List[str]:
        """Get accuracy metrics formatted for display."""
        metrics = self._calculate_accuracy_metrics()
        
        return [
            f"Accuracy Rate: {metrics['accuracy_rate']:.1%}",
            f"Enhanced: {metrics['enhancement_rate']:.1%}",
            f"Filtered FP: {metrics['filtering_rate']:.1%}",
            f"Total Processed: {metrics['total_processed']}"
        ]