import numpy as np
from typing import List, Dict, Optional
from deep_sort_realtime import DeepSort
import cv2

from ..config import config

class ObjectTracker:
    def __init__(self):
        self.tracker_type = config.tracking.tracker_type
        
        if self.tracker_type == "deepsort":
            self.tracker = DeepSort(
                max_age=config.tracking.max_age,
                n_init=config.tracking.min_hits,
                nn_budget=100,
                max_iou_distance=config.tracking.iou_threshold,
                max_cosine_distance=0.2,
                embedder="mobilenet",
                half=True,
                bgr=True
            )
        else:
            # Fallback to simple tracker if DeepSORT fails
            self.tracker = SimpleTracker()
            
        self.track_history = {}
        print(f"Object tracker initialized: {self.tracker_type}")
    
    def update(self, detections: List[Dict], frame_number: int) -> List[Dict]:
        """
        Update tracker with new detections and return tracked objects.
        
        Args:
            detections: List of detection dictionaries
            frame_number: Current frame number
            
        Returns:
            List of tracked objects with IDs and tracking info
        """
        if not detections:
            return []
        
        try:
            # Prepare detections for tracker
            tracker_detections = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                confidence = det['confidence']
                class_name = det['class']
                
                # DeepSORT expects [x1, y1, x2, y2, confidence, class]
                tracker_detections.append([x1, y1, x2, y2, confidence, class_name])
            
            # Update tracker
            if hasattr(self.tracker, 'update_tracks'):
                # DeepSORT
                tracks = self.tracker.update_tracks(tracker_detections, frame=None)
            else:
                # Simple tracker fallback
                tracks = self.tracker.update(tracker_detections, frame_number)
            
            # Convert tracks to our format
            tracked_objects = []
            for i, track in enumerate(tracks):
                if hasattr(track, 'is_confirmed') and not track.is_confirmed():
                    continue
                
                if hasattr(track, 'track_id'):
                    track_id = track.track_id
                    bbox = track.to_ltrb()  # [left, top, right, bottom]
                    
                    # Get original detection data
                    original_det = detections[min(i, len(detections)-1)]
                    
                    tracked_obj = {
                        'track_id': track_id,
                        'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                        'class': original_det.get('class', 'unknown'),
                        'confidence': original_det.get('confidence', 0.0),
                        'mask': original_det.get('mask'),
                        'features': original_det.get('features', {}),
                        'frame_number': frame_number
                    }
                    
                    tracked_objects.append(tracked_obj)
                    
                    # Update track history
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    
                    self.track_history[track_id].append({
                        'frame': frame_number,
                        'bbox': tracked_obj['bbox'],
                        'center': self._get_bbox_center(tracked_obj['bbox'])
                    })
                    
                    # Keep only recent history
                    if len(self.track_history[track_id]) > 30:
                        self.track_history[track_id] = self.track_history[track_id][-30:]
            
            return tracked_objects
            
        except Exception as e:
            print(f"Error in tracking update: {e}")
            # Fallback: return detections with dummy IDs
            return self._create_fallback_tracks(detections, frame_number)
    
    def _get_bbox_center(self, bbox: List[float]) -> List[float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    
    def _create_fallback_tracks(self, detections: List[Dict], frame_number: int) -> List[Dict]:
        """Create fallback tracks when tracker fails."""
        tracked_objects = []
        for i, det in enumerate(detections):
            tracked_obj = det.copy()
            tracked_obj['track_id'] = f"fallback_{i}_{frame_number}"
            tracked_obj['frame_number'] = frame_number
            tracked_objects.append(tracked_obj)
        return tracked_objects
    
    def get_track_history(self, track_id: int, frames: int = 10) -> List[Dict]:
        """Get recent history for a track."""
        if track_id in self.track_history:
            return self.track_history[track_id][-frames:]
        return []


class SimpleTracker:
    """Simple fallback tracker using IoU matching."""
    
    def __init__(self):
        self.next_id = 1
        self.tracks = {}
        self.max_age = 30
        
    def update(self, detections: List[List], frame_number: int) -> List:
        """Update tracks with simple IoU matching."""
        if not detections:
            return []
        
        # Simple tracking using IoU
        current_tracks = []
        
        for det in detections:
            x1, y1, x2, y2, conf, class_name = det
            bbox = [x1, y1, x2, y2]
            
            # Find best matching existing track
            best_iou = 0
            best_track_id = None
            
            for track_id, track_data in self.tracks.items():
                if frame_number - track_data['last_frame'] > self.max_age:
                    continue
                
                iou = self._calculate_iou(bbox, track_data['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                track_id = best_track_id
                self.tracks[track_id].update({
                    'bbox': bbox,
                    'last_frame': frame_number,
                    'class': class_name,
                    'confidence': conf
                })
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'bbox': bbox,
                    'last_frame': frame_number,
                    'class': class_name,
                    'confidence': conf
                }
            
            # Create track object
            track = SimpleTrack(track_id, bbox, class_name, conf)
            current_tracks.append(track)
        
        return current_tracks
    
    def _calculate_iou(self, box1: List, box2: List) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class SimpleTrack:
    """Simple track object for fallback tracker."""
    
    def __init__(self, track_id: int, bbox: List, class_name: str, confidence: float):
        self.track_id = track_id
        self.bbox = bbox
        self.class_name = class_name
        self.confidence = confidence
    
    def to_ltrb(self):
        """Return bounding box in left-top-right-bottom format."""
        return self.bbox
    
    def is_confirmed(self):
        """Always return True for simple tracker."""
        return True