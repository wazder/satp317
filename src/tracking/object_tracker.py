import numpy as np
from typing import List, Dict, Any
import supervision as sv

class ObjectTracker:
    def __init__(self):
        """Initialize ByteTracker from supervision."""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.2,
            lost_track_buffer=60,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        self.tracked_objects = {}
    
    def update(self, detections: List[Dict], frame_number: int) -> List[Dict]:
        """Update tracker with new detections."""
        if not detections:
            return []
        
        detection_list = []
        confidences = []
        class_ids = []
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            detection_list.append([x1, y1, x2, y2])
            confidences.append(det['confidence'])
            class_ids.append(0)
        
        if detection_list:
            detections_sv = sv.Detections(
                xyxy=np.array(detection_list),
                confidence=np.array(confidences),
                class_id=np.array(class_ids)
            )
            
            detections_sv = self.tracker.update_with_detections(detections_sv)
            
            tracked_objects = []
            for i, det in enumerate(detections):
                if hasattr(detections_sv, 'tracker_id') and i < len(detections_sv.tracker_id):
                    try:
                        det['track_id'] = int(detections_sv.tracker_id[i])
                    except (ValueError, TypeError):
                        det['track_id'] = -1
                else:
                    det['track_id'] = -1
                tracked_objects.append(det)
            
            return tracked_objects
        
        return []