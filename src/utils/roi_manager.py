import numpy as np
import cv2
from typing import List, Tuple, Dict, Any

class ROIManager:
    def __init__(self, roi_points: List[Tuple[int, int]] = None):
        """Initialize ROI manager with simplified logic to avoid Shapely dependency."""
        self.roi_points = roi_points or [(400, 300), (1520, 300), (1520, 780), (400, 780)]
        self.roi_polygon = np.array(self.roi_points, dtype=np.int32)
        
        # Calculate ROI area
        self.roi_area = cv2.contourArea(self.roi_polygon)
        
        print(f"ROI initialized with {len(self.roi_points)} points")
        print(f"ROI area: {int(self.roi_area)} pixels")
        
        # Track object states for entry/exit detection
        self.object_states = {}  # {track_id: {'in_roi': bool, 'last_frame': int}}
    
    def check_roi_crossings(self, tracked_objects: List[Dict], frame_number: int) -> List[Dict]:
        """Check if objects cross ROI boundaries and generate events."""
        events = []
        
        for obj in tracked_objects:
            track_id = obj.get('track_id', -1)
            if track_id == -1:
                continue
                
            # Get object center point
            bbox = obj['bbox']
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            center_point = (center_x, center_y)
            
            # Check if point is in ROI
            is_in_roi = self.is_point_in_roi(center_point)
            
            # Initialize object state if new
            if track_id not in self.object_states:
                self.object_states[track_id] = {
                    'in_roi': is_in_roi,
                    'last_frame': frame_number
                }
                
                # Generate entry event if object starts in ROI
                if is_in_roi:
                    events.append({
                        'track_id': track_id,
                        'event_type': 'ENTER',
                        'frame_number': frame_number,
                        'center': center_point,
                        'bbox': bbox,
                        'object_class': obj.get('class', 'unknown'),
                        'confidence': obj.get('confidence', 0.0)
                    })
            else:
                # Check for state change
                prev_state = self.object_states[track_id]['in_roi']
                
                if prev_state != is_in_roi:
                    event_type = 'ENTER' if is_in_roi else 'EXIT'
                    events.append({
                        'track_id': track_id,
                        'event_type': event_type,
                        'frame_number': frame_number,
                        'center': center_point,
                        'bbox': bbox,
                        'object_class': obj.get('class', 'unknown'),
                        'confidence': obj.get('confidence', 0.0)
                    })
                
                # Update state
                self.object_states[track_id].update({
                    'in_roi': is_in_roi,
                    'last_frame': frame_number
                })
        
        # Clean up old object states (objects not seen for 30 frames)
        current_track_ids = {obj.get('track_id', -1) for obj in tracked_objects}
        stale_ids = []
        for track_id, state in self.object_states.items():
            if (track_id not in current_track_ids and 
                frame_number - state['last_frame'] > 30):
                stale_ids.append(track_id)
        
        for track_id in stale_ids:
            del self.object_states[track_id]
        
        return events
    
    def is_point_in_roi(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside ROI using OpenCV."""
        try:
            result = cv2.pointPolygonTest(self.roi_polygon, point, False)
            return result >= 0  # >= 0 means inside or on the boundary
        except Exception:
            # Fallback: simple bounding box check
            x, y = point
            min_x = min(p[0] for p in self.roi_points)
            max_x = max(p[0] for p in self.roi_points)
            min_y = min(p[1] for p in self.roi_points)
            max_y = max(p[1] for p in self.roi_points)
            
            return min_x <= x <= max_x and min_y <= y <= max_y
    
    def draw_roi(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROI polygon on frame."""
        try:
            # Draw ROI polygon
            cv2.polylines(frame, [self.roi_polygon], True, (0, 255, 0), 3)
            
            # Add ROI label
            cv2.putText(frame, "ROI", 
                       (self.roi_points[0][0], self.roi_points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return frame
        except Exception as e:
            print(f"Warning: Could not draw ROI: {e}")
            return frame
    
    def get_roi_info(self) -> Dict:
        """Get ROI information."""
        return {
            'points': self.roi_points,
            'area': int(self.roi_area),
            'active_objects': len(self.object_states)
        }