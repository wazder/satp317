import torch
from ultralytics import YOLO
import numpy as np
from typing import List, Dict
import cv2

from ..config import config

class YOLODetector:
    def __init__(self):
        self.model = YOLO(config.models.yolo_model)
        self.target_class_ids = list(config.target_class_ids.values())
        self.class_names = {v: k for k, v in config.target_class_ids.items()}
        
        # Ensure model is loaded
        print(f"YOLO model loaded: {config.models.yolo_model}")
        print(f"Target classes: {config.target_classes}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect target objects in frame using YOLO.
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            List of detection dictionaries with bbox, class, confidence
        """
        # Run YOLO inference
        results = self.model(
            frame,
            conf=config.models.yolo_confidence,
            iou=config.models.yolo_iou_threshold,
            verbose=False
        )
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    class_id = class_ids[i]
                    
                    # Only keep target classes
                    if class_id in self.target_class_ids:
                        detection = {
                            'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                            'confidence': float(confidences[i]),
                            'class': self.class_names[class_id],
                            'class_id': class_id
                        }
                        detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw YOLO detections on frame.
        
        Args:
            frame: Input image
            detections: List of detection dictionaries
            
        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class']
            confidence = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(output_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return output_frame