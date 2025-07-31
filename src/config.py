import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class VideoConfig:
    input_fps: int = 30
    target_fps: int = 15  # For debug mode
    resolution: Tuple[int, int] = (1920, 1080)
    output_format: str = "mp4"

@dataclass
class ModelConfig:
    yolo_model: str = "yolov8n.pt"
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    yolo_confidence: float = 0.5
    yolo_iou_threshold: float = 0.45
    sam_iou_threshold: float = 0.5

@dataclass
class TrackingConfig:
    tracker_type: str = "deepsort"  # or "bytetrack"
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3

@dataclass
class ROIConfig:
    roi_type: str = "polygon"  # or "rectangle"
    roi_points: List[Tuple[int, int]] = None
    line_thickness: int = 3
    color: Tuple[int, int, int] = (0, 255, 0)

@dataclass
class VisualizationConfig:
    show_yolo_boxes: bool = True
    show_sam_masks: bool = True
    show_tracking_ids: bool = True
    show_roi: bool = True
    show_object_info: bool = True
    mask_alpha: float = 0.3
    font_scale: float = 0.6
    font_thickness: int = 2

@dataclass
class LoggingConfig:
    output_csv: bool = True
    output_json: bool = True
    log_features: bool = True
    log_roi_events: bool = True

@dataclass
class Config:
    video: VideoConfig = VideoConfig()
    models: ModelConfig = ModelConfig()
    tracking: TrackingConfig = TrackingConfig()
    roi: ROIConfig = ROIConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Target classes for YOLO detection
    target_classes: List[str] = None
    target_class_ids: Dict[str, int] = None
    
    def __post_init__(self):
        if self.target_classes is None:
            self.target_classes = ["person", "backpack", "handbag", "suitcase"]
        
        if self.target_class_ids is None:
            # COCO class IDs for target objects
            self.target_class_ids = {
                "person": 0,
                "backpack": 24,
                "handbag": 26,
                "suitcase": 28
            }
        
        if self.roi.roi_points is None:
            # Default ROI - center area of 1080p frame
            self.roi.roi_points = [
                (400, 300),
                (1520, 300),
                (1520, 780),
                (400, 780)
            ]

# Global config instance
config = Config()