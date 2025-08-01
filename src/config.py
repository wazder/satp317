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
    yolo_confidence: float = 0.25  # Changed default to LineLogic default
    yolo_iou_threshold: float = 0.45
    sam_iou_threshold: float = 0.5
    # LineLogic specific model parameters
    yolo_imgsz: int = 1024

@dataclass
class TrackingConfig:
    tracker_type: str = "bytetrack"  # Changed default to LineLogic's ByteTrack
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    # LineLogic ByteTrack parameters
    track_activation_threshold: float = 0.2
    lost_track_buffer: int = 60

@dataclass
class ROIConfig:
    roi_type: str = "polygon"  # or "rectangle" or "lines" for LineLogic
    roi_points: List[Tuple[int, int]] = None
    line_thickness: int = 3
    color: Tuple[int, int, int] = (0, 255, 0)
    
    # LineLogic line crossing configuration
    use_line_crossing: bool = True
    line_positions: List[int] = None  # X coordinates for vertical lines
    line_height: int = 1080
    line_ids: List[int] = None

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
    # LineLogic specific logging
    log_line_crossings: bool = True
    log_frame_validation: bool = True
    export_results_summary: bool = True

@dataclass
class LineLogicConfig:
    """Configuration specific to LineLogic frame-based validation."""
    use_frame_logic: bool = True
    min_safe_time: float = 0.5
    min_uncertain_time: float = 0.28 
    min_very_brief_time: float = 0.17
    
    # Classification confidence levels
    safe_threshold_frames: int = 0  # Will be calculated from fps
    uncertain_threshold_frames: int = 0
    very_brief_threshold_frames: int = 0

@dataclass
class Config:
    video: VideoConfig = VideoConfig()
    models: ModelConfig = ModelConfig()
    tracking: TrackingConfig = TrackingConfig()
    roi: ROIConfig = ROIConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    logging: LoggingConfig = LoggingConfig()
    linelogic: LineLogicConfig = LineLogicConfig()
    
    # Target classes for YOLO detection
    target_classes: List[str] = None
    target_class_ids: Dict[str, int] = None
    
    # Mode selection
    use_linelogic_mode: bool = True
    
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
        
        # Initialize LineLogic line configurations
        if self.roi.line_positions is None:
            self.roi.line_positions = [880, 960, 1040, 1120]  # Default from LineLogic
        
        if self.roi.line_ids is None:
            self.roi.line_ids = [1, 2, 3, 4]  # Default line IDs
    
    def calculate_frame_thresholds(self, fps: int):
        """Calculate frame thresholds based on FPS and time thresholds."""
        self.linelogic.safe_threshold_frames = int(self.linelogic.min_safe_time * fps)
        self.linelogic.uncertain_threshold_frames = int(self.linelogic.min_uncertain_time * fps)
        self.linelogic.very_brief_threshold_frames = int(self.linelogic.min_very_brief_time * fps)
        
        print(f"Frame thresholds for {fps} FPS:")
        print(f"  Safe: ≥{self.linelogic.safe_threshold_frames} frames ({self.linelogic.min_safe_time}s)")
        print(f"  Uncertain: {self.linelogic.uncertain_threshold_frames}-{self.linelogic.safe_threshold_frames-1} frames")
        print(f"  Very brief: {self.linelogic.very_brief_threshold_frames}-{self.linelogic.uncertain_threshold_frames-1} frames")
        print(f"  Discard: <{self.linelogic.very_brief_threshold_frames} frames")

# Global config instance
config = Config()