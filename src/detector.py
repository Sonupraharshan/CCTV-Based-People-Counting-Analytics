"""
YOLOv8 Detector Wrapper
Provides a unified interface for YOLOv8 object detection with batched inference support.
"""

import torch
import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path
from ultralytics import YOLO


class Detector:
    """YOLOv8 object detector with configurable parameters."""
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        conf_threshold: float = 0.5,
        nms_iou: float = 0.45,
        classes: Optional[List[int]] = None
    ):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to model weights (.pt, .onnx, .engine)
            device: Device for inference ('cuda' or 'cpu')
            conf_threshold: Confidence threshold for detections
            nms_iou: IOU threshold for non-maximum suppression
            classes: List of class IDs to detect (None for all classes)
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self.classes = classes if classes is not None else [0]  # 0 = person in COCO
        
        # Load model
        print(f"Loading detector from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"Detector loaded on {device}")
        
        # Model info
        self.model_type = self._get_model_type()
        print(f"Model type: {self.model_type}")
    
    def _get_model_type(self) -> str:
        """Determine model type from file extension."""
        extension = Path(self.model_path).suffix.lower()
        if extension == '.pt':
            return 'pytorch'
        elif extension == '.onnx':
            return 'onnx'
        elif extension in ['.engine', '.trt']:
            return 'tensorrt'
        else:
            return 'unknown'
    
    def detect(
        self,
        frame: np.ndarray,
        return_format: str = 'xyxy'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            return_format: Bounding box format ('xyxy' or 'xywh')
        
        Returns:
            Tuple of (boxes, scores, class_ids):
                - boxes: np.ndarray of shape (N, 4) - bounding boxes
                - scores: np.ndarray of shape (N,) - confidence scores
                - class_ids: np.ndarray of shape (N,) - class IDs
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.nms_iou,
            classes=self.classes,
            device=self.device,
            verbose=False
        )
        
        # Extract detections from first result
        result = results[0]
        
        if len(result.boxes) == 0:
            # No detections
            return np.array([]), np.array([]), np.array([])
        
        # Get boxes in requested format
        if return_format == 'xyxy':
            boxes = result.boxes.xyxy.cpu().numpy()
        elif return_format == 'xywh':
            boxes = result.boxes.xywh.cpu().numpy()
        else:
            raise ValueError(f"Unknown format: {return_format}")
        
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        return boxes, scores, class_ids
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        return_format: str = 'xyxy'
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Detect objects in a batch of frames (more efficient).
        
        Args:
            frames: List of input frames (BGR format)
            return_format: Bounding box format ('xyxy' or 'xywh')
        
        Returns:
            List of tuples (boxes, scores, class_ids) for each frame
        """
        results = self.model.predict(
            frames,
            conf=self.conf_threshold,
            iou=self.nms_iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
            stream=True  # Use streaming for memory efficiency
        )
        
        detections = []
        for result in results:
            if len(result.boxes) == 0:
                detections.append((np.array([]), np.array([]), np.array([])))
                continue
            
            # Get boxes in requested format
            if return_format == 'xyxy':
                boxes = result.boxes.xyxy.cpu().numpy()
            elif return_format == 'xywh':
                boxes = result.boxes.xywh.cpu().numpy()
            else:
                raise ValueError(f"Unknown format: {return_format}")
            
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            detections.append((boxes, scores, class_ids))
        
        return detections
    
    def export_onnx(
        self,
        output_path: str,
        dynamic: bool = False,
        simplify: bool = True
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            dynamic: Enable dynamic input shapes
            simplify: Simplify ONNX model
        
        Returns:
            Path to exported model
        """
        print(f"Exporting model to ONNX: {output_path}")
        exported = self.model.export(
            format='onnx',
            dynamic=dynamic,
            simplify=simplify
        )
        print(f"Model exported successfully: {exported}")
        return exported
    
    def warmup(self, img_size: Tuple[int, int] = (640, 640), num_runs: int = 3):
        """
        Warm up the model for accurate benchmarking.
        
        Args:
            img_size: Image size (height, width) for warm-up
            num_runs: Number of warm-up runs
        """
        print(f"Warming up detector with {num_runs} runs...")
        dummy_img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        
        for _ in range(num_runs):
            self.detect(dummy_img)
        
        print("Warmup complete")
    
    def get_fps(self, frame: np.ndarray, num_runs: int = 30) -> float:
        """
        Measure inference FPS.
        
        Args:
            frame: Sample frame for benchmark
            num_runs: Number of inference runs
        
        Returns:
            Average FPS
        """
        import time
        
        # Warm up
        self.warmup(num_runs=5)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            self.detect(frame)
        
        elapsed = time.time() - start_time
        fps = num_runs / elapsed
        
        print(f"Average FPS: {fps:.2f}")
        return fps
    
    def update_params(
        self,
        conf_threshold: Optional[float] = None,
        nms_iou: Optional[float] = None,
        classes: Optional[List[int]] = None
    ):
        """
        Update detector parameters on the fly.
        
        Args:
            conf_threshold: New confidence threshold
            nms_iou: New NMS IOU threshold
            classes: New class filter
        """
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if nms_iou is not None:
            self.nms_iou = nms_iou
        if classes is not None:
            self.classes = classes


def convert_bbox_format(
    bbox: np.ndarray,
    from_format: str,
    to_format: str,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None
) -> np.ndarray:
    """
    Convert bounding box between different formats.
    
    Args:
        bbox: Bounding box array
        from_format: Current format ('xyxy', 'xywh', 'cxcywh')
        to_format: Target format ('xyxy', 'xywh', 'cxcywh')
        img_width: Image width (needed for normalization)
        img_height: Image height (needed for normalization)
    
    Returns:
        Converted bounding box
    """
    if from_format == to_format:
        return bbox
    
    # Convert to xyxy first as intermediate format
    if from_format == 'xywh':
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
    elif from_format == 'cxcywh':
        cx, cy, w, h = bbox
        x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    elif from_format == 'xyxy':
        x1, y1, x2, y2 = bbox
    else:
        raise ValueError(f"Unknown from_format: {from_format}")
    
    # Convert from xyxy to target format
    if to_format == 'xyxy':
        return np.array([x1, y1, x2, y2])
    elif to_format == 'xywh':
        return np.array([x1, y1, x2 - x1, y2 - y1])
    elif to_format == 'cxcywh':
        return np.array([(x1 + x2)/2, (y1 + y2)/2, x2 - x1, y2 - y1])
    else:
        raise ValueError(f"Unknown to_format: {to_format}")
