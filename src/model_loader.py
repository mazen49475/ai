"""
Trash Bin Detection - Model Loader Module
Handles YOLO model loading and inference
"""

import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles YOLOv8 model loading and inference."""
    
    def __init__(self, config: dict):
        """Initialize model loader with configuration."""
        self.config = config
        self.model_config = config.get('model', {})
        
        self.model_path = self.model_config.get('path', 'models/best.pt')
        self.confidence = self.model_config.get('confidence_threshold', 0.5)
        self.iou = self.model_config.get('iou_threshold', 0.45)
        self.max_det = self.model_config.get('max_detections', 100)
        self.device = self.model_config.get('device', 'cpu')
        self.classes = config.get('detection', {}).get('classes', None)
        
        # Performance settings
        self.resize = config.get('performance', {}).get('resize_for_detection', True)
        self.imgsz = config.get('performance', {}).get('detection_size', 640)
        
        self.model = None
        self.loaded = False
    
    def load(self):
        """Load the YOLO model."""
        logger.info(f"Loading model from: {self.model_path}")
        
        # Verify model file exists
        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            from ultralytics import YOLO
            
            # Load model
            self.model = YOLO(str(model_file))
            
            # Warm up model
            logger.info("Warming up model...")
            dummy_input = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(dummy_input, verbose=False)
            
            self.loaded = True
            logger.info(f"Model loaded successfully. Classes: {self.model.names}")
            
        except ImportError:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, frame):
        """Run detection on a frame."""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if frame is None:
            return []
        
        try:
            # Run inference
            results = self.model.predict(
                frame,
                conf=self.confidence,
                iou=self.iou,
                max_det=self.max_det,
                device=self.device,
                classes=self.classes,
                imgsz=self.imgsz if self.resize else None,
                verbose=False
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def get_classes(self):
        """Get model class names."""
        if self.model is not None:
            return self.model.names
        return {}
    
    def get_info(self):
        """Get model information."""
        if not self.loaded:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'path': self.model_path,
            'classes': self.model.names,
            'num_classes': len(self.model.names),
            'device': self.device
        }


class ModelValidator:
    """Validates model file and compatibility."""
    
    @staticmethod
    def validate(model_path: str) -> dict:
        """Validate a model file."""
        result = {
            'valid': False,
            'path': model_path,
            'errors': [],
            'info': {}
        }
        
        path = Path(model_path)
        
        # Check file exists
        if not path.exists():
            result['errors'].append(f"File not found: {model_path}")
            return result
        
        # Check file extension
        if path.suffix.lower() not in ['.pt', '.pth', '.onnx', '.engine']:
            result['errors'].append(f"Unsupported file format: {path.suffix}")
            return result
        
        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        result['info']['size_mb'] = round(size_mb, 2)
        
        if size_mb < 0.1:
            result['errors'].append("Model file seems too small")
            return result
        
        # Try loading model
        try:
            from ultralytics import YOLO
            model = YOLO(str(path))
            
            result['info']['classes'] = model.names
            result['info']['num_classes'] = len(model.names)
            result['info']['task'] = model.task
            
            result['valid'] = True
            
        except Exception as e:
            result['errors'].append(f"Failed to load model: {str(e)}")
        
        return result
