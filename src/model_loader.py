"""
Trash Bin Detection - Model Loader Module
Handles YOLO model loading and inference
Optimized for Raspberry Pi 4 64-bit
"""

import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handles YOLOv8 model loading and inference.
    Optimized for Raspberry Pi 4 64-bit CPU inference.
    """

    def __init__(self, config: dict):
        """Initialize model loader with configuration."""
        self.config = config
        self.model_config = config.get('model', {})

        self.model_path = self.model_config.get('path', 'models/best.pt')
        self.confidence = self.model_config.get('confidence_threshold', 0.70)  # 70% default
        self.iou = self.model_config.get('iou_threshold', 0.45)
        self.max_det = self.model_config.get('max_detections', 100)
        self.device = self.model_config.get('device', 'cpu')  # CPU for Raspberry Pi
        self.classes = config.get('detection', {}).get('classes', None)

        # Performance settings optimized for Raspberry Pi 4
        self.resize = config.get('performance', {}).get('resize_for_detection', True)
        self.imgsz = config.get('performance', {}).get('detection_size', 640)

        self.model = None
        self.loaded = False

    def load(self):
        """Load the YOLO model optimized for Raspberry Pi 4."""
        logger.info(f"Loading model from: {self.model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Confidence threshold: {self.confidence * 100:.0f}%")

        # Verify model file exists
        model_file = Path(self.model_path)
        if not model_file.exists():
            # Try alternate locations
            alt_paths = [
                Path('best.pt'),
                Path('models') / 'best.pt',
                Path(__file__).parent.parent / 'models' / 'best.pt',
                Path(__file__).parent.parent / 'best.pt',
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_file = alt_path
                    logger.info(f"Found model at alternate location: {model_file}")
                    break
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            from ultralytics import YOLO

            # Load model
            logger.info("Loading YOLO model (this may take a moment on Raspberry Pi)...")
            self.model = YOLO(str(model_file))

            # Set model to evaluation mode for inference
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'eval'):
                self.model.model.eval()

            # Warm up model with dummy inference
            logger.info("Warming up model with dummy inference...")
            dummy_input = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_input, verbose=False, device=self.device)

            self.loaded = True
            
            # Log model info
            class_names = self.model.names
            logger.info(f"Model loaded successfully!")
            logger.info(f"Model classes ({len(class_names)}): {list(class_names.values())}")

        except ImportError:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect(self, frame):
        """
        Run detection on a frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            list: YOLO results
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if frame is None:
            return []

        try:
            # Run inference with settings optimized for Raspberry Pi
            results = self.model.predict(
                frame,
                conf=self.confidence,  # 70% confidence threshold
                iou=self.iou,
                max_det=self.max_det,
                device=self.device,  # CPU for Raspberry Pi
                classes=self.classes,
                imgsz=self.imgsz if self.resize else None,
                verbose=False,
                half=False  # Disable half precision on CPU
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
            'device': self.device,
            'confidence_threshold': self.confidence,
            'detection_size': self.imgsz
        }


class ModelValidator:
    """Validates model file and compatibility for Raspberry Pi."""

    @staticmethod
    def validate(model_path: str) -> dict:
        """
        Validate a model file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            dict: Validation results
        """
        result = {
            'valid': False,
            'path': model_path,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        path = Path(model_path)

        # Check file exists
        if not path.exists():
            result['errors'].append(f"File not found: {model_path}")
            return result

        # Check file extension
        valid_extensions = ['.pt', '.pth', '.onnx', '.engine']
        if path.suffix.lower() not in valid_extensions:
            result['errors'].append(f"Unsupported file format: {path.suffix}")
            return result

        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        result['info']['size_mb'] = round(size_mb, 2)

        if size_mb < 0.1:
            result['errors'].append("Model file seems too small")
            return result

        # Raspberry Pi specific warnings
        if size_mb > 100:
            result['warnings'].append(f"Large model ({size_mb:.1f}MB) may be slow on Raspberry Pi")

        # Try loading model
        try:
            from ultralytics import YOLO
            model = YOLO(str(path))

            result['info']['classes'] = model.names
            result['info']['num_classes'] = len(model.names)
            result['info']['task'] = model.task if hasattr(model, 'task') else 'detect'

            result['valid'] = True
            logger.info(f"Model validation successful: {len(model.names)} classes")

        except Exception as e:
            result['errors'].append(f"Failed to load model: {str(e)}")

        return result


def test_model():
    """Test model loading and inference."""
    import sys
    
    print("=" * 60)
    print("Model Loader Test")
    print("=" * 60)
    
    # Create minimal config
    config = {
        'model': {
            'path': 'models/best.pt',
            'confidence_threshold': 0.70,
            'iou_threshold': 0.45,
            'max_detections': 100,
            'device': 'cpu'
        },
        'detection': {
            'classes': None
        },
        'performance': {
            'resize_for_detection': True,
            'detection_size': 640
        }
    }
    
    try:
        # Load model
        loader = ModelLoader(config)
        loader.load()
        
        print(f"\nModel Info: {loader.get_info()}")
        
        # Test inference
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = loader.detect(test_frame)
        
        print(f"Test inference successful: {len(results)} results")
        print("\nModel loader test PASSED!")
        
    except Exception as e:
        print(f"Model loader test FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_model()
