"""
Trash Bin Detection - Utility Functions
Helper functions for configuration, logging, and drawing
"""

import os
import yaml
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        logging.warning(f"Config file not found: {config_path}. Using defaults.")
        return get_default_config()
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Merge with defaults for any missing keys
        default = get_default_config()
        config = deep_merge(default, config)
        
        return config
        
    except Exception as e:
        logging.error(f"Error loading config: {e}. Using defaults.")
        return get_default_config()


def get_default_config() -> dict:
    """Get default configuration."""
    return {
        'model': {
            'path': 'models/best.pt',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'max_detections': 100,
            'device': 'cpu'
        },
        'camera': {
            'source': 'picamera',
            'width': 640,
            'height': 480,
            'fps': 30,
            'rotation': 0,
            'flip_horizontal': False,
            'flip_vertical': False
        },
        'detection': {
            'classes': None,
            'min_box_area': 1000,
            'tracking_enabled': False
        },
        'server': {
            'host': '0.0.0.0',
            'port': 5000,
            'debug': False,
            'threaded': True
        },
        'output': {
            'save_images': True,
            'save_video': False,
            'output_dir': 'output',
            'image_format': 'jpg',
            'video_format': 'mp4',
            'annotate_frames': True,
            'show_fps': True,
            'show_timestamp': True
        },
        'logging': {
            'level': 'INFO',
            'log_dir': 'logs',
            'max_log_size_mb': 10,
            'backup_count': 5,
            'log_detections': True
        },
        'alerts': {
            'enabled': False,
            'min_confidence': 0.7,
            'cooldown_seconds': 30
        },
        'performance': {
            'skip_frames': 0,
            'resize_for_detection': True,
            'detection_size': 640,
            'use_threading': True,
            'buffer_size': 4
        }
    }


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def setup_logging(config: dict) -> logging.Logger:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO').upper())
    log_dir = Path(log_config.get('log_dir', 'logs'))
    max_size = log_config.get('max_log_size_mb', 10) * 1024 * 1024
    backup_count = log_config.get('backup_count', 5)
    
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('trash_detector')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = log_dir / f"detector_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def draw_detections(frame, detections: list, config: dict, fps: float = 0) -> np.ndarray:
    """Draw detection boxes and labels on frame."""
    if frame is None:
        return None
    
    output_config = config.get('output', {})
    
    # Colors for different classes (BGR format)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
    ]
    
    # Draw each detection
    for det in detections:
        bbox = det['bbox']
        confidence = det['confidence']
        class_name = det['class_name']
        class_id = det.get('class_id', 0)
        
        color = colors[class_id % len(colors)]
        
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name}: {confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            frame,
            (x1, y1 - label_height - 10),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    # Draw FPS
    if output_config.get('show_fps', True) and fps > 0:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
    
    # Draw timestamp
    if output_config.get('show_timestamp', True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame,
            timestamp,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
    
    # Draw detection count
    det_text = f"Detections: {len(detections)}"
    cv2.putText(
        frame,
        det_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )
    
    return frame


def create_thumbnail(frame, size=(320, 240)):
    """Create a thumbnail from frame."""
    if frame is None:
        return None
    
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    
    return intersection / union


def get_system_info():
    """Get system information for diagnostics."""
    import platform
    
    info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
    }
    
    # Try to get Raspberry Pi info
    try:
        with open('/proc/device-tree/model', 'r') as f:
            info['device_model'] = f.read().strip('\x00')
    except:
        pass
    
    # Try to get memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['memory_total_gb'] = round(mem.total / (1024**3), 2)
        info['memory_available_gb'] = round(mem.available / (1024**3), 2)
        info['cpu_percent'] = psutil.cpu_percent()
        info['cpu_count'] = psutil.cpu_count()
    except:
        pass
    
    return info
