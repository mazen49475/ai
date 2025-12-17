"""
Trash Bin Detection - Camera Manager Module
Handles Raspberry Pi Camera and other video sources
"""

import cv2
import numpy as np
import logging
from threading import Thread, Lock
import time

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages camera input from various sources."""
    
    def __init__(self, config: dict):
        """Initialize camera manager with configuration."""
        self.config = config
        self.camera_config = config.get('camera', {})
        
        self.source = self.camera_config.get('source', 'picamera')
        self.width = self.camera_config.get('width', 640)
        self.height = self.camera_config.get('height', 480)
        self.fps = self.camera_config.get('fps', 30)
        self.rotation = self.camera_config.get('rotation', 0)
        self.flip_h = self.camera_config.get('flip_horizontal', False)
        self.flip_v = self.camera_config.get('flip_vertical', False)
        
        self.camera = None
        self.picam2 = None
        self.frame = None
        self.running = False
        self.lock = Lock()
        self.thread = None
        
        # Performance settings
        self.use_threading = config.get('performance', {}).get('use_threading', True)
        self.buffer_size = config.get('performance', {}).get('buffer_size', 4)
    
    def initialize(self):
        """Initialize the camera based on source type."""
        logger.info(f"Initializing camera with source: {self.source}")
        
        if self.source == 'picamera':
            self._init_picamera()
        elif isinstance(self.source, int) or self.source.isdigit():
            self._init_usb_camera(int(self.source) if isinstance(self.source, str) else self.source)
        elif isinstance(self.source, str):
            self._init_video_file(self.source)
        else:
            raise ValueError(f"Unknown camera source: {self.source}")
        
        # Start capture thread if threading enabled
        if self.use_threading:
            self.running = True
            self.thread = Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            time.sleep(0.5)  # Allow camera to warm up
    
    def _init_picamera(self):
        """Initialize Raspberry Pi Camera using Picamera2."""
        try:
            from picamera2 import Picamera2
            from libcamera import Transform
            
            self.picam2 = Picamera2()
            
            # Configure camera
            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                buffer_count=self.buffer_size
            )
            
            # Apply rotation if needed
            if self.rotation in [90, 180, 270]:
                transform = Transform(hflip=self.rotation in [180, 270],
                                     vflip=self.rotation in [90, 180])
                config["transform"] = transform
            
            self.picam2.configure(config)
            self.picam2.start()
            
            logger.info("Picamera2 initialized successfully")
            
        except ImportError:
            logger.error("Picamera2 not available. Install with: pip install picamera2")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Picamera2: {e}")
            raise
    
    def _init_usb_camera(self, device_id: int):
        """Initialize USB webcam using OpenCV."""
        self.camera = cv2.VideoCapture(device_id)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open USB camera at index {device_id}")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        
        logger.info(f"USB camera initialized at index {device_id}")
    
    def _init_video_file(self, filepath: str):
        """Initialize video file as camera source."""
        self.camera = cv2.VideoCapture(filepath)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open video file: {filepath}")
        
        logger.info(f"Video file initialized: {filepath}")
    
    def _capture_loop(self):
        """Continuous capture loop for threaded operation."""
        while self.running:
            frame = self._read_frame()
            
            if frame is not None:
                with self.lock:
                    self.frame = frame
            
            time.sleep(1.0 / self.fps / 2)  # Half the frame interval
    
    def _read_frame(self):
        """Read a single frame from the camera."""
        frame = None
        
        try:
            if self.picam2 is not None:
                # Picamera2
                frame = self.picam2.capture_array()
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
            elif self.camera is not None:
                # OpenCV VideoCapture
                ret, frame = self.camera.read()
                if not ret:
                    return None
            
            if frame is not None:
                # Apply transformations
                frame = self._apply_transforms(frame)
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None
        
        return frame
    
    def _apply_transforms(self, frame):
        """Apply configured transformations to frame."""
        # Rotation (if not handled by Picamera2)
        if self.camera is not None and self.rotation != 0:
            if self.rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Flip
        if self.flip_h and self.flip_v:
            frame = cv2.flip(frame, -1)
        elif self.flip_h:
            frame = cv2.flip(frame, 1)
        elif self.flip_v:
            frame = cv2.flip(frame, 0)
        
        return frame
    
    def read(self):
        """Read current frame from camera."""
        if self.use_threading:
            with self.lock:
                return self.frame.copy() if self.frame is not None else None
        else:
            return self._read_frame()
    
    def get_properties(self):
        """Get camera properties."""
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'source': self.source
        }
    
    def release(self):
        """Release camera resources."""
        logger.info("Releasing camera resources...")
        
        self.running = False
        
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception as e:
                logger.error(f"Error closing Picamera2: {e}")
        
        if self.camera is not None:
            self.camera.release()
        
        logger.info("Camera released")


class SimulatedCamera:
    """Simulated camera for testing without hardware."""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame_count = 0
    
    def initialize(self):
        logger.info("Simulated camera initialized")
    
    def read(self):
        """Generate a test frame."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add some visual elements
        cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Simulated Camera", (10, self.height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1)
        
        self.frame_count += 1
        return frame
    
    def release(self):
        logger.info("Simulated camera released")
