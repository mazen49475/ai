"""
Trash Bin Detection - Main Detector Module
Real-time detection using YOLOv8 on Raspberry Pi 4 64-bit
Buzzer on GPIO17 - ON when class detected (>=70%), OFF when not detected
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import ModelLoader
from src.camera import CameraManager
from src.web_server import WebServer
from src.alerts import AlertManager
from src.utils import load_config, setup_logging, draw_detections

import cv2
import numpy as np


class TrashBinDetector:
    """
    Main class for trash bin detection system.
    Optimized for Raspberry Pi 4 64-bit with buzzer on GPIO17.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the detector with configuration."""
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config)

        self.logger.info("=" * 60)
        self.logger.info("Initializing Trash Bin Detector")
        self.logger.info("Platform: Raspberry Pi 4 64-bit")
        self.logger.info(f"Confidence Threshold: {self.config['model']['confidence_threshold'] * 100:.0f}%")
        self.logger.info("Buzzer: GPIO17 (BCM)")
        self.logger.info("=" * 60)

        # Initialize components
        self.model = None
        self.camera = None
        self.web_server = None
        self.alert_manager = None

        # State variables
        self.running = False
        self.current_frame = None
        self.current_detections = []
        self.fps = 0
        self.frame_count = 0
        self.detection_count = 0

        # Performance tracking
        self.last_fps_time = time.time()
        self.fps_frame_count = 0

    def initialize(self):
        """Initialize all components."""
        try:
            # Load YOLO model
            self.logger.info("Loading YOLO detection model...")
            self.model = ModelLoader(self.config)
            self.model.load()
            self.logger.info(f"Model loaded successfully - Classes: {list(self.model.get_classes().values())}")

            # Initialize camera
            self.logger.info("Initializing camera...")
            self.camera = CameraManager(self.config)
            self.camera.initialize()
            self.logger.info("Camera initialized successfully")

            # Initialize alert system (buzzer on GPIO17)
            self.logger.info("Initializing alert system (Buzzer on GPIO17)...")
            self.alert_manager = AlertManager(self.config)
            self.alert_manager.initialize()
            self.logger.info("Alert system initialized")

            # Initialize web server
            if self.config.get('server', {}).get('port'):
                self.logger.info("Starting web server...")
                self.web_server = WebServer(self.config, self)
                self.web_server.start()
                self.logger.info(f"Web server started on port {self.config['server']['port']}")

            self.logger.info("=" * 60)
            self.logger.info("All components initialized successfully!")
            self.logger.info("=" * 60)

            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def process_frame(self, frame):
        """
        Process a single frame for detection.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            tuple: (annotated_frame, detections_list)
        """
        if frame is None:
            return None, []

        # Run YOLO detection
        results = self.model.detect(frame)

        # Extract detections
        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.model.names[class_id]

                    # Filter by confidence threshold (70%)
                    if confidence >= self.config['model']['confidence_threshold']:
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        detections.append(detection)
                        self.detection_count += 1

        # Draw detections on frame
        annotated_frame = draw_detections(
            frame.copy(),
            detections,
            self.config,
            self.fps
        )

        # Add buzzer status indicator to frame
        if self.alert_manager:
            buzzer_status = "BUZZER: ON" if self.alert_manager.buzzer_state else "BUZZER: OFF"
            color = (0, 0, 255) if self.alert_manager.buzzer_state else (0, 255, 0)
            cv2.putText(
                annotated_frame,
                buzzer_status,
                (annotated_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        return annotated_frame, detections

    def calculate_fps(self):
        """Calculate current FPS."""
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time

        if elapsed >= 1.0:
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.last_fps_time = current_time

    def save_detection(self, frame, detections):
        """Save frame with detections."""
        if not self.config['output']['save_images']:
            return

        if len(detections) == 0:
            return

        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"detection_{timestamp}.{self.config['output']['image_format']}"
        filepath = output_dir / filename

        cv2.imwrite(str(filepath), frame)
        self.logger.debug(f"Saved detection to {filepath}")

    def run(self):
        """Main detection loop."""
        if not self.initialize():
            self.logger.error("Failed to initialize. Exiting.")
            return

        self.running = True
        self.logger.info("Starting detection loop...")
        self.logger.info("Press Ctrl+C to stop")

        skip_frames = self.config['performance']['skip_frames']
        frame_skip_counter = 0

        try:
            while self.running:
                # Capture frame
                frame = self.camera.read()

                if frame is None:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue

                self.frame_count += 1

                # Skip frames if configured (for performance on Pi)
                if skip_frames > 0:
                    frame_skip_counter += 1
                    if frame_skip_counter <= skip_frames:
                        continue
                    frame_skip_counter = 0

                # Process frame for detections
                annotated_frame, detections = self.process_frame(frame)

                # Update state
                self.current_frame = annotated_frame
                self.current_detections = detections

                # Calculate FPS
                self.calculate_fps()

                # Log detections
                if len(detections) > 0 and self.config['logging']['log_detections']:
                    self.logger.info(f"Detected {len(detections)} object(s): " +
                                   ", ".join([f"{d['class_name']} ({d['confidence']:.0%})"
                                            for d in detections]))

                # BUZZER CONTROL: Process detections for alerts
                # Buzzer ON when class detected >= 70%, OFF when not detected
                if self.alert_manager:
                    if len(detections) > 0:
                        # Class detected - check if buzzer should be ON
                        self.alert_manager.process_detections(detections)
                    else:
                        # No class detected - turn buzzer OFF
                        self.alert_manager.no_detections()

                # Save detection if enabled
                self.save_detection(annotated_frame, detections)

                # Small delay to prevent CPU overload
                time.sleep(0.001)

        except KeyboardInterrupt:
            self.logger.info("\nInterrupted by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Error in detection loop: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.cleanup()

    def get_frame(self):
        """Get current frame for web streaming."""
        if self.current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', self.current_frame)
            if ret:
                return buffer.tobytes()
        return None

    def get_status(self):
        """Get current status for API."""
        return {
            'running': self.running,
            'fps': round(self.fps, 1),
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'current_detections': self.current_detections,
            'buzzer_on': self.alert_manager.buzzer_state if self.alert_manager else False
        }

    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        self.logger.info("=" * 60)
        self.logger.info("Cleaning up...")

        if self.alert_manager:
            self.alert_manager.cleanup()
            self.logger.info("Buzzer turned OFF and GPIO cleaned up")

        if self.camera:
            self.camera.release()
            self.logger.info("Camera released")

        if self.web_server:
            self.web_server.stop()
            self.logger.info("Web server stopped")

        self.logger.info("Cleanup complete")
        self.logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Trash Bin Detection System for Raspberry Pi 4')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Path to model file (overrides config)')
    parser.add_argument('--source', type=str, help='Camera source (overrides config)')
    parser.add_argument('--confidence', type=float, default=0.70,
                       help='Confidence threshold (default: 0.70 = 70%%)')
    args = parser.parse_args()

    print("=" * 60)
    print("Trash Bin Detection System")
    print("Raspberry Pi 4 64-bit Edition")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Confidence: {args.confidence * 100:.0f}%")
    print(f"Buzzer: GPIO17 (BCM)")
    print("=" * 60)

    # Create detector
    detector = TrashBinDetector(args.config)

    # Override config if arguments provided
    if args.model:
        detector.config['model']['path'] = args.model
    if args.source:
        detector.config['camera']['source'] = args.source
    if args.confidence:
        detector.config['model']['confidence_threshold'] = args.confidence
        detector.config['alerts']['min_confidence'] = args.confidence

    # Run detector
    detector.run()


if __name__ == "__main__":
    main()
