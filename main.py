
#!/usr/bin/env python3
"""
Trash Bin Detection System for Raspberry Pi 4
Main entry point
"""

import cv2
import time
import yaml
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPIO Setup for Buzzer
BUZZER_PIN = 17
gpio_available = False

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    gpio_available = True
    logger.info(f"GPIO initialized - Buzzer on pin {BUZZER_PIN}")
except Exception as e:
    logger.warning(f"GPIO not available: {e}")

def buzzer_on():
    """Turn buzzer ON"""
    if gpio_available:
        GPIO.output(BUZZER_PIN, GPIO.HIGH)

def buzzer_off():
    """Turn buzzer OFF"""
    if gpio_available:
        GPIO.output(BUZZER_PIN, GPIO.LOW)

def load_config():
    """Load configuration from yaml file"""
    try:
        with open('config/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        # Default config
        return {
            'model': {
                'path': 'models/best.pt',
                'confidence_threshold': 0.70
            },
            'camera': {
                'source': 0,
                'width': 640,
                'height': 480
            }
        }

def main():
    """Main detection loop"""
    logger.info("=== Trash Bin Detection System ===")
    logger.info("Press 'q' to quit")
    
    # Load config
    config = load_config()
    confidence_threshold = config['model'].get('confidence_threshold', 0.70)
    logger.info(f"Confidence threshold: {confidence_threshold * 100}%")
    
    # Load YOLO model
    try:
        from ultralytics import YOLO
        model_path = config['model'].get('path', 'models/best.pt')
        model = YOLO(model_path)
        logger.info(f"Model loaded: {model_path}")
        
        # Get class names
        class_names = model.names
        logger.info(f"Classes: {class_names}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Initialize camera
    camera_source = config['camera'].get('source', 0)
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera'].get('width', 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera'].get('height', 480))
    logger.info(f"Camera initialized: source={camera_source}")
    
    # Test buzzer
    if gpio_available:
        logger.info("Testing buzzer...")
        buzzer_on()
        time.sleep(0.2)
        buzzer_off()
    
    buzzer_state = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Run detection
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            # Check for detections
            detection_found = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = class_names.get(cls_id, f"class_{cls_id}")
                        
                        if conf >= confidence_threshold:
                            detection_found = True
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw label
                            label = f"{cls_name}: {conf*100:.1f}%"
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            logger.info(f"Detected: {cls_name} ({conf*100:.1f}%)")
            
            # Control buzzer based on detection
            if detection_found and not buzzer_state:
                buzzer_on()
                buzzer_state = True
                logger.info("BUZZER ON - Object detected!")
            elif not detection_found and buzzer_state:
                buzzer_off()
                buzzer_state = False
                logger.info("BUZZER OFF - No detection")
            
            # Display buzzer status on frame
            status = "BUZZER: ON" if buzzer_state else "BUZZER: OFF"
            color = (0, 0, 255) if buzzer_state else (0, 255, 0)
            cv2.putText(frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Show frame
            cv2.imshow("Trash Bin Detection", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit requested")
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        buzzer_off()
        cap.release()
        cv2.destroyAllWindows()
        if gpio_available:
            GPIO.cleanup()
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()
