"""
Trash Bin Detection - Alert System Module
Buzzer on GPIO17 - ON when class detected (>=70% confidence), OFF when not detected
Optimized for Raspberry Pi 4 64-bit
"""

import logging
import time
from threading import Thread, Lock

logger = logging.getLogger(__name__)

# Try to import GPIO library for Raspberry Pi
GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    logger.info("RPi.GPIO library loaded successfully")
except ImportError:
    logger.warning("RPi.GPIO not available - running in simulation mode")
except Exception as e:
    logger.warning(f"GPIO initialization issue: {e} - running in simulation mode")


class BuzzerAlert:
    """
    Manages buzzer on GPIO17.
    Buzzer stays ON while class is detected with confidence >= threshold.
    Buzzer turns OFF when class is not detected.
    """

    def __init__(self, config: dict):
        """Initialize buzzer alert with configuration."""
        self.config = config
        self.alert_config = config.get('alerts', {})
        self.gpio_config = self.alert_config.get('gpio', {})

        # Configuration
        self.enabled = self.gpio_config.get('enabled', True)
        self.pin = self.gpio_config.get('pin', 17)  # GPIO17 for buzzer
        self.min_confidence = self.alert_config.get('min_confidence', 0.70)  # 70% threshold
        self.alert_classes = self.alert_config.get('alert_classes', [])

        # State tracking
        self.lock = Lock()
        self._initialized = False
        self.buzzer_on = False
        self._last_detection_time = 0
        
        # Debounce settings (prevent buzzer flickering)
        self.debounce_time = 0.3  # 300ms debounce

    def initialize(self):
        """Initialize GPIO for buzzer on pin 17."""
        if not self.enabled:
            logger.info("Buzzer alerts disabled in config")
            self._initialized = True
            return True

        if not GPIO_AVAILABLE:
            logger.info("GPIO not available - buzzer will run in simulation mode")
            self._initialized = True
            return True

        try:
            # Setup GPIO with BCM numbering
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Configure GPIO17 as output
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, GPIO.LOW)  # Start with buzzer OFF
            
            self._initialized = True
            self.buzzer_on = False
            
            logger.info(f"Buzzer initialized on GPIO{self.pin} (BCM)")
            logger.info(f"Confidence threshold: {self.min_confidence * 100:.0f}%")
            logger.info(f"Alert classes: {self.alert_classes}")

            # Test buzzer with short beep
            self._test_buzzer()
            
            return True

        except Exception as e:
            logger.error(f"Failed to initialize buzzer on GPIO{self.pin}: {e}")
            self._initialized = True  # Allow system to run without buzzer
            return False

    def _test_buzzer(self):
        """Test buzzer with short beeps on startup."""
        logger.info("Testing buzzer...")
        try:
            # Two short beeps to indicate startup
            for _ in range(2):
                self._buzzer_on()
                time.sleep(0.1)
                self._buzzer_off()
                time.sleep(0.1)
            logger.info("Buzzer test complete")
        except Exception as e:
            logger.error(f"Buzzer test failed: {e}")

    def check_and_alert(self, detections: list):
        """
        Check detections and control buzzer.
        Buzzer ON when class detected with confidence >= 70%.
        Buzzer OFF when no class detected.
        
        Args:
            detections: List of detection dictionaries with 'class_name' and 'confidence'
            
        Returns:
            bool: True if buzzer is ON, False if OFF
        """
        if not self._initialized or not self.enabled:
            return False

        # Check if any detection matches our alert criteria
        target_detected = self._check_detections(detections)

        # Control buzzer based on detection
        current_time = time.time()
        
        if target_detected:
            # Detection found - turn buzzer ON
            if not self.buzzer_on:
                self._buzzer_on()
                logger.info(f"BUZZER ON - Class detected with confidence >= {self.min_confidence * 100:.0f}%")
            self._last_detection_time = current_time
        else:
            # No detection - turn buzzer OFF (with debounce)
            if self.buzzer_on:
                # Add small debounce to prevent flickering
                if (current_time - self._last_detection_time) > self.debounce_time:
                    self._buzzer_off()
                    logger.info("BUZZER OFF - No class detected")

        return self.buzzer_on

    def _check_detections(self, detections: list) -> bool:
        """
        Check if any detection matches alert criteria.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            bool: True if matching detection found
        """
        if not detections:
            return False

        for det in detections:
            class_name = det.get('class_name', '').lower()
            confidence = det.get('confidence', 0)

            # Check confidence threshold (70%)
            if confidence < self.min_confidence:
                continue

            # If no specific alert classes defined, alert on any detection >= threshold
            if not self.alert_classes:
                logger.debug(f"Detection: {class_name} ({confidence:.2%}) - ALERT!")
                return True

            # Check if class matches any alert class
            for alert_class in self.alert_classes:
                alert_lower = alert_class.lower()
                
                # Match by exact name, partial match, or common terms
                if (alert_lower in class_name or 
                    class_name in alert_lower or
                    self._fuzzy_match(class_name, alert_lower)):
                    logger.debug(f"Detection: {class_name} ({confidence:.2%}) matches '{alert_class}' - ALERT!")
                    return True

        return False

    def _fuzzy_match(self, detected: str, target: str) -> bool:
        """
        Fuzzy match detection class with target class.
        Handles variations like 'garbage_container', 'garbage-container', etc.
        """
        # Common variations
        detected_clean = detected.replace('_', ' ').replace('-', ' ').lower()
        target_clean = target.replace('_', ' ').replace('-', ' ').lower()
        
        # Check if key words match
        detected_words = set(detected_clean.split())
        target_words = set(target_clean.split())
        
        # If any word matches, consider it a match
        return bool(detected_words & target_words)

    def _buzzer_on(self):
        """Turn buzzer ON."""
        with self.lock:
            if GPIO_AVAILABLE and self.enabled:
                try:
                    GPIO.output(self.pin, GPIO.HIGH)
                except Exception as e:
                    logger.error(f"Error turning buzzer ON: {e}")
            self.buzzer_on = True

    def _buzzer_off(self):
        """Turn buzzer OFF."""
        with self.lock:
            if GPIO_AVAILABLE and self.enabled:
                try:
                    GPIO.output(self.pin, GPIO.LOW)
                except Exception as e:
                    logger.error(f"Error turning buzzer OFF: {e}")
            self.buzzer_on = False

    def force_off(self):
        """Force buzzer OFF (for emergency/cleanup)."""
        self._buzzer_off()
        logger.info("Buzzer forced OFF")

    def cleanup(self):
        """Cleanup GPIO resources."""
        logger.info("Cleaning up buzzer...")
        
        # Ensure buzzer is OFF
        self._buzzer_off()
        
        if GPIO_AVAILABLE and self._initialized and self.enabled:
            try:
                GPIO.cleanup(self.pin)
                logger.info(f"GPIO{self.pin} cleaned up")
            except Exception as e:
                logger.error(f"GPIO cleanup error: {e}")


class AlertManager:
    """
    Manages all alert systems.
    Currently supports GPIO buzzer on GPIO17.
    """

    def __init__(self, config: dict):
        """Initialize alert manager with configuration."""
        self.config = config
        self.buzzer = BuzzerAlert(config)
        self.alerts_enabled = config.get('alerts', {}).get('enabled', True)
        
        logger.info(f"Alert Manager initialized - Alerts {'enabled' if self.alerts_enabled else 'disabled'}")

    def initialize(self):
        """Initialize all alert systems."""
        if self.alerts_enabled:
            return self.buzzer.initialize()
        return True

    def process_detections(self, detections: list) -> bool:
        """
        Process detections and trigger appropriate alerts.
        
        Args:
            detections: List of detection dictionaries from model
            
        Returns:
            bool: True if any alert was triggered
        """
        if not self.alerts_enabled:
            return False
            
        return self.buzzer.check_and_alert(detections)

    def no_detections(self):
        """Called when no detections in current frame - turns buzzer OFF."""
        if self.alerts_enabled:
            return self.buzzer.check_and_alert([])
        return False

    @property
    def buzzer_state(self) -> bool:
        """Get current buzzer state."""
        return self.buzzer.buzzer_on

    @property
    def buzzer_on(self) -> bool:
        """Get current buzzer state (alias for compatibility)."""
        return self.buzzer.buzzer_on

    def cleanup(self):
        """Cleanup all alert resources."""
        logger.info("Cleaning up Alert Manager...")
        self.buzzer.cleanup()
        logger.info("Alert Manager cleanup complete")
