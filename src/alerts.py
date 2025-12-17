"""
Trash Bin Detection - Alert System Module
Handles buzzer alerts via GPIO on Raspberry Pi
"""

import logging
import time
from threading import Thread, Lock

logger = logging.getLogger(__name__)

# Try to import GPIO library
GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    logger.warning("RPi.GPIO not available. Buzzer alerts disabled.")


class BuzzerAlert:
    """Manages buzzer alerts via GPIO."""
    
    def __init__(self, config: dict):
        """Initialize buzzer alert system."""
        self.config = config
        self.alert_config = config.get('alerts', {})
        self.gpio_config = self.alert_config.get('gpio', {})
        
        self.enabled = self.gpio_config.get('enabled', True) and GPIO_AVAILABLE
        self.pin = self.gpio_config.get('pin', 17)  # Default GPIO 17
        self.duration = self.gpio_config.get('duration', 0.5)  # Beep duration in seconds
        self.pattern = self.gpio_config.get('pattern', 'single')  # single, double, continuous
        
        # Alert control
        self.min_confidence = self.alert_config.get('min_confidence', 0.5)
        self.cooldown = self.alert_config.get('cooldown_seconds', 5)
        self.last_alert_time = 0
        self.lock = Lock()
        
        # Classes that trigger alerts
        self.alert_classes = self.alert_config.get('alert_classes', ['garbage', 'garbage bin', 'overflow'])
        
        self._initialized = False
    
    def initialize(self):
        """Initialize GPIO for buzzer."""
        if not self.enabled:
            logger.info("Buzzer alerts disabled")
            return False
        
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, GPIO.LOW)
            
            self._initialized = True
            logger.info(f"Buzzer initialized on GPIO {self.pin}")
            
            # Test beep on startup
            self._beep(0.1)
            time.sleep(0.1)
            self._beep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize buzzer: {e}")
            self.enabled = False
            return False
    
    def check_and_alert(self, detections: list):
        """Check detections and trigger alert if needed."""
        if not self.enabled or not self._initialized:
            return
        
        # Check if any detection should trigger alert
        should_alert = False
        for det in detections:
            class_name = det.get('class_name', '').lower()
            confidence = det.get('confidence', 0)
            
            # Check if class matches and confidence is high enough
            if any(alert_class.lower() in class_name for alert_class in self.alert_classes):
                if confidence >= self.min_confidence:
                    should_alert = True
                    break
        
        if should_alert:
            self._trigger_alert()
    
    def _trigger_alert(self):
        """Trigger buzzer alert with cooldown."""
        current_time = time.time()
        
        with self.lock:
            # Check cooldown
            if current_time - self.last_alert_time < self.cooldown:
                return
            
            self.last_alert_time = current_time
        
        # Run alert in separate thread to not block main loop
        Thread(target=self._play_alert_pattern, daemon=True).start()
    
    def _play_alert_pattern(self):
        """Play the configured alert pattern."""
        try:
            if self.pattern == 'single':
                self._beep(self.duration)
            elif self.pattern == 'double':
                self._beep(self.duration / 2)
                time.sleep(0.1)
                self._beep(self.duration / 2)
            elif self.pattern == 'triple':
                for _ in range(3):
                    self._beep(self.duration / 3)
                    time.sleep(0.1)
            elif self.pattern == 'long':
                self._beep(self.duration * 2)
            else:
                self._beep(self.duration)
                
        except Exception as e:
            logger.error(f"Error playing alert: {e}")
    
    def _beep(self, duration: float):
        """Sound the buzzer for specified duration."""
        if not self._initialized:
            return
        
        try:
            GPIO.output(self.pin, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(self.pin, GPIO.LOW)
        except Exception as e:
            logger.error(f"Beep error: {e}")
    
    def manual_alert(self):
        """Manually trigger an alert (for testing)."""
        if self.enabled and self._initialized:
            self._beep(self.duration)
            logger.info("Manual alert triggered")
    
    def cleanup(self):
        """Cleanup GPIO resources."""
        if self._initialized:
            try:
                GPIO.output(self.pin, GPIO.LOW)
                GPIO.cleanup(self.pin)
                logger.info("Buzzer GPIO cleaned up")
            except Exception as e:
                logger.error(f"GPIO cleanup error: {e}")


class AlertManager:
    """Manages all alert types."""
    
    def __init__(self, config: dict):
        self.config = config
        self.buzzer = BuzzerAlert(config)
        self.alerts_enabled = config.get('alerts', {}).get('enabled', True)
    
    def initialize(self):
        """Initialize all alert systems."""
        if self.alerts_enabled:
            self.buzzer.initialize()
    
    def process_detections(self, detections: list):
        """Process detections and trigger appropriate alerts."""
        if self.alerts_enabled:
            self.buzzer.check_and_alert(detections)
    
    def cleanup(self):
        """Cleanup all alert systems."""
        self.buzzer.cleanup()
