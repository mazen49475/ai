"""
Trash Bin Detection - Alert System Module
Buzzer stays ON while garbage container is detected, OFF when not
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
    logger.warning("RPi.GPIO not available - running in simulation mode")


class BuzzerAlert:
    """Manages buzzer - stays ON while garbage container detected."""

    def __init__(self, config: dict):
        self.config = config
        self.alert_config = config.get('alerts', {})
        self.gpio_config = self.alert_config.get('gpio', {})

        self.enabled = self.gpio_config.get('enabled', True)
        self.pin = self.gpio_config.get('pin', 17)
        self.min_confidence = self.alert_config.get('min_confidence', 0.7)
        self.alert_classes = self.alert_config.get('alert_classes', ['garbage container'])

        self.lock = Lock()
        self._initialized = False
        self.buzzer_on = False

    def initialize(self):
        if not GPIO_AVAILABLE:
            logger.info("GPIO not available - simulation mode")
            self._initialized = True
            return True

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, GPIO.LOW)
            self._initialized = True
            self.buzzer_on = False
            logger.info(f"Buzzer initialized on GPIO {self.pin}")
            
            # Test beep
            self._buzzer_on()
            time.sleep(0.2)
            self._buzzer_off()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize buzzer: {e}")
            self._initialized = True
            return False

    def check_and_alert(self, detections: list):
        if not self._initialized:
            return False

        target_detected = False
        for det in detections:
            class_name = det.get('class_name', '').lower()
            confidence = det.get('confidence', 0)

            for alert_class in self.alert_classes:
                alert_lower = alert_class.lower()
                if (alert_lower in class_name or class_name in alert_lower or
                    'garbage' in class_name or 'container' in class_name):
                    if confidence >= self.min_confidence:
                        target_detected = True
                        break
            if target_detected:
                break

        if target_detected:
            if not self.buzzer_on:
                self._buzzer_on()
                logger.info("BUZZER ON - Garbage container detected!")
        else:
            if self.buzzer_on:
                self._buzzer_off()
                logger.info("BUZZER OFF - No detection")

        return target_detected

    def _buzzer_on(self):
        with self.lock:
            if GPIO_AVAILABLE:
                try:
                    GPIO.output(self.pin, GPIO.HIGH)
                except Exception as e:
                    logger.error(f"Buzzer ON error: {e}")
            self.buzzer_on = True

    def _buzzer_off(self):
        with self.lock:
            if GPIO_AVAILABLE:
                try:
                    GPIO.output(self.pin, GPIO.LOW)
                except Exception as e:
                    logger.error(f"Buzzer OFF error: {e}")
            self.buzzer_on = False

    def cleanup(self):
        self._buzzer_off()
        if GPIO_AVAILABLE and self._initialized:
            try:
                GPIO.cleanup(self.pin)
                logger.info("GPIO cleaned up")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


class AlertManager:
    def __init__(self, config: dict):
        self.config = config
        self.buzzer = BuzzerAlert(config)
        self.alerts_enabled = config.get('alerts', {}).get('enabled', True)

    def initialize(self):
        if self.alerts_enabled:
            self.buzzer.initialize()

    def process_detections(self, detections: list):
        if self.alerts_enabled:
            return self.buzzer.check_and_alert(detections)
        return False

    @property
    def buzzer_on(self):
        return self.buzzer.buzzer_on

    def cleanup(self):
        self.buzzer.cleanup()
