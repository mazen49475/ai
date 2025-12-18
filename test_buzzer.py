#!/usr/bin/env python3
"""
Test script for GPIO17 buzzer on Raspberry Pi 4
Run this to verify buzzer wiring before running the main detector
"""

import time
import sys

print("=" * 60)
print("Buzzer Test - GPIO17")
print("Raspberry Pi 4 64-bit")
print("=" * 60)

try:
    import RPi.GPIO as GPIO
    print("✓ RPi.GPIO library loaded")
except ImportError:
    print("✗ RPi.GPIO not available")
    print("  This script must be run on Raspberry Pi")
    print("  Install with: pip install RPi.GPIO")
    sys.exit(1)

BUZZER_PIN = 17  # GPIO17 (BCM numbering)

try:
    # Setup GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    print(f"✓ GPIO{BUZZER_PIN} configured as output")
    
    print("\n" + "=" * 60)
    print("Starting buzzer test...")
    print("You should hear the buzzer beep 3 times")
    print("=" * 60 + "\n")
    
    # Test pattern: 3 short beeps
    for i in range(3):
        print(f"Beep {i+1}/3 - BUZZER ON")
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(0.3)
        
        print(f"Beep {i+1}/3 - BUZZER OFF")
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        time.sleep(0.3)
    
    print("\n" + "=" * 60)
    print("Now testing continuous ON/OFF (5 seconds each)...")
    print("=" * 60 + "\n")
    
    # Test continuous ON
    print("BUZZER ON for 5 seconds...")
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(5)
    
    # Test OFF
    print("BUZZER OFF for 5 seconds...")
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(5)
    
    print("\n" + "=" * 60)
    print("✓ Buzzer test PASSED!")
    print("  Buzzer is working correctly on GPIO17")
    print("=" * 60)
    
except KeyboardInterrupt:
    print("\n\nTest interrupted by user")
except Exception as e:
    print(f"\n✗ Buzzer test FAILED: {e}")
    sys.exit(1)
finally:
    # Cleanup
    try:
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        GPIO.cleanup(BUZZER_PIN)
        print("GPIO cleaned up")
    except:
        pass
