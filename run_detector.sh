#!/bin/bash
# =============================================================================
# Trash Bin Detection - Raspberry Pi 4 64-bit Run Script
# Buzzer on GPIO17: ON when class detected (>=70%), OFF when not detected
# =============================================================================

echo "============================================================"
echo "Trash Bin Detection System"
echo "Raspberry Pi 4 64-bit Edition"
echo "============================================================"
echo "Buzzer: GPIO17 (BCM)"
echo "Confidence Threshold: 70%"
echo "============================================================"

# Change to script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if running on Raspberry Pi
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    echo "Device: $MODEL"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python: $PYTHON_VERSION"

# Check if model exists
if [ ! -f "models/best.pt" ]; then
    if [ -f "best.pt" ]; then
        echo "Moving model to models directory..."
        mkdir -p models
        cp best.pt models/best.pt
    else
        echo "ERROR: Model file not found!"
        echo "Please ensure best.pt is in the models/ directory"
        exit 1
    fi
fi

echo "============================================================"
echo "Starting detection system..."
echo "Press Ctrl+C to stop"
echo "============================================================"

# Run the detector
python3 src/detector.py --config config/config.yaml --confidence 0.70 "$@"
