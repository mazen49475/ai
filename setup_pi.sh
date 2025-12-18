#!/bin/bash
# =============================================================================
# Trash Bin Detection - Raspberry Pi 4 64-bit Setup Script
# =============================================================================

echo "============================================================"
echo "Trash Bin Detection - Setup Script"
echo "Raspberry Pi 4 64-bit"
echo "============================================================"

# Check if running on Raspberry Pi
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    echo "Device: $MODEL"
else
    echo "Warning: Not running on Raspberry Pi"
fi

# Update system packages
echo ""
echo "Updating system packages..."
sudo apt-get update

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libopencv-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libcamera-dev \
    libcap-dev \
    python3-libcamera \
    python3-picamera2

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "Creating directories..."
mkdir -p models
mkdir -p output
mkdir -p logs

# Copy model if in root directory
if [ -f "best.pt" ] && [ ! -f "models/best.pt" ]; then
    echo "Moving model to models directory..."
    cp best.pt models/best.pt
fi

# Set permissions
echo ""
echo "Setting permissions..."
chmod +x run_detector.sh
chmod +x setup_pi.sh

# Enable camera if on Raspberry Pi
if [ -f /boot/config.txt ]; then
    echo ""
    echo "Note: Make sure camera is enabled in raspi-config"
    echo "Run: sudo raspi-config -> Interface Options -> Camera -> Enable"
fi

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "To run the detector:"
echo "  ./run_detector.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python3 src/detector.py"
echo ""
echo "Buzzer is connected to GPIO17"
echo "Confidence threshold: 70%"
echo "============================================================"
