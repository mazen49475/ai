#!/bin/bash

# ============================================
# Trash Bin Detection - Raspberry Pi 4 Setup
# ============================================
# This script automatically installs all dependencies
# and configures the system for trash bin detection
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "============================================"
echo "   Trash Bin Detection System Setup"
echo "   For Raspberry Pi 4 Model B"
echo "============================================"
echo -e "${NC}"

# Check if running on Raspberry Pi
check_raspberry_pi() {
    if [ -f /proc/device-tree/model ]; then
        MODEL=$(cat /proc/device-tree/model)
        if [[ "$MODEL" == *"Raspberry Pi"* ]]; then
            echo -e "${GREEN}✓ Detected: $MODEL${NC}"
            return 0
        fi
    fi
    echo -e "${YELLOW}⚠ Warning: Not running on Raspberry Pi. Some features may not work.${NC}"
    return 0
}

# Update system
update_system() {
    echo -e "${BLUE}[1/8] Updating system packages...${NC}"
    sudo apt-get update
    sudo apt-get upgrade -y
    echo -e "${GREEN}✓ System updated${NC}"
}

# Install system dependencies
install_system_deps() {
    echo -e "${BLUE}[2/8] Installing system dependencies...${NC}"
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        python3-picamera2 \
        libcamera-dev \
        libcamera-apps \
        libatlas-base-dev \
        libjpeg-dev \
        libtiff-dev \
        libpng-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libgtk-3-dev \
        libcanberra-gtk3-dev \
        libhdf5-dev \
        libhdf5-serial-dev \
        libqt5gui5 \
        libqt5webkit5 \
        libqt5test5 \
        git \
        cmake \
        pkg-config
    echo -e "${GREEN}✓ System dependencies installed${NC}"
}

# Enable camera
enable_camera() {
    echo -e "${BLUE}[3/8] Configuring camera...${NC}"
    
    # Check if camera is already enabled in config.txt
    if grep -q "^camera_auto_detect=1" /boot/config.txt 2>/dev/null || \
       grep -q "^camera_auto_detect=1" /boot/firmware/config.txt 2>/dev/null; then
        echo -e "${GREEN}✓ Camera already enabled${NC}"
    else
        # For newer Raspberry Pi OS (Bookworm)
        if [ -f /boot/firmware/config.txt ]; then
            echo -e "${YELLOW}Adding camera configuration to /boot/firmware/config.txt${NC}"
            sudo bash -c 'echo "camera_auto_detect=1" >> /boot/firmware/config.txt'
            sudo bash -c 'echo "dtoverlay=imx219" >> /boot/firmware/config.txt'
        # For older Raspberry Pi OS
        elif [ -f /boot/config.txt ]; then
            echo -e "${YELLOW}Adding camera configuration to /boot/config.txt${NC}"
            sudo bash -c 'echo "camera_auto_detect=1" >> /boot/config.txt'
            sudo bash -c 'echo "dtoverlay=imx219" >> /boot/config.txt'
        fi
        echo -e "${YELLOW}⚠ Camera enabled. Reboot required after setup.${NC}"
    fi
}

# Create virtual environment
setup_venv() {
    echo -e "${BLUE}[4/8] Creating Python virtual environment...${NC}"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
    
    if [ -d "venv" ]; then
        echo -e "${YELLOW}Virtual environment already exists. Recreating...${NC}"
        rm -rf venv
    fi
    
    python3 -m venv venv --system-site-packages
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    echo -e "${GREEN}✓ Virtual environment created${NC}"
}

# Install Python dependencies
install_python_deps() {
    echo -e "${BLUE}[5/8] Installing Python dependencies...${NC}"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
    source venv/bin/activate
    
    # Install from requirements.txt
    pip install -r requirements.txt
    
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
}

# Setup directories
setup_directories() {
    echo -e "${BLUE}[6/8] Setting up directories...${NC}"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
    
    mkdir -p models logs output static templates config
    
    echo -e "${GREEN}✓ Directories created${NC}"
}

# Check for model file
check_model() {
    echo -e "${BLUE}[7/8] Checking for model file...${NC}"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    if [ -f "$SCRIPT_DIR/models/best.pt" ]; then
        echo -e "${GREEN}✓ Model file found${NC}"
    else
        echo -e "${YELLOW}⚠ Model file not found at models/best.pt${NC}"
        echo -e "${YELLOW}  Please copy your best.pt model to the models/ directory${NC}"
        echo -e "${YELLOW}  Command: cp /path/to/best.pt $SCRIPT_DIR/models/best.pt${NC}"
    fi
}

# Create systemd service
create_service() {
    echo -e "${BLUE}[8/8] Creating systemd service...${NC}"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    USER=$(whoami)
    
    sudo bash -c "cat > /etc/systemd/system/trashbin-detector.service << EOF
[Unit]
Description=Trash Bin Detection Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/venv/bin/python $SCRIPT_DIR/src/detector.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF"
    
    sudo systemctl daemon-reload
    echo -e "${GREEN}✓ Systemd service created${NC}"
    echo -e "${YELLOW}  To enable auto-start: sudo systemctl enable trashbin-detector${NC}"
    echo -e "${YELLOW}  To start service: sudo systemctl start trashbin-detector${NC}"
}

# Main installation
main() {
    echo -e "${BLUE}Starting installation...${NC}"
    echo ""
    
    check_raspberry_pi
    update_system
    install_system_deps
    enable_camera
    setup_venv
    install_python_deps
    setup_directories
    check_model
    create_service
    
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}   Installation Complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo -e "Next steps:"
    echo -e "  1. Copy your model: ${YELLOW}cp /path/to/best.pt models/best.pt${NC}"
    echo -e "  2. Reboot (if camera was just enabled): ${YELLOW}sudo reboot${NC}"
    echo -e "  3. Activate environment: ${YELLOW}source venv/bin/activate${NC}"
    echo -e "  4. Run detector: ${YELLOW}python src/detector.py${NC}"
    echo -e "  5. Access web interface: ${YELLOW}http://$(hostname -I | awk '{print $1}'):5000${NC}"
    echo ""
    echo -e "For auto-start on boot:"
    echo -e "  ${YELLOW}sudo systemctl enable trashbin-detector${NC}"
    echo -e "  ${YELLOW}sudo systemctl start trashbin-detector${NC}"
    echo ""
}

# Run main
main "$@"
