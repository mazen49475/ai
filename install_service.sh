#!/bin/bash
# Install Trash Bin Detection as a system service
# Run: sudo bash install_service.sh

echo "Installing Trash Bin Detection Service..."

# Copy service file
sudo cp trashbin.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable trashbin.service

# Start service now
sudo systemctl start trashbin.service

echo ""
echo "✓ Service installed and started!"
echo ""
echo "Commands:"
echo "  sudo systemctl status trashbin    # Check status"
echo "  sudo systemctl stop trashbin      # Stop service"
echo "  sudo systemctl start trashbin     # Start service"
echo "  sudo systemctl restart trashbin   # Restart service"
echo "  sudo systemctl disable trashbin   # Disable auto-start"
echo "  sudo journalctl -u trashbin -f    # View logs"
echo ""
echo "Web interface: http://$(hostname -I | awk '{print $1}'):5000"
