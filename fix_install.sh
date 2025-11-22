#!/bin/bash

# Fix SAM3 installation with proper dependency resolution
echo "=== Fixing SAM3 Installation ==="

# Step 1: Uninstall conflicting packages
echo "Step 1: Uninstalling existing sam3 and numpy..."
pip uninstall -y sam3 numpy

# Step 2: Install PyTorch first (for macOS with MPS support)
echo "Step 2: Installing PyTorch..."
pip install torch torchvision torchaudio

# Step 3: Install compatible numpy version
echo "Step 3: Installing compatible NumPy..."
pip install "numpy>=2.0,<2.3"

# Step 4: Install SAM3 in editable mode with notebooks dependencies
echo "Step 4: Installing SAM3 with notebook dependencies..."
pip install -e ".[notebooks]"

# Step 5: Install additional dependencies for the video predictor example
echo "Step 5: Installing additional dependencies..."
pip install opencv-python matplotlib scikit-learn scikit-image einops decord

echo ""
echo "=== Installation Complete ==="
echo "You can now run the notebook with: jupyter notebook examples/sam3_video_predictor_example.ipynb"
