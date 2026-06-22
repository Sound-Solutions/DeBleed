#!/bin/bash
# DeBleed Trainer - Install Dependencies
# Run this once before training

echo "Installing DeBleed Trainer dependencies..."
pip3 install torch torchaudio numpy onnx onnxruntime soundfile

if [ $? -eq 0 ]; then
    echo ""
    echo "Installation complete!"
    echo "Now run: ./train.sh"
else
    echo ""
    echo "Installation failed. Make sure Python 3 and pip3 are installed."
    exit 1
fi
