# DeBleed Build Instructions

## Prerequisites

### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install CMake (via Homebrew)
brew install cmake

# Install Python dependencies
pip3 install torch torchaudio numpy onnx onnxruntime soundfile
```

### Linux
```bash
# Install build tools
sudo apt-get install build-essential cmake git

# Install JUCE dependencies
sudo apt-get install libasound2-dev libcurl4-openssl-dev libfreetype6-dev \
    libx11-dev libxcomposite-dev libxcursor-dev libxext-dev libxinerama-dev \
    libxrandr-dev libxrender-dev libwebkit2gtk-4.0-dev libglu1-mesa-dev

# Install Python dependencies
pip3 install torch torchaudio numpy onnx onnxruntime soundfile
```

### Windows
1. Install Visual Studio 2019 or later with C++ workload
2. Install CMake (add to PATH)
3. Install Python 3.9+ and pip install dependencies

## Building the Plugin

### Quick Build (macOS/Linux)
```bash
# Clone or navigate to project
cd DeBleed

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release -j$(nproc)
```

### With Custom ONNX Runtime Path
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/path/to/onnxruntime
```

### Build Outputs

After building, you'll find:

- **VST3**: `build/plugin/DeBleed_artefacts/Release/VST3/DeBleed.vst3`
- **AU** (macOS): `build/plugin/DeBleed_artefacts/Release/AU/DeBleed.component`
- **Standalone**: `build/plugin/DeBleed_artefacts/Release/Standalone/DeBleed`

## Installing the Plugin

### macOS
```bash
# VST3
cp -r build/plugin/DeBleed_artefacts/Release/VST3/DeBleed.vst3 ~/Library/Audio/Plug-Ins/VST3/

# AU
cp -r build/plugin/DeBleed_artefacts/Release/AU/DeBleed.component ~/Library/Audio/Plug-Ins/Components/
```

### Linux
```bash
cp -r build/plugin/DeBleed_artefacts/Release/VST3/DeBleed.vst3 ~/.vst3/
```

### Windows
Copy `DeBleed.vst3` to `C:\Program Files\Common Files\VST3\`

## Testing the Python Trainer

Before building the plugin, verify the Python trainer works:

```bash
cd python
python3 test_trainer.py
```

Expected output:
```
=== DeBleed Trainer Test ===
...
ALL TESTS PASSED!
```

## Project Structure

```
DeBleed/
├── CMakeLists.txt          # Top-level build config
├── BUILD.md                # This file
├── python/
│   ├── trainer.py          # Neural network trainer
│   ├── test_trainer.py     # Trainer tests
│   └── requirements.txt    # Python dependencies
├── plugin/
│   ├── CMakeLists.txt      # Plugin build config
│   └── Source/
│       ├── PluginProcessor.cpp/h   # Audio processing
│       ├── PluginEditor.cpp/h      # GUI
│       ├── NeuralGateEngine.cpp/h  # ONNX inference
│       ├── STFTProcessor.cpp/h     # STFT/iSTFT
│       ├── TrainerProcess.cpp/h    # Python subprocess IPC
│       └── AudioDropZone.cpp/h     # Drag-drop component
└── models/                 # Trained models stored here
```

## Usage

1. Open the plugin in your DAW
2. Drag a folder of **clean vocal** recordings to the left drop zone
3. Drag a folder of **stage bleed** recordings to the right drop zone
4. Click "Train Model" and wait for training to complete
5. Adjust **Strength** (mask intensity) and **Mix** (dry/wet) as needed

## Troubleshooting

### "Trainer executable not found"
Ensure Python 3 is in your PATH and the trainer.py script is accessible.

### ONNX Runtime errors
Make sure ONNX Runtime is properly installed and the library path is set:
- macOS: `DYLD_LIBRARY_PATH=/path/to/onnxruntime/lib`
- Linux: `LD_LIBRARY_PATH=/path/to/onnxruntime/lib`

### Plugin doesn't load
Check that the ONNX Runtime dylib/so is copied alongside the plugin bundle.
