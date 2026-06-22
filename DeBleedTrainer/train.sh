#!/bin/bash
# DeBleed Trainer - Training Script
#
# USAGE:
#   ./train.sh <clean_audio_folder> <noise_audio_folder> [epochs]
#
# EXAMPLE:
#   ./train.sh "Millie for training" "Stage Noise for Training" 50
#
# Put your audio folders in the same directory as this script,
# or provide full paths.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default epochs
EPOCHS="${3:-50}"

# Check arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "DeBleed Neural Trainer"
    echo "======================"
    echo ""
    echo "Usage: ./train.sh <clean_folder> <noise_folder> [epochs]"
    echo ""
    echo "Example:"
    echo "  ./train.sh \"Millie for training\" \"Stage Noise for Training\" 50"
    echo ""
    echo "Put your audio folders in this directory:"
    echo "  $SCRIPT_DIR"
    echo ""
    exit 1
fi

# Resolve paths (check if relative to script dir or absolute)
if [ -d "$SCRIPT_DIR/$1" ]; then
    CLEAN_DIR="$SCRIPT_DIR/$1"
elif [ -d "$1" ]; then
    CLEAN_DIR="$1"
else
    echo "ERROR: Clean audio folder not found: $1"
    echo "Put it in: $SCRIPT_DIR"
    exit 1
fi

if [ -d "$SCRIPT_DIR/$2" ]; then
    NOISE_DIR="$SCRIPT_DIR/$2"
elif [ -d "$2" ]; then
    NOISE_DIR="$2"
else
    echo "ERROR: Noise audio folder not found: $2"
    echo "Put it in: $SCRIPT_DIR"
    exit 1
fi

# Create output directory
OUTPUT_DIR="$SCRIPT_DIR/trained_model"
mkdir -p "$OUTPUT_DIR"

echo "DeBleed Neural Trainer"
echo "======================"
echo "Clean audio: $CLEAN_DIR"
echo "Noise audio: $NOISE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo ""
echo "Starting training..."
echo ""

python3 "$SCRIPT_DIR/neural5045_trainer.py" \
    --clean_audio_dir "$CLEAN_DIR" \
    --noise_audio_dir "$NOISE_DIR" \
    --output_path "$OUTPUT_DIR" \
    --epochs "$EPOCHS"

if [ $? -eq 0 ]; then
    echo ""
    echo "Training complete!"
    echo "Model saved to: $OUTPUT_DIR"
else
    echo ""
    echo "Training failed. Check the error messages above."
    exit 1
fi
