#!/bin/bash
# Touches plugin bundles to update mtime for DAW detection

PRODUCT_NAME="$1"
BUILT_PRODUCTS_DIR="$2"

# Exit gracefully if no product name (helper targets)
[ -z "$PRODUCT_NAME" ] && exit 0
[ -z "$BUILT_PRODUCTS_DIR" ] && exit 0

# Touch build output
[ -d "${BUILT_PRODUCTS_DIR}/${PRODUCT_NAME}.vst3" ] && touch "${BUILT_PRODUCTS_DIR}/${PRODUCT_NAME}.vst3"
[ -d "${BUILT_PRODUCTS_DIR}/${PRODUCT_NAME}.component" ] && touch "${BUILT_PRODUCTS_DIR}/${PRODUCT_NAME}.component"

# Touch installed copies
[ -d "$HOME/Library/Audio/Plug-Ins/VST3/${PRODUCT_NAME}.vst3" ] && touch "$HOME/Library/Audio/Plug-Ins/VST3/${PRODUCT_NAME}.vst3"
[ -d "$HOME/Library/Audio/Plug-Ins/Components/${PRODUCT_NAME}.component" ] && touch "$HOME/Library/Audio/Plug-Ins/Components/${PRODUCT_NAME}.component"

exit 0
