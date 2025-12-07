#!/bin/bash
# Script to fix onnxruntime executable stack issue
# This script attempts to fix the "cannot enable executable stack" error

echo "Attempting to fix onnxruntime executable stack issue..."

# Find the onnxruntime library
ONNXRUNTIME_LIB=$(python -c "import onnxruntime; import os; print(os.path.dirname(onnxruntime.__file__))")/capi/onnxruntime_pybind11_state*.so

if [ -f "$ONNXRUNTIME_LIB" ]; then
    echo "Found onnxruntime library: $ONNXRUNTIME_LIB"
    
    # Try to fix using execstack if available
    if command -v execstack &> /dev/null; then
        echo "Using execstack to fix the binary..."
        execstack -c $ONNXRUNTIME_LIB
        echo "✅ Fixed using execstack"
    else
        echo "⚠️  execstack not found. Trying alternative method..."
        # Try using setarch
        if command -v setarch &> /dev/null; then
            echo "setarch is available, but this won't fix the binary directly."
            echo "You may need to install execstack:"
            echo "  Arch Linux: sudo pacman -S prelink (includes execstack)"
            echo "  Ubuntu/Debian: sudo apt-get install execstack"
        fi
        
        # Alternative: try to use chrpath or patchelf if available
        if command -v patchelf &> /dev/null; then
            echo "Trying patchelf..."
            patchelf --set-rpath $(dirname $ONNXRUNTIME_LIB) $ONNXRUNTIME_LIB 2>/dev/null || true
        fi
    fi
else
    echo "⚠️  Could not find onnxruntime library"
    echo "   Library path: $ONNXRUNTIME_LIB"
fi

echo ""
echo "If the issue persists, you can try:"
echo "1. Reinstall onnxruntime: pip uninstall onnxruntime && pip install onnxruntime"
echo "2. Install execstack and run: execstack -c <path_to_library>"
echo "3. Use a different embedding backend that doesn't require onnxruntime"

