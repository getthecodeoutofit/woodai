#!/bin/bash
# Fix dependency conflicts after installing requirements.txt
# This script resolves the unstructured-inference / onnxruntime conflict

echo "🔧 Fixing dependency conflicts..."

# Uninstall unstructured-inference to resolve onnxruntime conflict
# unstructured-inference requires onnxruntime<1.16, but chromadb needs >=1.16
if pip show unstructured-inference > /dev/null 2>&1; then
    echo "⚠️  Removing unstructured-inference (conflicts with onnxruntime 1.16+)..."
    pip uninstall -y unstructured-inference
    echo "✅ Removed unstructured-inference"
else
    echo "ℹ️  unstructured-inference not installed, skipping..."
fi

echo "✅ Dependency conflicts resolved!"
echo ""
echo "Note: unstructured-inference is optional and only needed for advanced"
echo "      document layout detection. The base unstructured package works"
echo "      fine without it for most use cases."


