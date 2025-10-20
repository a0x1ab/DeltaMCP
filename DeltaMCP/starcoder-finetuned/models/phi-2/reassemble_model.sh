#!/bin/bash
# Script to reassemble the split model file
# Usage: ./reassemble_model.sh

echo "Reassembling model-00001-of-00002.safetensors..."

# Check if all parts exist
if [ ! -f "model-00001-of-00002.safetensors.partaa" ] || \
   [ ! -f "model-00001-of-00002.safetensors.partab" ] || \
   [ ! -f "model-00001-of-00002.safetensors.partac" ] || \
   [ ! -f "model-00001-of-00002.safetensors.partad" ]; then
    echo "Error: Not all parts found!"
    echo "Required files:"
    echo "  - model-00001-of-00002.safetensors.partaa"
    echo "  - model-00001-of-00002.safetensors.partab"
    echo "  - model-00001-of-00002.safetensors.partac"
    echo "  - model-00001-of-00002.safetensors.partad"
    exit 1
fi

# Remove existing file if it exists
if [ -f "model-00001-of-00002.safetensors" ]; then
    echo "Removing existing model-00001-of-00002.safetensors..."
    rm "model-00001-of-00002.safetensors"
fi

# Concatenate all parts
echo "Concatenating parts..."
cat model-00001-of-00002.safetensors.part* > model-00001-of-00002.safetensors

# Verify the file size
expected_size=4995584424
actual_size=$(stat -f%z "model-00001-of-00002.safetensors" 2>/dev/null || stat -c%s "model-00001-of-00002.safetensors" 2>/dev/null)

if [ "$actual_size" = "$expected_size" ]; then
    echo "✅ Successfully reassembled model-00001-of-00002.safetensors"
    echo "   Size: $actual_size bytes (matches expected size)"
else
    echo "❌ Error: File size mismatch!"
    echo "   Expected: $expected_size bytes"
    echo "   Actual: $actual_size bytes"
    exit 1
fi