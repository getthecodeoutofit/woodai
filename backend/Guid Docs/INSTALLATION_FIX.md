# Installation Fix for TensorFlow/Keras Compatibility

## Issue
When running `python main.py`, you may encounter:
```
ValueError: Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers. Please install the backwards-compatible tf-keras package with `pip install tf-keras`.
```

## Root Cause
- `easyocr` (used for OCR) imports TensorFlow
- When TensorFlow is detected, `transformers` tries to use TensorFlow modules
- `transformers` requires `tf-keras` (Keras 2 compatibility) when TensorFlow is present
- Even though we use PyTorch, the presence of TensorFlow triggers this requirement

## Solution

### Step 1: Install tf-keras
```bash
cd /home/ravi/woodai/backend
pip install tf-keras
```

Or install all requirements (which includes tf-keras):
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python -c "import tf_keras; print('tf-keras installed successfully')"
```

### Step 3: Run the Application
```bash
python main.py
```

## What We Fixed

1. **Added tf-keras to requirements.txt** - This package provides Keras 2 compatibility for transformers
2. **Set environment variables early** - `USE_TF=0` and `USE_TORCH=1` are set before any imports
3. **Made easyocr import lazy** - EasyOCR is now imported only when needed (in `AdvancedOCREngine.__init__`), not at module level
4. **Updated python-docx** - Fixed version conflict with unstructured package

## Notes

- The TensorFlow warnings (AttributeError: 'MessageFactory' object has no attribute 'GetPrototype') are harmless and can be ignored
- EasyOCR will still work correctly - it just imports TensorFlow when needed for OCR operations
- The application uses PyTorch for all ML operations, TensorFlow is only used by EasyOCR for OCR

## Alternative: Disable EasyOCR

If you don't need OCR functionality, you can disable EasyOCR by not installing it:
```bash
pip uninstall easyocr
```

The application will fall back to Tesseract OCR if available, or skip OCR entirely.


