"""
Test script to verify all dependencies are installed correctly
"""

import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print("✓ NumPy installed:", np.__version__)
except:
    print("✗ NumPy not found")

try:
    import cv2
    print("✓ OpenCV installed:", cv2.__version__)
except:
    print("✗ OpenCV not found")

try:
    import torch
    print("✓ PyTorch installed:", torch.__version__)
except:
    print("✗ PyTorch not found")

try:
    import pandas as pd
    print("✓ Pandas installed:", pd.__version__)
except:
    print("✗ Pandas not found")

try:
    from ultralytics import YOLO
    print("✓ Ultralytics (YOLO) installed")
except:
    print("✗ Ultralytics not found")

try:
    import streamlit
    print("✓ Streamlit installed:", streamlit.__version__)
except:
    print("✗ Streamlit not found")

print("\n✅ Setup verification complete!")