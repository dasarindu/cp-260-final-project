"""
Configuration and path settings for the 3D reconstruction pipeline.
"""
import os
import json
import numpy as np

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
INTRINSIC_PATH = os.path.join(PROJECT_ROOT, "intrinsic.json")
POSES_PATH = os.path.join(DATA_DIR, "poses.json")
SAMPLE_ANSWERS_PATH = os.path.join(PROJECT_ROOT, "sample_answers.json")

# Load camera intrinsics from JSON
with open(INTRINSIC_PATH, 'r') as f:
    _intrinsic_data = json.load(f)

K_MATRIX = np.array(_intrinsic_data['camera_matrix'], dtype=np.float64)
IMG_W = _intrinsic_data['image_width']
IMG_H = _intrinsic_data['image_height']

# Frame indices we have images for
FRAME_IDS = [319, 333, 353, 359, 365, 371, 390, 400,
             426, 449, 461, 468, 471, 496, 515, 531]

# Feature extraction and matching
DOWNSCALE = 0.4
MAX_FEATURES = 5000
LOWE_RATIO = 0.7
REPROJ_THRESH = 3.5  # pixels

# Point cloud filtering
VOXEL_SIZE = 0.003
OUTLIER_NEIGHBORS = 25
OUTLIER_STD = 1.8

# Create output directories
for d in ["", "annotations"]:
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)