"""
Data loading, configuration, and camera utilities.
Handles paths, intrinsics, poses, and image loading.
"""
import os
import re
import json
import cv2
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
REPROJ_THRESH = 3.5

# Point cloud filtering
VOXEL_SIZE = 0.003
OUTLIER_NEIGHBORS = 25
OUTLIER_STD = 1.8

# Create output directories
for _d in ["", "annotations"]:
    os.makedirs(os.path.join(OUTPUT_DIR, _d), exist_ok=True)


def load_poses(path=None):
    """Load camera poses. Returns {frame_id(int): 4x4 c2w matrix}."""
    if path is None:
        path = POSES_PATH
    with open(path, 'r') as f:
        raw = json.load(f)
    return {int(k): np.array(v, dtype=np.float64) for k, v in raw.items()}


def load_images(frame_ids=None, scale=1.0):
    """Load images as BGR arrays. Returns {frame_id: image}."""
    if frame_ids is None:
        frame_ids = FRAME_IDS

    images = {}
    for idx in frame_ids:
        fpath = os.path.join(DATA_DIR, f"frame_{idx:06d}.png")
        if not os.path.exists(fpath):
            print(f"  [WARN] Not found: {fpath}")
            continue
        img = cv2.imread(fpath)
        if img is None:
            print(f"  [WARN] Failed to read: {fpath}")
            continue
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        images[idx] = img
    return images


def scale_K(K, scale):
    """Scale intrinsic matrix by a resize factor."""
    Ks = K.copy()
    Ks[0, 0] *= scale
    Ks[1, 1] *= scale
    Ks[0, 2] *= scale
    Ks[1, 2] *= scale
    return Ks


def build_projection(K, pose_c2w):
    """Build 3x4 projection matrix from intrinsics and c2w pose."""
    w2c = np.linalg.inv(pose_c2w)
    return K @ w2c[:3, :]


def load_sample_answers(path=None):
    """Load sample answers JSON, replacing placeholders with 0.0."""
    if path is None:
        path = SAMPLE_ANSWERS_PATH

    with open(path, 'r') as f:
        raw = f.read()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    for ph in ['X', 'Y', 'Z', 'W', 'H', 'L', 'rx', 'ry', 'rz']:
        raw = re.sub(r'(?<!["\w])' + ph + r'(?!["\w])', '0.0', raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [WARN] Could not parse sample_answers.json: {e}")
        return []


def load_dataset():
    """Load everything: images, poses, K. Returns (images, poses, K)."""
    print("  Loading poses...")
    all_poses = load_poses()
    poses = {idx: all_poses[idx] for idx in FRAME_IDS if idx in all_poses}

    print("  Loading images...")
    images = load_images()

    print(f"  Loaded {len(images)} images, {len(poses)} poses")
    return images, poses, K_MATRIX
