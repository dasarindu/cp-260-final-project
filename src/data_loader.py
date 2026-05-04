"""
Data loading utilities for images, camera poses, and intrinsics.
"""
import os
import re
import json
import cv2
import numpy as np
from . import config


def load_poses(path=None):
    """
    Load camera poses from poses.json.
    Returns dict mapping frame index (int) to 4x4 camera-to-world matrix.
    """
    if path is None:
        path = config.POSES_PATH
    with open(path, 'r') as f:
        raw = json.load(f)
    poses = {}
    for key, mat in raw.items():
        poses[int(key)] = np.array(mat, dtype=np.float64)
    return poses


def load_images(frame_ids=None, scale=1.0):
    """
    Load images for given frame indices from the Data directory.
    Returns dict mapping frame index to BGR numpy array.
    """
    if frame_ids is None:
        frame_ids = config.FRAME_IDS

    images = {}
    for idx in frame_ids:
        fname = f"frame_{idx:06d}.png"
        fpath = os.path.join(config.DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [WARN] Image not found: {fpath}")
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
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale
    K_scaled[1, 2] *= scale
    return K_scaled


def build_projection(K, pose_c2w):
    """
    Build 3x4 projection matrix P = K @ [R|t] from intrinsics
    and a camera-to-world pose (inverts to get world-to-camera).
    """
    w2c = np.linalg.inv(pose_c2w)
    Rt = w2c[:3, :]  # top 3 rows of 4x4 = [R | t] already
    return K @ Rt


def load_sample_answers(path=None):
    """
    Load sample answers JSON. Handles placeholder values (X, Y, Z etc.)
    by replacing them with 0.0 so the file can be parsed.
    """
    if path is None:
        path = config.SAMPLE_ANSWERS_PATH

    with open(path, 'r') as f:
        raw = f.read()

    # Try parsing directly first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Replace placeholder letters with 0.0
    for placeholder in ['X', 'Y', 'Z', 'W', 'H', 'L', 'rx', 'ry', 'rz']:
        raw = re.sub(r'(?<!["\w])' + placeholder + r'(?!["\w])', '0.0', raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [WARN] Could not parse sample_answers.json: {e}")
        return []


def load_dataset():
    """
    Load everything at once: images, poses, and intrinsic matrix.
    Returns (images_dict, poses_dict, K_matrix).
    """
    print("  Loading poses...")
    all_poses = load_poses()
    poses = {idx: all_poses[idx] for idx in config.FRAME_IDS if idx in all_poses}

    print("  Loading images...")
    images = load_images()

    K = config.K_MATRIX
    print(f"  Loaded {len(images)} images, {len(poses)} poses")
    return images, poses, K
