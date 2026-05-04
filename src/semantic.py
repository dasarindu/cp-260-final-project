"""
Semantic annotation loading and visualization.
Annotations are stored in annotations.json and loaded at runtime.
"""
import os
import json
import cv2
import numpy as np
from . import data

# Load annotations from JSON file
_ANNOTATIONS_PATH = os.path.join(data.PROJECT_ROOT, "annotations.json")
with open(_ANNOTATIONS_PATH, 'r') as _f:
    _raw = json.load(_f)

# Convert string keys to int keys
ANNOTATIONS = {}
for entity, frames in _raw.items():
    ANNOTATIONS[entity] = {int(k): v for k, v in frames.items()}

ENTITY_COLORS = {
    "power_socket": (0, 0, 255),
    "ethernet_socket": (255, 0, 0),
    "vga_socket": (0, 255, 0),
}


def get_annotations(entity_name=None):
    """Get annotations for one entity or all."""
    if entity_name is None:
        return ANNOTATIONS
    return ANNOTATIONS.get(entity_name, {})


def get_entity_names():
    """Return list of all annotated entity names."""
    return list(ANNOTATIONS.keys())


def get_annotated_frames(entity_name):
    """Return frame indices where this entity is annotated."""
    return list(ANNOTATIONS.get(entity_name, {}).keys())


def get_roi_center(bbox):
    """Get center point (u, v) of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_roi_points(bbox, n_samples=100):
    """Sample a grid of points inside a bounding box."""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    nx = max(int(np.sqrt(n_samples * w / h)), 3)
    ny = max(int(np.sqrt(n_samples * h / w)), 3)

    xs = np.linspace(x1, x2, nx)
    ys = np.linspace(y1, y2, ny)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def visualize_annotations(images, output_dir=None):
    """Draw bounding boxes on images and save."""
    if output_dir is None:
        output_dir = os.path.join(data.OUTPUT_DIR, "annotations")

    for idx, img in images.items():
        vis = img.copy()
        has_any = False

        for entity_name, ann_dict in ANNOTATIONS.items():
            if idx in ann_dict:
                bbox = ann_dict[idx]
                color = ENTITY_COLORS.get(entity_name, (0, 255, 255))
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                cv2.putText(vis, entity_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                has_any = True

        if has_any:
            out_path = os.path.join(output_dir, f"annotated_{idx:06d}.png")
            cv2.imwrite(out_path, vis)
            print(f"  Saved: {out_path}")
