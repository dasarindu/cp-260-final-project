"""
Utility functions for OBB projection, visualization, and output saving.
"""
import json
import cv2
import numpy as np


def project_obb_to_image(obb, K, pose_c2w):
    """
    Project an OBB's 8 corners onto an image.
    Returns (8, 2) array of 2D points, or None if all corners are behind camera.
    """
    center = np.array(obb['center'])
    extent = np.array(obb['extent'])
    R_obb = np.array(obb['rotation'])

    # 8 corner offsets in local OBB frame
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ], dtype=np.float64)

    # Transform corners to world frame
    corners_local = signs * extent
    corners_world = (R_obb @ corners_local.T).T + center

    # World to camera
    w2c = np.linalg.inv(pose_c2w)
    R_cam = w2c[:3, :3]
    t_cam = w2c[:3, 3]
    corners_cam = (R_cam @ corners_world.T).T + t_cam

    if not np.any(corners_cam[:, 2] > 0):
        return None

    # Project to 2D
    corners_2d = np.full((8, 2), np.nan)
    for i in range(8):
        if corners_cam[i, 2] > 0:
            pt = K @ corners_cam[i]
            corners_2d[i] = pt[:2] / pt[2]

    return corners_2d


def draw_obb_on_image(image, corners_2d, label="", color=(0, 255, 0), thickness=2):
    """
    Draw a projected OBB as a wireframe on an image.
    Connects the 12 edges of the box between the 8 projected corners.
    """
    if corners_2d is None:
        return image

    vis = image.copy()
    pts = corners_2d.astype(int)

    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    for i, j in edges:
        if not (np.isnan(pts[i]).any() or np.isnan(pts[j]).any()):
            cv2.line(vis, tuple(pts[i]), tuple(pts[j]), color, thickness)

    if label:
        valid = pts[~np.isnan(pts).any(axis=1)]
        if len(valid) > 0:
            top = valid.min(axis=0)
            cv2.putText(vis, label, tuple(top - [0, 10]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return vis


def save_answers_json(entities, output_path):
    """
    Save OBB results in the submission format.
    Each entry has 'entity', and 'obb' with 'center', 'extent', 'rotation'.
    """
    output = []
    for ent in entities:
        entry = {
            "entity": ent["entity"],
            "obb": {
                "center": [float(v) for v in ent["obb"]["center"]],
                "extent": [float(v) for v in ent["obb"]["extent"]],
                "rotation": [
                    [float(v) for v in row]
                    for row in ent["obb"]["rotation"]
                ]
            }
        }
        output.append(entry)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {output_path}")


def validate_against_sample(answers, sample_answers):
    """
    Compare VGA socket OBB against the sample answer.
    Prints center distance, extent comparison, and rotation error.
    """
    sample_vga = None
    our_vga = None
    for s in sample_answers:
        if s['entity'] == 'vga_socket':
            sample_vga = s['obb']
    for a in answers:
        if a['entity'] == 'vga_socket':
            our_vga = a['obb']

    if sample_vga is None:
        print("  [INFO] No VGA socket in sample answers")
        return
    if our_vga is None:
        print("  [INFO] No VGA socket in our answers")
        return

    # Center comparison
    c_gt = np.array(sample_vga['center'])
    c_ours = np.array(our_vga['center'])
    dist = np.linalg.norm(c_gt - c_ours)
    print(f"\n  --- VGA Socket Validation ---")
    print(f"  Center distance: {dist:.4f} m ({dist*1000:.1f} mm)")
    print(f"    Ground truth: [{c_gt[0]:.6f}, {c_gt[1]:.6f}, {c_gt[2]:.6f}]")
    print(f"    Ours:         [{c_ours[0]:.6f}, {c_ours[1]:.6f}, {c_ours[2]:.6f}]")

    # Extent comparison
    e_gt = np.array(sample_vga['extent'])
    e_ours = np.array(our_vga['extent'])
    print(f"\n  Extents (half-sizes in meters):")
    print(f"    Ground truth: [{e_gt[0]:.6f}, {e_gt[1]:.6f}, {e_gt[2]:.6f}]")
    print(f"    Ours:         [{e_ours[0]:.6f}, {e_ours[1]:.6f}, {e_ours[2]:.6f}]")

    # Rotation comparison
    R_gt = np.array(sample_vga['rotation'])
    R_ours = np.array(our_vga['rotation'])

    try:
        R_diff = R_ours @ R_gt.T
        trace = np.clip(np.trace(R_diff), -1.0, 3.0)
        angle = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
        print(f"\n  Rotation error: {angle:.1f} degrees")
    except Exception as e:
        print(f"\n  Rotation comparison failed: {e}")

    # Volume comparison
    vol_gt = np.prod(e_gt) * 8
    vol_ours = np.prod(e_ours) * 8
    print(f"\n  Volume: ground truth={vol_gt:.2e} m^3, ours={vol_ours:.2e} m^3")
    if vol_gt > 0:
        print(f"  Volume ratio: {vol_ours/vol_gt:.2f}x")
