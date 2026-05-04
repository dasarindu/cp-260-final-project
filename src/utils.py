"""
Utility functions for OBB projection, visualization, and output.
"""
import json
import cv2
import numpy as np


def project_obb_to_image(obb, K, pose_c2w):
    """Project OBB's 8 corners onto an image. Returns (8, 2) or None."""
    center = np.array(obb['center'])
    extent = np.array(obb['extent'])
    R_obb = np.array(obb['rotation'])

    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ], dtype=np.float64)

    corners_world = (R_obb @ (signs * extent).T).T + center

    w2c = np.linalg.inv(pose_c2w)
    corners_cam = (w2c[:3, :3] @ corners_world.T).T + w2c[:3, 3]

    if not np.any(corners_cam[:, 2] > 0):
        return None

    corners_2d = np.full((8, 2), np.nan)
    for i in range(8):
        if corners_cam[i, 2] > 0:
            pt = K @ corners_cam[i]
            corners_2d[i] = pt[:2] / pt[2]

    return corners_2d


def draw_obb_on_image(image, corners_2d, label="", color=(0, 255, 0), thickness=2):
    """Draw projected OBB wireframe on an image."""
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
    """Save OBB results in the submission JSON format."""
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
    """Compare VGA socket OBB against ground truth sample answer."""
    gt_vga = None
    our_vga = None
    for s in sample_answers:
        if s['entity'] == 'vga_socket':
            gt_vga = s['obb']
    for a in answers:
        if a['entity'] == 'vga_socket':
            our_vga = a['obb']

    if gt_vga is None or our_vga is None:
        print("  [INFO] Cannot validate — VGA socket missing from one of the answer sets")
        return

    c_gt = np.array(gt_vga['center'])
    c_ours = np.array(our_vga['center'])
    dist = np.linalg.norm(c_gt - c_ours)

    e_gt = np.array(gt_vga['extent'])
    e_ours = np.array(our_vga['extent'])

    print(f"\n  --- VGA Socket Validation ---")
    print(f"  Center error: {dist:.4f} m ({dist*1000:.1f} mm)")
    print(f"    GT:   [{c_gt[0]:.6f}, {c_gt[1]:.6f}, {c_gt[2]:.6f}]")
    print(f"    Ours: [{c_ours[0]:.6f}, {c_ours[1]:.6f}, {c_ours[2]:.6f}]")

    print(f"\n  Extents (half-sizes):")
    print(f"    GT:   [{e_gt[0]:.6f}, {e_gt[1]:.6f}, {e_gt[2]:.6f}]")
    print(f"    Ours: [{e_ours[0]:.6f}, {e_ours[1]:.6f}, {e_ours[2]:.6f}]")

    try:
        R_gt = np.array(gt_vga['rotation'])
        R_ours = np.array(our_vga['rotation'])
        R_diff = R_ours @ R_gt.T
        trace = np.clip(np.trace(R_diff), -1.0, 3.0)
        angle = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
        print(f"\n  Rotation error: {angle:.1f} degrees")
    except Exception as e:
        print(f"\n  Rotation comparison failed: {e}")

    vol_gt = np.prod(e_gt) * 8
    vol_ours = np.prod(e_ours) * 8
    print(f"\n  Volume: GT={vol_gt:.2e} m^3, Ours={vol_ours:.2e} m^3")
    if vol_gt > 0:
        print(f"  Volume ratio: {vol_ours/vol_gt:.2f}x")


def plot_validation_summary(answers, sample_answers, save_path=None):
    """
    Create a matplotlib bar chart comparing our extents vs ground truth
    for the VGA socket. Saves to file if path provided.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gt_vga = None
    our_vga = None
    for s in sample_answers:
        if s['entity'] == 'vga_socket':
            gt_vga = s['obb']
    for a in answers:
        if a['entity'] == 'vga_socket':
            our_vga = a['obb']

    if gt_vga is None or our_vga is None:
        return

    e_gt = np.array(gt_vga['extent']) * 1000  # convert to mm
    e_ours = np.array(our_vga['extent']) * 1000

    labels = ['Width', 'Height', 'Depth']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, e_gt, width, label='Ground Truth', color='steelblue')
    ax.bar(x + width/2, e_ours, width, label='Ours', color='coral')

    ax.set_ylabel('Half-extent (mm)')
    ax.set_title('VGA Socket: Extent Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved validation plot: {save_path}")
    plt.close()
