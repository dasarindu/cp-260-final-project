"""
3D OBB pose estimation via multi-view triangulation.
Uses Open3D's oriented bounding box for fitting.
"""
import os
import cv2
import numpy as np
from itertools import combinations
from . import data
from .semantic import get_annotations, get_entity_names, get_roi_center

# Approximate connector depth in meters
CONNECTOR_DEPTH = 0.006


def triangulate_point_dlt(observations, K, poses):
    """
    Triangulate a single 3D point from 2D observations using DLT.
    Returns (3,) array or None if behind any camera.
    """
    frame_ids = list(observations.keys())
    if len(frame_ids) < 2:
        return None

    A = []
    for idx in frame_ids:
        P = data.build_projection(K, poses[idx])
        u, v = observations[idx]
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])
    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]

    for idx in frame_ids:
        w2c = np.linalg.inv(poses[idx])
        if (w2c[:3, :3] @ X + w2c[:3, 3])[2] <= 0:
            return None

    return X


def triangulate_entity_roi(entity_name, annotations, K, poses, grid_size=400):
    """
    Triangulate dense 3D points for an entity by sampling corresponding
    grid points across its 2D bounding boxes in multiple views.
    """
    entity_ann = annotations.get(entity_name, {})
    frame_ids = sorted(entity_ann.keys())

    if len(frame_ids) < 2:
        print(f"  [WARN] Need >=2 views for {entity_name}")
        return np.zeros((0, 3))

    pts_3d = []
    n_side = int(np.sqrt(grid_size))
    ts = np.linspace(0.05, 0.95, n_side)

    for id1, id2 in combinations(frame_ids, 2):
        if id1 not in poses or id2 not in poses:
            continue

        bb1 = entity_ann[id1]
        bb2 = entity_ann[id2]
        P1 = data.build_projection(K, poses[id1])
        P2 = data.build_projection(K, poses[id2])
        w2c1 = np.linalg.inv(poses[id1])
        w2c2 = np.linalg.inv(poses[id2])

        for ty in ts:
            for tx in ts:
                u1 = bb1[0] + tx * (bb1[2] - bb1[0])
                v1 = bb1[1] + ty * (bb1[3] - bb1[1])
                u2 = bb2[0] + tx * (bb2[2] - bb2[0])
                v2 = bb2[1] + ty * (bb2[3] - bb2[1])

                p1 = np.array([[u1, v1]], dtype=np.float64)
                p2 = np.array([[u2, v2]], dtype=np.float64)

                pts4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
                pt = (pts4d[:3] / pts4d[3:]).flatten()

                z1 = (w2c1[:3, :3] @ pt + w2c1[:3, 3])[2]
                z2 = (w2c2[:3, :3] @ pt + w2c2[:3, 3])[2]
                if z1 <= 0 or z2 <= 0:
                    continue

                r1 = P1 @ np.append(pt, 1)
                r2 = P2 @ np.append(pt, 1)
                e1 = np.linalg.norm(r1[:2] / r1[2] - [u1, v1])
                e2 = np.linalg.norm(r2[:2] / r2[2] - [u2, v2])

                if e1 < 50 and e2 < 50:
                    pts_3d.append(pt)

    if len(pts_3d) == 0:
        print(f"  [WARN] No 3D points for {entity_name}")
        return np.zeros((0, 3))

    points = np.array(pts_3d)
    points = _filter_outliers_mad(points)
    print(f"  {entity_name}: {len(points)} 3D points")
    return points


def _filter_outliers_mad(points, threshold=3.0):
    """Remove outliers using Median Absolute Deviation."""
    if len(points) < 5:
        return points

    median = np.median(points, axis=0)
    dists = np.linalg.norm(points - median, axis=1)
    med_dist = np.median(dists)
    mad = 1.4826 * med_dist

    if mad < 1e-10:
        return points

    return points[dists < threshold * mad]


def _orient_rotation_to_convention(R):
    """
    Re-orient a rotation matrix to match the sample answer convention:
        Row 0 = Y-dominant axis (connector width)
        Row 1 = Z-dominant axis (connector height)
        Row 2 = X-dominant axis (panel outward normal)
    """
    rows = list(R)

    # Assign each row to the world axis it's most aligned with
    assignment = {}
    used = set()
    for axis in [0, 1, 2]:
        best_row = -1
        best_score = -1
        for i, row in enumerate(rows):
            if i in used:
                continue
            score = abs(row[axis])
            if score > best_score:
                best_score = score
                best_row = i
        assignment[axis] = rows[best_row].copy()
        used.add(best_row)

    # Row 2 = X-dominant, positive X
    row2 = assignment[0]
    if row2[0] < 0:
        row2 = -row2

    # Row 0 = Y-dominant, positive Y
    row0 = assignment[1]
    if row0[1] < 0:
        row0 = -row0

    # Row 1 = cross product for right-handedness
    row1 = np.cross(row2, row0)
    row1 /= (np.linalg.norm(row1) + 1e-12)

    # Re-orthogonalize row0
    row0 = np.cross(row1, row2)
    row0 /= (np.linalg.norm(row0) + 1e-12)

    R_out = np.array([row0, row1, row2])
    if np.linalg.det(R_out) < 0:
        R_out[1] = -R_out[1]

    return R_out


def fit_obb(points_3d, entity_name):
    """
    Fit an Oriented Bounding Box using Open3D, then adjust rotation
    to match the sample answer convention and apply depth prior.
    """
    import open3d as o3d

    if len(points_3d) < 3:
        print("  [WARN] Too few points for OBB")
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    obb = pcd.get_oriented_bounding_box()

    center = np.array(obb.center)
    R_raw = np.array(obb.R)
    extents_full = np.array(obb.extent)  # full extents from Open3D

    # Open3D gives full extents, we need half-extents
    half_extents = extents_full / 2.0

    # Sort half-extents descending and reorder rotation columns to match
    order = np.argsort(half_extents)[::-1]
    half_extents = half_extents[order]
    R_sorted = R_raw[:, order]  # reorder columns

    # Convert to row-based rotation matrix
    R_rows = R_sorted.T

    # Re-orient to match GT convention
    R_final = _orient_rotation_to_convention(R_rows)

    # Override the smallest extent with depth prior
    extent_out = np.zeros(3)
    extent_out[0] = half_extents[0]      # width (largest)
    extent_out[1] = half_extents[1]      # height (medium)
    extent_out[2] = CONNECTOR_DEPTH      # depth prior

    if extent_out[1] < CONNECTOR_DEPTH / 2.0:
        extent_out[1] = CONNECTOR_DEPTH

    return {
        "center": center.tolist(),
        "extent": extent_out.tolist(),
        "rotation": R_final.tolist()
    }


def estimate_all_poses(annotations, K, poses, n_grid_points=400):
    """Estimate 3D OBB for all annotated entities."""
    print("\n--- OBB Pose Estimation ---")
    results = []

    for name in get_entity_names():
        print(f"\n  Processing: {name}")

        points_3d = triangulate_entity_roi(
            name, annotations, K, poses, grid_size=n_grid_points
        )

        if len(points_3d) < 3:
            print(f"  [SKIP] Not enough points for {name}")
            continue

        obb = fit_obb(points_3d, name)
        if obb is None:
            continue

        results.append({"entity": name, "obb": obb})

        c = obb['center']
        e = obb['extent']
        print(f"    Center: [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")
        print(f"    Extent: [{e[0]:.6f}, {e[1]:.6f}, {e[2]:.6f}]")

    return results


def validate_with_projection(results, images, K, poses):
    """Project OBBs onto images and save for visual verification."""
    from .utils import project_obb_to_image, draw_obb_on_image
    from .semantic import ENTITY_COLORS

    viz_frames = [471, 496, 515]

    for idx in viz_frames:
        if idx not in images or idx not in poses:
            continue

        vis = images[idx].copy()
        for res in results:
            name = res['entity']
            color = ENTITY_COLORS.get(name, (0, 255, 255))
            corners = project_obb_to_image(res['obb'], K, poses[idx])
            vis = draw_obb_on_image(vis, corners, label=name, color=color)

        out_path = os.path.join(data.OUTPUT_DIR, "annotations",
                                f"obb_projection_{idx:06d}.png")
        cv2.imwrite(out_path, vis)
        print(f"  Saved: {out_path}")
