"""
Sparse 3D reconstruction from multi-view feature matching and triangulation.
Uses SIFT features across posed views to build a 3D point cloud.
"""
import cv2
import numpy as np
from itertools import combinations
from tqdm import tqdm
from . import config
from .data_loader import build_projection


def extract_features(images, n_features=None, scale=None):
    """
    Extract SIFT keypoints and descriptors from all images.
    Keypoint coordinates are returned in original image resolution.
    """
    if n_features is None:
        n_features = config.MAX_FEATURES
    if scale is None:
        scale = config.DOWNSCALE

    sift = cv2.SIFT_create(nfeatures=n_features)
    features = {}

    for idx, img in tqdm(images.items(), desc="  Extracting features"):
        if scale != 1.0:
            img_small = cv2.resize(img, None, fx=scale, fy=scale)
        else:
            img_small = img

        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        kps, descs = sift.detectAndCompute(gray, None)

        # Scale keypoint coordinates back to original resolution
        if scale != 1.0:
            for kp in kps:
                kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)

        features[idx] = (kps, descs)

    return features


def match_features(features, ratio_thresh=None):
    """
    Match SIFT descriptors between all image pairs using Lowe's ratio test.
    Returns dict mapping (idx1, idx2) to list of good matches.
    """
    if ratio_thresh is None:
        ratio_thresh = config.LOWE_RATIO

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    frame_ids = sorted(features.keys())
    matches = {}

    for idx1, idx2 in tqdm(combinations(frame_ids, 2), desc="  Matching pairs",
                           total=len(frame_ids) * (len(frame_ids) - 1) // 2):
        _, desc1 = features[idx1]
        _, desc2 = features[idx2]

        if desc1 is None or desc2 is None:
            continue
        if len(desc1) < 2 or len(desc2) < 2:
            continue

        raw = matcher.knnMatch(desc1, desc2, k=2)

        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)

        if len(good) >= 10:
            matches[(idx1, idx2)] = good

    print(f"  Found {len(matches)} valid image pairs")
    return matches


def triangulate_matches(features, matches, poses, K):
    """
    Triangulate 3D points from matched features using known poses.
    Filters points by reprojection error and positive depth.
    """
    all_points = []
    all_colors = []

    for (idx1, idx2), match_list in tqdm(matches.items(), desc="  Triangulating"):
        if idx1 not in poses or idx2 not in poses:
            continue

        kps1, _ = features[idx1]
        kps2, _ = features[idx2]

        P1 = build_projection(K, poses[idx1])
        P2 = build_projection(K, poses[idx2])

        pts1 = np.array([kps1[m.queryIdx].pt for m in match_list], dtype=np.float64)
        pts2 = np.array([kps2[m.trainIdx].pt for m in match_list], dtype=np.float64)

        # Triangulate all matches at once
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts4d[:3] / pts4d[3:]).T

        # Filter each point by reprojection error and positive depth
        for i, pt3d in enumerate(pts3d):
            pt_h = np.append(pt3d, 1.0)

            proj1 = P1 @ pt_h
            if proj1[2] <= 0:
                continue
            err1 = np.linalg.norm(proj1[:2] / proj1[2] - pts1[i])

            proj2 = P2 @ pt_h
            if proj2[2] <= 0:
                continue
            err2 = np.linalg.norm(proj2[:2] / proj2[2] - pts2[i])

            if err1 < config.REPROJ_THRESH and err2 < config.REPROJ_THRESH:
                all_points.append(pt3d)
                all_colors.append([200, 200, 200])

    if len(all_points) == 0:
        print("  [WARN] No points triangulated!")
        return np.zeros((0, 3)), np.zeros((0, 3))

    points_3d = np.array(all_points)
    colors = np.array(all_colors, dtype=np.uint8)
    print(f"  Triangulated {len(points_3d)} points")
    return points_3d, colors


def filter_point_cloud(points_3d, colors=None):
    """
    Clean up point cloud with voxel downsampling and statistical outlier removal.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    if colors is not None and len(colors) == len(points_3d):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)

    pcd = pcd.voxel_down_sample(config.VOXEL_SIZE)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=config.OUTLIER_NEIGHBORS,
        std_ratio=config.OUTLIER_STD
    )

    pts_out = np.asarray(pcd.points)
    clr_out = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
    print(f"  Filtered: {len(points_3d)} -> {len(pts_out)} points")
    return pts_out, clr_out


def save_point_cloud(points_3d, colors, path):
    """Save point cloud as PLY file."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)

    o3d.io.write_point_cloud(path, pcd)
    print(f"  Saved point cloud: {path} ({len(points_3d)} points)")


def build_sparse_reconstruction(images, poses, K):
    """
    Full sparse reconstruction: extract features, match, triangulate, filter.
    Returns (points_3d, colors, features, matches).
    """
    print("\n--- Sparse 3D Reconstruction ---")

    features = extract_features(images)
    matches = match_features(features)
    points_3d, colors = triangulate_matches(features, matches, poses, K)

    if len(points_3d) == 0:
        return points_3d, colors, features, matches

    points_3d, colors = filter_point_cloud(points_3d, colors)
    return points_3d, colors, features, matches
