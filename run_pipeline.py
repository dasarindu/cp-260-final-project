#!/usr/bin/env python3
"""
Metric-Semantic 3D Reconstruction Pipeline
CP260-2026 Final Project
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import data
from src.data import load_dataset, load_sample_answers
from src.reconstruction import build_sparse_reconstruction, save_point_cloud
from src.semantic import get_annotations, visualize_annotations
from src.pose_estimation import estimate_all_poses, validate_with_projection
from src.utils import save_answers_json, validate_against_sample, plot_validation_summary


def main():
    parser = argparse.ArgumentParser(description="3D Reconstruction Pipeline")
    parser.add_argument("--skip-reconstruction", action="store_true",
                        help="Skip sparse reconstruction step")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing answers.json")
    parser.add_argument("--grid-points", type=int, default=400,
                        help="Grid density for ROI triangulation")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 50)
    print("  3D Reconstruction Pipeline")
    print("=" * 50)

    # Step 1: Load data
    print("\n[Step 1] Loading dataset...")
    images, poses, K = load_dataset()

    # Step 2: Sparse reconstruction
    if not args.skip_reconstruction and not args.validate_only:
        print("\n[Step 2] Building sparse reconstruction...")
        points_3d, colors, features, matches = \
            build_sparse_reconstruction(images, poses, K)

        if len(points_3d) > 0:
            ply_path = os.path.join(data.OUTPUT_DIR, "point_cloud.ply")
            save_point_cloud(points_3d, colors, ply_path)
    else:
        print("\n[Step 2] Skipping reconstruction")

    # Step 3: Semantic annotations
    print("\n[Step 3] Loading annotations...")
    annotations = get_annotations()
    print(f"  Entities: {list(annotations.keys())}")
    for name, ann in annotations.items():
        print(f"    {name}: frames {list(ann.keys())}")
    visualize_annotations(images)

    # Step 4: OBB pose estimation
    if not args.validate_only:
        print("\n[Step 4] Estimating 3D poses...")
        results = estimate_all_poses(
            annotations, K, poses, n_grid_points=args.grid_points
        )

        # Step 5: Save results
        print("\n[Step 5] Saving results...")
        answers_path = os.path.join(data.OUTPUT_DIR, "answers.json")
        save_answers_json(results, answers_path)

        # Step 6: Validate
        print("\n[Step 6] Validating...")
        validate_with_projection(results, images, K, poses)

        sample = load_sample_answers()
        validate_against_sample(results, sample)

        # Save validation plot
        plot_path = os.path.join(data.OUTPUT_DIR, "annotations", "validation_plot.png")
        plot_validation_summary(results, sample, save_path=plot_path)
    else:
        answers_path = os.path.join(data.OUTPUT_DIR, "answers.json")
        if os.path.exists(answers_path):
            with open(answers_path, 'r') as f:
                results = json.load(f)
            sample = load_sample_answers()
            validate_against_sample(results, sample)
            validate_with_projection(results, images, K, poses)
        else:
            print(f"  [ERROR] No answers.json found at {answers_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Output: {data.OUTPUT_DIR}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
