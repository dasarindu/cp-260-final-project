# Metric-Semantic 3D Reconstruction Pipeline

CP260-2026 Final Project — Metric-Semantic reconstruction of a desktop PC scene from posed multi-view images. The pipeline identifies and localizes power, ethernet, and VGA sockets on the back panel as 3D Oriented Bounding Boxes (OBBs).

## Overview

Given 16 posed images of a desktop PC, the pipeline:
1. Builds a sparse 3D point cloud using SIFT feature matching and triangulation
2. Uses manually annotated 2D bounding boxes to identify socket locations
3. Triangulates dense 3D points within each socket's ROI
4. Fits Oriented Bounding Boxes using PCA with physical depth priors
5. Validates results by projecting OBBs back onto images

## Project Structure

├── run_pipeline.py          # Main pipeline script
├── intrinsic.json           # Camera intrinsic parameters
├── sample_answers.json      # Sample ground truth for validation
├── requirements.txt         # Python dependencies
├── src/
│   ├── config.py            # Paths and parameters
│   ├── data_loader.py       # Image, pose, and intrinsic loading
│   ├── reconstruction.py    # Sparse 3D reconstruction
│   ├── semantic.py          # 2D bounding box annotations
│   ├── pose_estimation.py   # OBB fitting via triangulation + PCA
│   └── utils.py             # Projection, visualization, I/O
├── Data/                    # Dataset (images + poses.json)
├── docs/                    # Project report
└── output/                  # Generated results
## Setup

```bash
pip install numpy opencv-python open3d tqdm
```

## Usage

Place frame images and poses.json in the Data/ directory, then run:

```bash
python run_pipeline.py
```

Results are saved to output/ including answers.json and annotated images.

## Dataset

Download from: https://drive.google.com/file/d/1U8kTzhToFkHihi6Qw0UTSO2M9_JLo3i/view
