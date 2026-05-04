# Metric-Semantic 3D Reconstruction Pipeline

CP260-2026 Final Project — Metric-Semantic reconstruction of a desktop PC scene from posed multi-view images. The pipeline identifies and localizes power, ethernet, and VGA sockets on the back panel as 3D Oriented Bounding Boxes (OBBs).

## Overview

Given 16 posed images of a desktop PC, the pipeline:
1. Builds a sparse 3D point cloud using SIFT feature matching and triangulation
2. Loads 2D bounding box annotations from a JSON file to identify socket locations
3. Triangulates dense 3D points within each socket's ROI across multiple views
4. Fits Oriented Bounding Boxes using Open3D with physical depth priors
5. Validates results by projecting OBBs back onto images and comparing against ground truth

## Project Structure

```
├── run_pipeline.py          # Main pipeline script
├── intrinsic.json           # Camera intrinsic parameters
├── annotations.json         # 2D bounding box annotations for each socket
├── sample_answers.json      # Sample ground truth for VGA validation
├── requirements.txt         # Python dependencies
├── src/
│   ├── data.py              # Configuration, paths, data loading
│   ├── semantic.py          # Annotation loading and visualization
│   ├── reconstruction.py    # Sparse 3D reconstruction (SIFT + triangulation)
│   ├── pose_estimation.py   # OBB fitting via triangulation + Open3D
│   └── utils.py             # Projection, visualization, validation
├── Data/                    # Dataset (images + poses.json)
├── docs/                    # Project report
└── output/                  # Generated results
```

## Setup

```bash
pip install numpy opencv-python open3d tqdm matplotlib
```

## Usage

Place frame images and poses.json in the `Data/` directory, then run:

```bash
python run_pipeline.py
```

To skip the sparse reconstruction step:

```bash
python run_pipeline.py --skip-reconstruction
```

## Output

Results are saved to `output/`:
- `answers.json` — OBB poses in submission format
- `point_cloud.ply` — sparse 3D reconstruction
- `annotations/` — annotated images and OBB projection visualizations

## Dataset

Download from: https://drive.google.com/file/d/1U8kTzhToFkHihi6Qw0UTSO2M9_JLo3i/view
