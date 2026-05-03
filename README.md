# AiMeshGeoSegmenter

AI-powered semantic segmentation framework for detecting analytic surfaces from 3D mesh data. Given a tessellated STL mesh, the model classifies each face/patch into geometric primitive categories (plane, cylinder, sphere, cone, fillet, etc.), enabling direct analytic surface fitting without NURBS.

## Motivation

Traditional reverse engineering pipelines rely on B-spline / NURBS surface fitting, which involves complex parameter tuning (knot vectors, control points, degree selection). This project takes a different route:

1. **AI classification** — a learned model predicts what type of analytic surface each mesh region represents
2. **Analytic fitting** — based on the predicted class, fit the corresponding closed-form equation (e.g., least-squares cylinder, RANSAC plane)

The result: a lightweight, explainable, and editable CAD representation of the scanned or triangulated geometry.

## Pipeline (Planned)

```
STEP/IGES  -->  STL Mesh  -->  Face-level Segmentation  -->  Per-primitive Analytic Fit
                                    (plane / cylinder       (point-normal, axis+radius,
                                     sphere / cone /         sphere center+radius,
                                     fillet / ...)           cone apex+angle, ...)
```

## Supported Primitive Types

| Type | Parameters | Fitting Method |
|------|-----------|----------------|
| Plane | `(n, d)` — normal + offset | Least-squares / RANSAC |
| Cylinder | `(axis, radius)` — axis direction + point + radius | Non-linear least-squares |
| Sphere | `(center, radius)` | Linear least-squares |
| Cone | `(apex, axis, angle)` | Non-linear least-squares |
| Fillet / Blend | TBD | TBD |

## Project Structure (Planned)

```
AiMeshGeoSegmenter/
├── data/             # Dataset pipeline (STEP->STL, labeling)
├── models/           # Segmentation network definitions
├── fitting/          # Analytic surface fitting backends
├── eval/             # Evaluation metrics and visualizers
├── scripts/          # Training, inference, and data prep scripts
└── configs/          # YAML config files for experiments
```

## Requirements

- Python 3.11+
- PyTorch + PyTorch Geometric (or equivalent)
- pythonocc-core (for STEP reading)
- numpy, scipy, open3d

## Quick Start

```bash
git clone git@github.com:haiyichen001/AiMeshGeoSegmenter.git
cd AiMeshGeoSegmenter
conda create -n aimesh python=3.11 -y
conda activate aimesh
pip install -r requirements.txt
```

## License

MIT
