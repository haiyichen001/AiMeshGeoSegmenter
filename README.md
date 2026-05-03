# AiMeshGeoSegmenter

Mesh face-level semantic segmentation for analytic surface type recognition. Part of the **STL-to-STEP** reverse engineering pipeline.

## The Big Picture: STL -> STEP

```
STL Mesh
  -> [1] Mesh Segmentation        (which faces belong to the same surface?)
    -> [2] AI Type Recognition    (plane / cylinder / sphere / cone?)  <-- THIS PROJECT
      -> [3] Analytic Fitting     (least-squares per primitive)
        -> [4] Surface Trim + Topology Reconstruction (intersections, B-Rep)
          -> [5] STEP Export
```

This project covers **steps 1 and 2 only**: segment a triangulated mesh into surface patches and classify each patch by its analytic primitive type. The downstream fitting, topology reconstruction, and STEP export are handled by the companion CAD engine.

## Design Goals

- **Lightweight** — inference on consumer CPU, model under 100 MB
- **Supervision from STEP** — use existing STEP files as ground truth: read B-Rep faces, tessellate, transfer surface type labels
- **Four primitives first** — plane, cylinder, sphere, cone. Fillet/blend left for the CAD engine to auto-reconstruct via rolling ball.

## Pipeline

```
STEP (B-Rep)                     STL Mesh
     |                               |
  pythonocc read              [1] Region growing
  face -> surface type         + curvature-based
       |                         clustering
       v                               |
  per-face labels  ---------->  [2] GNN / MeshCNN classifier
       |                               |
       +-- per-patch type ------------+
```

## Supported Types

| Type | STEP Identifier |
|------|-----------------|
| Plane | `PLANE` |
| Cylinder | `CYLINDRICAL_SURFACE` |
| Sphere | `SPHERICAL_SURFACE` |
| Cone | `CONICAL_SURFACE` |

Fillet/blend is intentionally excluded — it is reconstructed automatically by the CAD engine via intersection-rolling-ball after basic surfaces are fitted.

## Project Structure

```
AiMeshGeoSegmenter/
├── data/             # STEP -> labeled mesh dataset pipeline
├── models/           # GNN / MeshCNN classifiers
├── scripts/          # Training, inference, evaluation
└── configs/          # YAML experiment configs
```

## Requirements

- Python 3.11+
- PyTorch + PyTorch Geometric
- pythonocc-core (STEP reading)
- numpy, scipy, open3d

## Quick Start

```bash
git clone git@github.com:haiyichen001/AiMeshGeoSegmenter.git
cd AiMeshGeoSegmenter
conda create -n aimesh python=3.11 -y
conda activate aimesh
pip install -r requirements.txt
```

## Related Work

| Paper | Venue | Link |
|-------|-------|------|
| SPFN — Supervised Primitive Fitting | CVPR 2019 Oral | [arXiv 1811.08988](https://arxiv.org/abs/1811.08988), [code](https://github.com/lingxiaoli94/SPFN) |
| CPFN — Cascaded Primitive Fitting | ICCV 2021 | [code](https://github.com/erictuanle/CPFN) |
| PrimitiveNet — Primitive Instance Segmentation | ICCV 2021 | [code](https://github.com/hjwdzh/PrimitiveNet) |
| NVDNet — Split-and-Fit B-Rep Reconstruction | SIGGRAPH 2024 | [arXiv 2406.05261](https://arxiv.org/abs/2406.05261), [code](https://github.com/yilinliu77/NVDNet) |
| STEP-Parts — B-Rep Partition for CAD Learning | 2026-04 | arXiv 2604.14927 |
| MeshCNN — CNN for 3D Meshes | SIGGRAPH 2019 | [code](https://github.com/ranahanocka/MeshCNN) |

## License

MIT
