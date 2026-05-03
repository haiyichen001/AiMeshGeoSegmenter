# AiMeshGeoSegmenter

Surface type classification for STL-to-STEP reverse engineering. Given a tessellated mesh, the model labels each face/patch into one of 8 geometric categories, enabling downstream analytic or B-spline fitting.

## The Big Picture

```
STL Mesh (triangles)
  -> [1] Edge Classifier (MLP)         merge or cut adjacent triangles?     <-- AI
    -> [2] Mesh Segmentation           group connected triangles into patches
      -> [3] Face Classifier (GNN)     plane / cylinder / sphere / ...?     <-- AI
        -> [4] Surface Fitting         analytic or B-spline
          -> [5] Topology Reconstruction + STEP Export
```

This project covers **steps 1-3 only**.

## Status

- **Samples**: 22,372 STEP/STL pairs
- **Models**:
  - Edge Classifier: MLP binary (merge/cut adjacent triangles), ~5K params
  - Face Classifier: GraphSAGE 8-class, 80K params, 325KB, 97.2% accuracy
- **Viewer**: http://localhost:8006 (Compare / Labels / Infer)
- **Infer tab**: upload STL → full AI pipeline → labeled 3D view

## Supported Classes

| # | Class | Fitting Strategy | Detection |
|---|-------|-----------------|-----------|
| 1 | Plane | Point-normal equation | OCCT GeomAbs_Plane |
| 2 | Cylinder | Axis + radius | OCCT GeomAbs_Cylinder |
| 3 | Sphere | Center + radius | OCCT GeomAbs_Sphere |
| 4 | Cone | Apex + axis + angle | OCCT GeomAbs_Cone (structural) |
| 5 | Torus | Major/minor radii + axis | OCCT GeomAbs_Torus |
| 6 | Fillet | Analytic (cylinder/torus subset) | Cylinder/Torus with 2+ neighbors, small area ratio, tangent edges |
| 7 | Chamfer | Analytic (plane/cone subset) | Small Cone with 2+ neighbors, area < 500mm² |
| 8 | Freeform | B-spline | BSpline, Bezier, Revolution, Extrusion |

## Project Structure

```
AiMeshGeoSegmenter/
├── data/
│   ├── step/         (22,372 STEP)
│   ├── stl/          (22,372 STL)
│   ├── labels/       (per-face type JSON, viewer source)
│   └── graphs/       (training-ready NPZ graph files)
├── scripts/
│   ├── label_faces.py       STEP -> label JSON + stats
│   ├── refine_labels.py     Fillet/chamfer post-processing
│   ├── extract_features.py  Label JSON -> graph NPZ (GNN input)
│   ├── train.py             GNN training + eval
│   ├── step_to_stl.py       STEP -> STL batch conversion
│   ├── label_abc.py         STEP -> graph NPZ (direct, skips JSON)
│   ├── filter_abc.py        Dataset filtering by complexity
│   └── prune_abc.py         Dataset pruning
├── models/
│   └── face_classifier.pt   (326 KB, 80K params)
└── viewer/ -> http://localhost:8006
    ├── web_server.py         Flask: STEP / STL / Labels API
    ├── multi_viewer.html     Compare tab (STEP + STL side-by-side)
    └── labels_viewer.html    Labels tab (8-class color-coded faces)
```

## Requirements

- Python 3.11+
- PyTorch + PyTorch Geometric
- pythonocc-core
- numpy, scipy, trimesh, flask

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
| SPFN — Supervised Primitive Fitting | CVPR 2019 | [arXiv](https://arxiv.org/abs/1811.08988), [code](https://github.com/lingxiaoli94/SPFN) |
| CPFN — Cascaded Primitive Fitting | ICCV 2021 | [code](https://github.com/erictuanle/CPFN) |
| PrimitiveNet — Primitive Instance Segmentation | ICCV 2021 | [code](https://github.com/hjwdzh/PrimitiveNet) |
| NVDNet — Split-and-Fit B-Rep Reconstruction | SIGGRAPH 2024 | [arXiv](https://arxiv.org/abs/2406.05261) |
| STEP-Parts — B-Rep Partition for CAD Learning | 2026-04 | arXiv 2604.14927 |
| MeshCNN — CNN for 3D Meshes | SIGGRAPH 2019 | [code](https://github.com/ranahanocka/MeshCNN) |

## License

MIT
