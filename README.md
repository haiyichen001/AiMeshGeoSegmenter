# AiMeshGeoSegmenter

End-to-end AI pipeline: STL file in, 8-class surface type labels out.

## Architecture

```
STL (triangles)
  -> [Stage 1] Edge Classifier (MLP) — merge or cut adjacent triangles
  -> Surface patches
  -> [Stage 2] Face Classifier (GNN) — classify each patch into 1 of 8 types
  -> Labeled STL
```

| Stage | Model | Params | Size | CV Accuracy |
|-------|-------|--------|------|-------------|
| Edge MLP | SGDClassifier (sklearn) | — | 172 MB data | 93.08% +/- 0.07 |
| Face GNN | GraphSAGE | 80,072 | 325 KB | 94.50% +/- 0.53 |

Both models trained and evaluated on STL-domain data. Zero domain shift.

## 8 Output Classes

| # | Class | Fitting Strategy |
|---|-------|-----------------|
| 1 | Plane | Point-normal equation |
| 2 | Cylinder | Axis + radius |
| 3 | Sphere | Center + radius |
| 4 | Cone | Apex + axis + angle |
| 5 | Torus | Major + minor radii |
| 6 | Fillet | Rolling-ball or analytic |
| 7 | Chamfer | Plane/Cone subset |
| 8 | Freeform | B-spline |

## Label Generation: Detection Methods

### 6 Base Types — STEP Direct Read

| Class | Detection |
|-------|-----------|
| Plane | `BRepAdaptor_Surface.GetType() == GeomAbs_Plane` |
| Cylinder | `BRepAdaptor_Surface.GetType() == GeomAbs_Cylinder` |
| Sphere | `BRepAdaptor_Surface.GetType() == GeomAbs_Sphere` |
| Cone | `BRepAdaptor_Surface.GetType() == GeomAbs_Cone` |
| Torus | `BRepAdaptor_Surface.GetType() == GeomAbs_Torus` |
| Freeform | BSpline, Bezier, Revolution, Extrusion, Other |

### 2 Refined Types — Weighted Scoring

**Fillet** (applied to Cylinder/Torus faces):

| Signal | Weight | Criterion |
|--------|--------|-----------|
| Radius (cylinder radius or torus minor radius) | 0.50 | < 5% of part bounding-box diagonal |
| Neighbor count | 0.25 | exactly 2 |
| Area ratio | 0.25 | area / (area + neighbor_area) < 15% |

Score > 0.5 -> fillet. Rationale: a structural cylinder has a large radius proportional to the part; a fillet cylinder has a tiny radius. No edge-continuity check needed.

**Chamfer — Cone faces:**

| Signal | Weight | Criterion |
|--------|--------|-----------|
| Axial half-length | 0.50 | < 5% of part diagonal |
| Neighbor count | 0.25 | exactly 2 |
| Area ratio | 0.25 | < 15% |

**Chamfer — Plane faces:**

| Signal | Weight | Criterion |
|--------|--------|-----------|
| Dihedral angle to neighbors | 0.50 | > 15 degrees |
| Neighbor count | 0.25 | exactly 2 |
| Area ratio | 0.25 | < 5% (stricter than cone) |

**Sphere recovery** (applied to Freeform faces that are geometrically spherical):

Least-squares sphere fit on face vertices. If RMS error < 2% of radius => re-label as sphere.

## Training vs Inference — Data Flow

Training and inference both operate on **STL tessellation**. No domain mismatch.

```
TRAINING                                  INFERENCE
=======                                   =========
STEP B-Rep -> face type labels            STL file
STL file                                    |
  |                                         | infer.py MLP stage
  | KD-tree map: STL triangle -> face       v
  v                                         Patches (from MLP grouping)
Group STL triangles by face                  |
  |                                         | infer.py GNN stage
  | Extract 18-dim features                 v
  v                                         Labeled patches
Graph NPZ (nodes=faces, edges=adj)
  |
  | train.py
  v
GNN model <------------------------------ GNN model
```

Both sides see STL triangles. Zero domain shift for both MLP and GNN stages.

## Training Pipeline

```
STEP files
  -> step_to_stl.py          multi-density STL generation (random deflection 0.05%-2% of diagonal)
  -> label_faces.py          STEP B-Rep -> face type labels (JSON)
  -> refine_labels.py        fillet / chamfer / sphere detection via weighted scoring
  -> extract_features.py     JSON labels -> 18-dim graph features (NPZ)
  -> train_mlp.py            MLP edge classifier from STL adjacency, 5-fold CV
  -> train.py                GraphSAGE face classifier from graph NPZ, 5-fold CV + rotation aug
```

## Inference

```
STL file
  -> infer.py
       Stage 1: MLP groups triangles into patches
       Stage 2: GNN labels each patch
       Output: 8-class labels
```

## Project Structure

```
AiMeshGeoSegmenter/
├── data/
│   ├── step/           ~21K STEP files
│   ├── stl/            ~21K multi-density STL files
│   ├── labels/         per-face type JSON
│   ├── graphs/         training-ready NPZ graph files
│   └── answers/        evaluation ground truth
├── scripts/
│   ├── label_faces.py         STEP -> labels
│   ├── refine_labels.py       weighted scoring
│   ├── extract_features.py    labels -> graphs
│   ├── train.py               GNN training
│   ├── train_mlp.py           MLP training
│   ├── step_to_stl.py         STEP -> STL
│   ├── infer.py               inference pipeline
│   └── eval_pipeline.py       evaluation
├── models/              model weights + training plots
└── viewer/              localhost:8006 (Compare / Labels / Infer)
```

## Requirements

- Python 3.11+
- PyTorch + PyTorch Geometric
- pythonocc-core
- numpy, scipy, trimesh, flask

## Related Work

| Paper | Venue | Link |
|-------|-------|------|
| SPFN | CVPR 2019 | [arXiv](https://arxiv.org/abs/1811.08988) |
| CPFN | ICCV 2021 | [code](https://github.com/erictuanle/CPFN) |
| FilletRec (ZJU) | 2025 | [arXiv](https://arxiv.org/abs/2511.05561) |
| MeshCNN | SIGGRAPH 2019 | [code](https://github.com/ranahanocka/MeshCNN) |
| NVDNet | SIGGRAPH 2024 | [arXiv](https://arxiv.org/abs/2406.05261) |

## License

MIT
