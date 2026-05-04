# AiMeshGeoSegmenter

Single-stage GAT model: STL triangles in, 8-class surface labels out.

## Motivation

Previous two-stage pipeline (MLP edge classifier + GNN face classifier) had an inherent flaw: the MLP was trained on B-Rep face groupings, but at inference time it segments STL triangles into different patch boundaries. This domain mismatch caused feature distribution shift in the GNN, degrading end-to-end accuracy despite both models scoring >94% individually.

## New Architecture

**GAT (Graph Attention Network) directly on STL triangles.**

```
STL file (triangles)
  |
  | Each triangle = one graph node (features: normal, area, center)
  | Triangle adjacency = graph edges (shared edge in the mesh)
  |
  v
GAT (3 layers, ~200K params)
  |
  | Attention-weighted message passing on triangle adjacency graph
  |
  v
Per-triangle 8-class label (plane/cylinder/sphere/cone/torus/fillet/chamfer/freeform)
```

Why GAT:
- Works directly on triangle mesh — no point cloud conversion needed
- Attention mechanism learns which neighbors belong to the same surface
- Single model, no domain mismatch between training and inference
- ~200K params, GPU training, CPU inference capable

## 8 Output Classes

| # | Class | Detection |
|---|-------|-----------|
| 1 | Plane | GeomAbs_Plane, not chamfer |
| 2 | Cylinder | GeomAbs_Cylinder, not fillet |
| 3 | Sphere | GeomAbs_Sphere + BSpline least-squares recovery (error < 2%) |
| 4 | Cone | GeomAbs_Cone, not chamfer |
| 5 | Torus | GeomAbs_Torus, not fillet |
| 6 | Fillet | Cylinder: 2 neighbors + radius < 10% diagonal + convex only |
|   |        | Torus: 2 neighbors + minor radius < 10% diagonal + convex only |
| 7 | Chamfer | Cone: 2 neighbors + half-length < 10% + angle 85-95 deg + convex only |
|   |         | Plane: 2 neighbors + half-length < 10% + angle 85-95 deg + convex only |
|   |         | Angle: uses axis direction for curved faces (Cylinder/Torus/Cone) instead of vertex normals |
| 8 | Freeform | Everything else (BSpline, Bezier, Revolution, Extrusion) |

Convexity is detected via `BRepAdaptor_Surface.D1()` exact surface normal at face midpoint.
All classification (type mapping + fillet/chamfer/sphere refinement) is done in a single pass by `label_faces.py`.

## Training Data

```
STEP -> label_faces.py -> per-face labels
STL file -> per-triangle mesh
      |
KD-tree: map each STL triangle to nearest B-Rep face
      |
      v
Per-triangle training data: (normal, area, center, face_label)
      |
      v
GAT training on triangle adjacency graph
```

## Project Structure

```
AiMeshGeoSegmenter/
├── data/
│   ├── step/       20,112 STEP source files
│   ├── stl/        20,112 STL files
│   └── labels/     20,112 per-face label JSONs
├── scripts/
│   ├── label_faces.py       STEP -> face type labels (includes refinement)
│   ├── step_to_stl.py       STEP -> multi-density STL
│   ├── build_data.py        STL + labels -> training data
│   ├── train.py             GAT training
│   └── infer.py             STL -> 8-class labels (inference)
├── models/                  model weights + training logs
└── viewer/                  localhost:8006
```

## Status

- Dataset: 20,112 STEP/STL pairs, 197K annotated faces
- Training: in progress
- GPU: NVIDIA RTX 5060 Ti, 17 GB VRAM

## License

MIT
