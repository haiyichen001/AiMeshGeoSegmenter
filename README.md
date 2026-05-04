# AiMeshGeoSegmenter

Two-stage AI pipeline: STL in, 8-class surface type labels out.

## Architecture

```
STL (triangles)
  -> [Stage 1] Edge Classifier (MLP, 52KB) — merge or cut adjacent triangles
  -> Surface patches
  -> [Stage 2] Face Classifier (GNN, 334KB) — classify each patch into 1 of 8 types
  -> Labeled patches
```

## Results (5-Fold CV, NVIDIA RTX 5060 Ti, TF32)

| Model | Accuracy | Size | Train Time |
|-------|----------|------|------------|
| MLP (Edge) | 94.19% +/- 0.03 | 52 KB | 27s |
| GNN (Face) | 91.32% +/- 0.35 | 334 KB | ~5 min |

## 8 Output Classes

| # | Class | Detection |
|---|-------|-----------|
| 1 | Plane | STEP GeomAbs_Plane |
| 2 | Cylinder | STEP GeomAbs_Cylinder |
| 3 | Sphere | STEP GeomAbs_Sphere + least-squares recovery from BSpline |
| 4 | Cone | STEP GeomAbs_Cone |
| 5 | Torus | STEP GeomAbs_Torus |
| 6 | Fillet | Radius < 5% diagonal + 2 neighbors + area < 15% |
| 7 | Chamfer | Cone: half-length + 2 neighbors + area. Plane: 15-75deg angle + 2 neighbors + area < 5% |
| 8 | Freeform | Everything else |

## Dataset

- **20,112** STEP/STL pairs (>= 3 faces)
- **197,218** annotated faces
- Source: ISO standard parts + ABC single-solid samples
- STL: multi-density (random deflection 0.05%-2% of diagonal)

## Data Flow (zero domain shift)

Both training and inference operate on STL triangles. No tessellation mismatch.

```
TRAINING                                  INFERENCE
STEP -> face labels                       STL file
STL file                                    |
  |                                         | MLP stage
  | KD-tree: STL tri -> face                v
  v                                         Patches
Group STL tris by face                      |
  |                                         | GNN stage
  | 26-dim features (from STL tris)         v
  v                                         Labels
Graph NPZ (nodes=faces, edges=adj)
  |
  | train.py (GPU, TF32)
  v
GNN model
```

## Project Structure

```
AiMeshGeoSegmenter/
├── data/
│   ├── step/             20,112 STEP
│   ├── stl/              20,112 STL
│   ├── labels/           per-face JSON
│   ├── graphs/           GNN training NPZ
│   └── mlp_edges/        MLP training NPZ
├── scripts/
│   ├── label_faces.py         STEP -> labels
│   ├── refine_labels.py       fillet/chamfer/sphere
│   ├── extract_features_stl.py labels+STL -> 26-dim graphs
│   ├── train.py               GNN training
│   ├── train_mlp.py           MLP training
│   ├── build_mlp_data.py      STL+labels -> edge data
│   ├── step_to_stl.py         STEP -> multi-density STL
│   ├── infer.py               full inference pipeline
│   └── eval_pipeline.py       evaluation
├── models/               weights + training logs
└── viewer/               localhost:8006 (Compare/Labels/Infer/Model)
```

## License

MIT
