# AiMeshGeoSegmenter

End-to-end AI pipeline for classifying surface types on tessellated CAD meshes.
Given an STL file, it segments the mesh into surface patches and labels each patch
as one of 8 analytic primitives, enabling downstream STL-to-STEP reverse engineering.

## Architecture: Two-Stage AI Pipeline

```
STL file (triangles)
  |
  |-- [Stage 1] Edge Classifier (MLP)
  |      Input:  per-pair features (dihedral angle, area ratio, edge-length ratio)
  |      Output: merge=1 or cut=0 for every adjacent triangle pair
  |      Model:  4,289 params, 13 KB
  |      Accuracy: 92.88% +/- 0.38 (5-fold CV)
  |      Training data: STL adjacency pairs (same domain as inference)
  |
  |-- Connected-component grouping -> surface patches
  |
  |-- [Stage 2] Face Classifier (GNN / GraphSAGE)
         Input:  per-patch 18-dim geometric features + adjacency graph
         Output: 8-class label per patch
         Model:  80,072 params, 325 KB
```

## 8 Output Classes

| # | Class | Geometric Signature | Fitting Method |
|---|-------|-------------------|----------------|
| 1 | Plane | Zero curvature in all directions | Point-normal equation |
| 2 | Cylinder | Zero curvature along axis, constant curvature radially | Axis + radius |
| 3 | Sphere | Equal non-zero curvature in all directions | Center + radius |
| 4 | Cone | Varying curvature along generator | Apex + axis + angle |
| 5 | Torus | Two distinct principal curvatures | Major + minor radii |
| 6 | Fillet | Small cylinder/torus bridging two faces at an angle | Rolling-ball or analytic |
| 7 | Chamfer | Small cone bridging two faces at an angle | Plane subset or analytic |
| 8 | Freeform | Arbitrary curvature (BSpline, Bezier, etc.) | B-spline |

## Dataset

- **Total samples**: ~21,000 STEP/STL pairs (>= 3 B-Rep faces each)
- **Total annotated faces**: ~197,000
- **Source**: ISO standard mechanical parts + randomly sampled simple single-solid parts from the ABC dataset
- **STL generation**: random deflection per part (0.05% to 2% of bounding box diagonal) for multi-density training
- **Labeling**: per-face surface types extracted from STEP B-Rep via pythonocc.
  Fillet/chamfer labels refined via geometric heuristics (adjacency + area ratio + tangent-edge check).

## Training Pipeline

```
STEP files
  -> label_faces.py         read B-Rep face types, tessellate, store per-face JSON
  -> refine_labels.py       detect fillet / chamfer from adjacency + geometry
  -> extract_features.py    compute 18-dim per-face features, build adjacency graph, save NPZ
  -> train.py               train GraphSAGE with 5-fold cross-validation + rotation augmentation

STL files (multi-density, from step_to_stl.py)
  -> train_mlp.py           train MLP edge classifier from STL adjacency pairs
                              with 5-fold CV, training plots saved to models/
```

Inference (from STL only):
```
STL file
  -> infer.py               Stage 1: MLP segments triangles into patches
                             Stage 2: GNN classifies each patch
                             Output: 8-class labels
```

## Project Structure

```
AiMeshGeoSegmenter/
|-- data/
|   |-- step/                ~21K STEP source files
|   |-- stl/                 ~21K multi-density STL files
|   |-- labels/              per-face type JSON (source for viewer)
|   |-- graphs/              training-ready NPZ graph files
|   |-- answers/             evaluation ground truth
|-- scripts/
|   |-- label_faces.py       STEP B-Rep -> face type labels (JSON)
|   |-- refine_labels.py     fillet / chamfer post-processing
|   |-- extract_features.py  JSON labels -> graph features (NPZ)
|   |-- train.py             GraphSAGE training with k-fold CV
|   |-- train_mlp.py         MLP edge classifier training with k-fold CV
|   |-- step_to_stl.py       STEP -> multi-density STL batch conversion
|   |-- infer.py             full STL inference pipeline
|   |-- eval_pipeline.py     end-to-end evaluation
|-- models/                  all model weights + training plots
|   |-- edge_classifier.pt   Stage 1 MLP, 4.3K params, 13 KB
|   |-- face_classifier.pt   Stage 2 GraphSAGE, 80K params, 325 KB
|   |-- mlp_fold1~5.png      per-fold training curves
|   |-- mlp_kfold_summary.png k-fold accuracy summary
|-- viewer/                  localhost:8006
    |-- web_server.py        Flask server (STEP / STL / Labels / Infer API)
    |-- multi_viewer.html    Compare tab (STEP + STL side-by-side)
    |-- labels_viewer.html   Labels tab (8-class color-coded faces)
    |-- infer.html           Infer tab (STL upload -> predict -> visualize)
```

## Results

### Edge Classifier (MLP) - 5-Fold CV

| Fold | Accuracy |
|------|----------|
| 1 | 92.67% |
| 2 | 92.27% |
| 3 | 93.37% |
| 4 | 92.98% |
| 5 | 93.10% |
| **Mean +/- Std** | **92.88% +/- 0.38** |

Training data: STL triangle adjacency pairs. Model: 4,289 params, 13 KB.

### Face Classifier (GNN) - 5-Fold CV

| Fold | Accuracy |
|------|----------|
| 1 | 97.22% |
| 2 | 97.35% |
| 3 | 97.29% |
| 4 | 97.37% |
| 5 | 97.23% |
| **Mean +/- Std** | **97.29% +/- 0.06** |

Training data: 18-dim per-face geometric features from label JSONs. Model: 80,072 params, 325 KB.

## Related Work

| Paper | Venue | Link |
|-------|-------|------|
| SPFN - Supervised Primitive Fitting | CVPR 2019 | [arXiv](https://arxiv.org/abs/1811.08988), [code](https://github.com/lingxiaoli94/SPFN) |
| CPFN - Cascaded Primitive Fitting | ICCV 2021 | [code](https://github.com/erictuanle/CPFN) |
| PrimitiveNet - Primitive Instance Segmentation | ICCV 2021 | [code](https://github.com/hjwdzh/PrimitiveNet) |
| NVDNet - Split-and-Fit B-Rep Reconstruction | SIGGRAPH 2024 | [arXiv](https://arxiv.org/abs/2406.05261) |
| STEP-Parts - B-Rep Partition for CAD Learning | 2026-04 | arXiv 2604.14927 |
| MeshCNN - CNN for 3D Meshes | SIGGRAPH 2019 | [code](https://github.com/ranahanocka/MeshCNN) |

## License

MIT
