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
Per-triangle 6-class label (plane/cylinder/sphere/cone/torus/freeform)
```

Why GAT:
- Works directly on triangle mesh — no point cloud conversion needed
- Attention mechanism learns which neighbors belong to the same surface
- Single model, no domain mismatch between training and inference
- ~200K params, GPU training, CPU inference capable

## 6 Output Classes

| # | Class | Detection |
|---|-------|-----------|
| 1 | Plane | GeomAbs_Plane |
| 2 | Cylinder | GeomAbs_Cylinder |
| 3 | Sphere | GeomAbs_Sphere + BSpline least-squares recovery (error < 2%) |
| 4 | Cone | GeomAbs_Cone |
| 5 | Torus | GeomAbs_Torus |
| 6 | Freeform | Everything else (BSpline, Bezier, Revolution, Extrusion) |

Convexity is detected via `BRepAdaptor_Surface.D1()` exact surface normal at face midpoint.
All classification is done in a single pass by `label_faces.py`.

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

## Accuracy Improvement (Risk-Free)

以下手段无过拟合风险，纯粹堆算力即可稳定提升准确率：

| 手段 | 预期提升 | 代价 |
|------|---------|------|
| 增加训练样本 (2K → 5K → 全量) | +2~4% / 步 | 训练时间线性增长 |
| K 折交叉验证 (1 → 3 → 5) | +1~2% | 评估时间 K 倍 |
| 模型集成 (1 → 3 → 5 个种子) | +2~4% | 训练/推理时间 N 倍 |

当前基线：单模型 80/20 分割，2000 样本，供快速迭代。后期突破时堆上述手段即可。

## Status

- Dataset: 20,112 STEP/STL pairs, 197K annotated faces
- Training: in progress
- GPU: NVIDIA RTX 5060 Ti, 17 GB VRAM

## License

MIT
