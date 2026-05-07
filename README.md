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

## Model Input (36-dim per-triangle features)

| # | Feature | Category |
|---|---------|----------|
| 0-2 | Normal vector (nx, ny, nz) | Geometry |
| 3-5 | Fourier position encoding (x: sin/cos πx·2πx) | Position |
| 6-8 | Fourier position encoding (y) | Position |
| 9-11 | Fourier position encoding (z) | Position |
| 12 | log10(area) | Scale |
| 13 | Neighbor normal variance | 1-hop |
| 14 | Mean dihedral angle | 1-hop |
| 15 | Max dihedral angle | 1-hop |
| 16 | Compactness | Shape |
| 17 | Edge length ratio | Shape |
| 18 | Vertex normal std | Curvature |
| 19-24 | Fourier position encoding (3π) | Position |
| 25 | log(total triangles) | Part-level |
| 26-28 | Part elongation (X/Y/Z axis ratio) | Part-level |
| 29-30 | 2-hop mean/std dihedral | Multi-scale |
| 31-32 | 2-hop mean/std area ratio | Multi-scale |
| 33-35 | 4-hop mean/std dihedral + area | Multi-scale |

Edge features (3-dim): dihedral angle, relative edge length, normal-direction sign.

Model: 3-layer GAT (192 hidden, 4 heads, 695K params) with edge features, SWA, Focal Loss.

## Training Data

## Ultimate Goal: STL → STEP Reverse Engineering

```
STL file (triangle mesh)
  │
  ├─ GAT: per-triangle 6-class classification
  ├─ MLP Edge: boundary detection
  ├─ Connected Components + Vote: group triangles into faces
  │
  ▼
Per-face numerical fitting:
  ├─ Plane: RANSAC + least-squares
  ├─ Cylinder: axis estimation + radius LSQ
  ├─ Cone: apex + semi-angle fitting
  ├─ Sphere: center + radius LSQ
  ├─ Torus: axis + major/minor radius
  └─ Freeform: B-Spline surface interpolation
  │
  ▼
pythonOCC B-Rep construction → STEP output
```

## Training Data

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

## Best Configuration (2K samples, sweep results)

| Rank | Config | Test Acc | Time |
|------|--------|----------|------|
| 1 | 26-dim + 3-layer + JK | **83.04%** | 280s |
| 2 | 26-dim + 4-layer | 82.91% | 405s |
| 3 | 26-dim + 256h + 4L | 82.85% | 532s |
| 4 | 26-dim baseline | 82.80% | 296s |
| 5 | 14-dim baseline | 81.94% | 288s |

Key techniques: Jumping Knowledge (JK), AMP, DropEdge, SWA, Focal Loss, GPU-direct training.

## Results

| Version | Samples | Key Changes | Test Acc |
|---------|---------|-------------|----------|
| v1 | 2K | 10-dim baseline | 63.25% |
| v2 | 2K | 14-dim + curvature + Focal Loss | 71.23% |
| v3 | 2K | 26-dim + JK + AMP + DropEdge + SWA | 83.04% |
| v4 | 20K | 3-seed ensemble | **87.88%** |

Per-class accuracy (v4 ensemble, 2,986 test parts):
- torus 97.06%, plane 90.24%, cone 88.57%, cylinder 87.47%, sphere 73.99%, freeform 64.83%

## Limitations

- **Sphere label quality**: OCCT GeomAbs_Sphere only. Freeform→sphere recovery requires >=100 vertices + subdivision + <0.5% RMS error. Extrusion surfaces excluded. Small false-sphere faces eliminated.
- **Cylinder/Cone/Torus recovery from freeform**: Not yet implemented (requires RANSAC iterative fitting, ~20-30 min). OCCT-mislabeled freeform faces may contain hidden cylinders/cones.
- **Per-triangle inconsistency**: same B-Rep face may have different labels on adjacent triangles. Mitigated by MLP edge classifier + area-weighted voting.
- **Rotation sensitivity**: Fourier position encoding is not rotation-invariant. Rotated parts may produce different predictions.
- **Mesh resolution**: Low-vertex-count faces (<100 vertices) cannot reliably fit geometric primitives. Subdivision is used as a workaround for sphere fitting only.

## Pipeline (MVP)

```
STL mesh
  │
  ├─ GAT (26-dim + JK + AMP + DropEdge)
  │     └→ per-triangle 6-class prediction (87.9% ensemble test acc)
  │
  ├─ MLP Edge Classifier (4-dim, 95.4%)
  │     └→ detect face boundaries (adjacent triangles: same face? yes/no)
  │
  ├─ Connected Components
  │     └→ group triangles into regions by MLP-predicted boundaries
  │
  └─ Area-Weighted Majority Vote
        └→ per-face final label
           → plane / cylinder / sphere / cone / torus / freeform
```

Infer page shows 3 views side-by-side:
- Left: GAT raw per-triangle predictions (wireframe)
- Mid: MLP regions with GAT raw colors (no vote) — see boundary quality
- Right: MLP Voted (final per-face labels with area-weighted vote)

## Model Architecture

| Component | Detail |
|-----------|--------|
| GAT | 3-layer + Jumping Knowledge, 192h×4, 919K params |
| GAT Input | 26-dim: normals(3) + Fourier(12) + geometry/stats(11) + edge(3) |
| Edge MLP | 4→64→32→16→1, 2.9K params, 95.4% acc |
| MLP Input | 4-dim: dihedral angle, area ratio, dist, convexity |
| Training | AMP + Focal Loss(γ=2) + DropEdge 15% + SWA + Cosine LR

## Future Improvements

- [ ] Retrain GAT with per-epoch logging (3-seed × 3-fold = 9 curves)
- [ ] Cylinder/Cone/Torus recovery from freeform faces (RANSAC fitting, ~20-30 min)
- [ ] Rotation-invariant features for GAT
- [ ] Test-time augmentation for inference
- [ ] STL→STEP full pipeline (numerical fitting + B-Rep construction)

## API

Model inference API available at [<redacted>](https://<redacted>):

```
POST https://<redacted>/api/infer
Content-Type: multipart/form-data
Body: stl=<file>

Response: JSON with per-face classification and mesh data
```

## Status

- Dataset: 19,902 STEP/STL pairs, ~200K annotated faces
- GAT: 919K params, 88.1% val / 87.6% test
- MLP Edge: 3-seed ensemble, 99.3% accuracy
- GPU: NVIDIA RTX 5060 Ti, 16 GB VRAM

## License

MIT
