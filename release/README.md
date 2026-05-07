# AiMeshGeoSegmenter - Release v1.0

Surface type classification for 3D triangle meshes. Given an STL file, predicts per-face labels: plane, cylinder, sphere, cone, torus, or freeform.

## Quick Start

```bash
pip install -r requirements.txt
python infer_standalone.py input.stl
```

Outputs JSON with per-face classification, vertex data, and colors.

## Model

| Component | Architecture | Accuracy |
|-----------|-------------|----------|
| GAT classifier | 3-layer + JK, 192h x4, 919K params | 87.5% (3-fold CV) |
| Edge classifier | MLP 4-64-32-16-1, 3-seed ensemble | 99.3% |

## Files

- `model.pt` - GAT classifier weights
- `edge_classifier_0/1/2.pt` - MLP edge classifier ensemble
- `infer_standalone.py` - inference script (zero external imports beyond pip)
- `requirements.txt` - Python dependencies

## API Response

```json
{
  "faces": [
    {"vertices": [...], "triangles": [...], "type": "plane", "color": "#4db8ff", "center": [...]},
    ...
  ],
  "center": [x, y, z],
  "span": 123.4,
  "num_faces": 13
}
```

## License

MIT
