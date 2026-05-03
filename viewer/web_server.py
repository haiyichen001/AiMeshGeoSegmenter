"""
AiMeshGeoSegmenter Viewer — STEP + STL + Labels | port 8006
"""
import sys, json, struct, io, math, os
from pathlib import Path
import numpy as np
from flask import Flask, jsonify, request, send_file

DATA = Path(r"D:\AiMeshGeoSegmenter")
WEB_ROOT = Path(__file__).parent
STEP_DIR = DATA / "data" / "step"
STL_DIR = DATA / "data" / "stl"
LABEL_DIR = DATA / "data" / "labels"
app = Flask(__name__)

# STEP shape cache
_shape_cache = {}

def get_shape(part):
    if part not in _shape_cache:
        from OCC.Core.STEPControl import STEPControl_Reader
        sr = STEPControl_Reader()
        path = STEP_DIR / f"{part}.step"
        if not path.exists():
            return None
        if sr.ReadFile(str(path)) != 1:
            return None
        sr.TransferRoots()
        _shape_cache[part] = sr.OneShape()
    return _shape_cache[part]

# 8-class color map
COLORS = {
    "plane": "#4db8ff", "cylinder": "#44cc44", "sphere": "#ff44ff",
    "cone": "#ff8844", "torus": "#ffcc00", "fillet": "#00cccc",
    "chamfer": "#ff6644", "freeform": "#888888",
}

@app.route("/")
def index():
    return send_file(str(WEB_ROOT / "multi_viewer.html"))

@app.route("/labels")
def labels_page():
    return send_file(str(WEB_ROOT / "labels_viewer.html"))

@app.route("/api/parts")
def api_parts():
    parts = sorted([f.stem for f in STEP_DIR.glob("*.step")])
    return jsonify(parts)

# ── STEP mesh ──
@app.route("/api/mesh/<part>")
def api_mesh(part):
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    shape = get_shape(part)
    if shape is None:
        return jsonify({"error": "part not found"}), 404

    bbox = Bnd_Box(); brepbndlib.Add(shape, bbox)
    x1, y1, z1, x2, y2, z2 = bbox.Get()
    span = max(x2 - x1, y2 - y1, z2 - z1)
    BRepMesh_IncrementalMesh(shape, span / 100.0).Perform()

    all_verts, all_norms, all_tris = [], [], []
    vi = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = exp.Current()
        loc = TopLoc_Location()
        tri = BRep_Tool().Triangulation(face, loc)
        if tri is None: exp.Next(); continue
        trsf = loc.Transformation()
        nv = tri.NbNodes()
        local_idx = {}
        for i in range(1, nv + 1):
            p = tri.Node(i); p.Transform(trsf)
            all_verts.extend([p.X(), p.Y(), p.Z()])
            local_idx[i] = vi; vi += 1
        has_n = tri.HasNormals()
        for i in range(1, nv + 1):
            if has_n: n = tri.Normal(i); all_norms.extend([n.X(), n.Y(), n.Z()])
            else: all_norms.extend([0, 0, 1])
        for i in range(1, tri.NbTriangles() + 1):
            t = tri.Triangle(i)
            all_tris.extend([local_idx[t.Value(1)], local_idx[t.Value(2)], local_idx[t.Value(3)]])
        exp.Next()

    return jsonify({
        "vertices": all_verts, "normals": all_norms, "triangles": all_tris,
        "center": [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2], "span": span,
    })

# ── STL mesh ──
@app.route("/api/stl/<part>")
def api_stl(part):
    stl_path = STL_DIR / f"{part}.stl"
    if not stl_path.exists():
        return jsonify({"error": "STL not found"}), 404

    import trimesh as tm
    m = tm.load(str(stl_path))
    verts = m.vertices.astype(np.float32)
    faces = m.faces.astype(np.int32)
    norms = m.vertex_normals.astype(np.float32) if hasattr(m, 'vertex_normals') and m.vertex_normals is not None else np.zeros_like(verts)

    return jsonify({
        "vertices": verts.flatten().tolist(), "normals": norms.flatten().tolist(),
        "triangles": faces.flatten().tolist(),
        "center": m.centroid.tolist() if hasattr(m, 'centroid') else [0, 0, 0],
        "span": float(m.extents.max()) if hasattr(m, 'extents') else 1.0,
    })

# ── Labels (from precomputed JSON) ──
@app.route("/api/labels/<part>")
def api_labels(part):
    path = LABEL_DIR / f"{part}.json"
    if not path.exists():
        return jsonify({"error": "labels not found"}), 404

    with open(path) as f:
        data = json.load(f)

    faces_out = []
    for face in data["faces"]:
        label = face.get("label", "freeform")
        faces_out.append({
            "vertices": face.get("vertices", []),
            "triangles": face.get("triangles", []),
            "type": label,
            "color": COLORS.get(label, "#ffffff"),
            "center": face.get("center", [0, 0, 0]),
        })

    return jsonify({
        "faces": faces_out,
        "center": data.get("center", [0, 0, 0]),
        "span": data.get("span", 1.0),
    })

# ── Raw STL download ──
@app.route("/api/stl_raw/<part>")
def api_stl_raw(part):
    stl_path = STL_DIR / f"{part}.stl"
    if not stl_path.exists():
        return jsonify({"error": "STL not found"}), 404
    return send_file(str(stl_path), mimetype='application/octet-stream')

# ── Inference: upload STL → predict labels ──
@app.route("/infer")
def infer_page():
    return send_file(str(WEB_ROOT / "infer.html"))

@app.route("/api/infer", methods=["POST"])
def api_infer():
    import tempfile, subprocess

    if 'stl' not in request.files:
        return jsonify({"error": "no STL file"}), 400
    file = request.files['stl']
    if file.filename == '':
        return jsonify({"error": "empty filename"}), 400

    tmp = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
    try:
        file.save(tmp.name)
        tmp.close()
        py = r"C:\miniconda3\envs\occ\python.exe"
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        out = subprocess.run(
            [py, str(DATA / "scripts" / "infer.py"), tmp.name],
            capture_output=True, text=True, timeout=60,
            cwd=str(DATA), env=env
        )
        if out.returncode != 0:
            return jsonify({"error": out.stderr.strip()}), 500
        return jsonify(json.loads(out.stdout))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    print("AiMeshGeoSegmenter Viewer → http://localhost:8006")
    app.run(host="127.0.0.1", port=8006, debug=False)
