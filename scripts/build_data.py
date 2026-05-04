"""
构建 GAT 训练数据: STL 三角形 + per-triangle B-Rep 面标签

每零件一个 NPZ: vertices, faces, per_tri_normals, per_tri_label
"""
import os, json, time, collections, numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import trimesh
from scipy.spatial import KDTree

ROOT = Path(r"D:\AiMeshGeoSegmenter")
STL_DIR = ROOT / "data" / "stl"
LABEL_DIR = ROOT / "data" / "labels"
DATA_DIR = ROOT / "data" / "patches"
os.makedirs(DATA_DIR, exist_ok=True)

LABEL_NAMES = ["plane","cylinder","sphere","cone","torus","fillet","chamfer","freeform"]
L2I = {n:i for i,n in enumerate(LABEL_NAMES)}


def process_one(label_file):
    stem = label_file.replace('.json', '')
    stl_path = STL_DIR / f"{stem}.stl"
    label_path = LABEL_DIR / label_file
    out_path = DATA_DIR / f"{stem}.npz"
    if out_path.exists(): return (stem, {"_skip":1})

    if not stl_path.exists(): return (stem, {"_no_stl":1})

    try:
        with open(label_path) as f: label_data = json.load(f)
        faces = label_data["faces"]
        if not faces: return (stem, {"_empty":1})

        mesh = trimesh.load(str(stl_path))
        if isinstance(mesh, trimesh.Scene): mesh = trimesh.util.concatenate(mesh.dump())
        stl_verts = np.array(mesh.vertices, dtype=np.float32)
        stl_faces = np.array(mesh.faces, dtype=np.int32)

        # KD-tree: STL triangle center -> nearest B-Rep face
        all_ctrs, all_fids = [], []
        for face in faces:
            verts = face.get("vertices",[]); tris_ = face.get("triangles",[])
            if len(verts)<9 or len(tris_)<3: continue
            v = np.array(verts).reshape(-1,3); t = np.array(tris_).reshape(-1,3)
            ctrs = v[t].mean(axis=1)
            all_ctrs.append(ctrs); all_fids.append(np.full(len(ctrs), face["id"], dtype=np.int32))
        if not all_ctrs: return (stem, {"_no_label":1})

        tree = KDTree(np.vstack(all_ctrs))
        stl_ctrs = stl_verts[stl_faces].mean(axis=1)
        _, idx = tree.query(stl_ctrs)
        stl_fids = np.concatenate(all_fids)[idx]

        # Face label mapping
        fl_map = {f["id"]: L2I.get(f.get("label","freeform"), L2I["freeform"]) for f in faces}
        max_fid = max(fl_map.keys()) if fl_map else 0
        per_tri_label = np.array([fl_map.get(min(fid, max_fid), L2I["freeform"]) for fid in stl_fids], dtype=np.int8)

        # Per-triangle normals
        e1 = stl_verts[stl_faces[:,1]] - stl_verts[stl_faces[:,0]]
        e2 = stl_verts[stl_faces[:,2]] - stl_verts[stl_faces[:,0]]
        tri_normals = np.cross(e1, e2)
        nrm = np.linalg.norm(tri_normals, axis=1, keepdims=True).clip(1e-15)
        tri_normals /= nrm
        tri_areas = nrm.flatten() * 0.5

        np.savez_compressed(out_path,
            vertices=stl_verts, faces=stl_faces,
            tri_normals=tri_normals, tri_areas=tri_areas,
            tri_labels=per_tri_label, part_name=stem)

        dist = collections.Counter([LABEL_NAMES[l] for l in per_tri_label])
        return (stem, dict(dist))
    except Exception as e:
        return (stem, {"_err": str(e)[:100]})


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(LABEL_DIR) if f.endswith('.json') and f != 'distribution.json')
    n = max(1, cpu_count()-1)
    print(f"Files: {len(files)}, Workers: {n}")
    ok = fail = 0; stats = collections.Counter(); t0 = time.time()
    todo = [f for f in files if not (DATA_DIR/f.replace('.json','.npz')).exists()]
    print(f"To do: {len(todo)}")
    with Pool(n) as p:
        for stem, s in p.imap_unordered(process_one, todo, chunksize=20):
            if "_err" in s or "_no" in s or "_empty" in s: fail += 1
            else: ok += 1; stats.update(s)
            if (ok+fail) % 1000 == 0: print(f"  ok={ok} fail={fail}")
    elapsed = time.time()-t0
    print(f"\nDONE: ok={ok} fail={fail} in {elapsed/60:.1f}min")
    total_tris = sum(stats.values())
    print(f"Parts: {ok}, Triangles: {total_tris/1e6:.1f}M")
    for name in LABEL_NAMES: print(f"  {name:12s} {stats.get(name,0):10d}")
