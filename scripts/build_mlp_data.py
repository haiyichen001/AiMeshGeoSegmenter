"""
构建 MLP 训练样本集: 从 STL+labels 预计算边缘特征, 存到 data/mlp_edges/
"""
import os, json, random, time, collections, numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import trimesh
from scipy.spatial import KDTree

ROOT = Path(r"D:\AiMeshGeoSegmenter")
STL_DIR = ROOT / "data" / "stl"
LABEL_DIR = ROOT / "data" / "labels"
MLP_DIR = ROOT / "data" / "mlp_edges"
os.makedirs(MLP_DIR, exist_ok=True)


def process_one(label_file):
    stem = label_file.replace('.json', '')
    stl_path = STL_DIR / f"{stem}.stl"
    out_path = MLP_DIR / f"{stem}.npz"
    if out_path.exists():
        return (stem, {"_skip": 1})

    if not stl_path.exists():
        return (stem, {"_no_stl": 1})

    try:
        with open(os.path.join(LABEL_DIR, label_file)) as f:
            data = json.load(f)
        faces = data["faces"]
        if not faces:
            return (stem, {"_empty": 1})

        mesh = trimesh.load(str(stl_path))
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        stl_verts = np.array(mesh.vertices, dtype=np.float32)
        stl_faces = np.array(mesh.faces, dtype=np.int32)
        stl_adj = mesh.face_adjacency
        if len(stl_adj) == 0:
            return (stem, {"_no_adj": 1})

        # Map STL triangles to face IDs via KD-tree
        all_centers, all_ids = [], []
        for face in faces:
            verts = face.get("vertices", [])
            tris = face.get("triangles", [])
            if len(verts) < 9 or len(tris) < 3: continue
            v = np.array(verts).reshape(-1, 3)
            t = np.array(tris).reshape(-1, 3)
            centers = v[t].mean(axis=1)
            all_centers.append(centers)
            all_ids.append(np.full(len(centers), face["id"], dtype=np.int32))
        if not all_centers: return (stem, {"_no_label": 1})

        tree = KDTree(np.vstack(all_centers))
        stl_centers = stl_verts[stl_faces].mean(axis=1)
        _, idx = tree.query(stl_centers)
        stl_face_ids = np.concatenate(all_ids)[idx]

        # Per-triangle normals
        e1 = stl_verts[stl_faces[:,1]] - stl_verts[stl_faces[:,0]]
        e2 = stl_verts[stl_faces[:,2]] - stl_verts[stl_faces[:,0]]
        tri_normals = np.cross(e1, e2)
        tri_nrm = np.linalg.norm(tri_normals, axis=1, keepdims=True).clip(1e-15)
        tri_areas = tri_nrm.flatten() * 0.5
        tri_normals /= tri_nrm

        # Edge features
        X, y = [], []
        for a, b in stl_adj:
            dihedral = np.arccos(np.clip(np.dot(tri_normals[a], tri_normals[b]), -1, 1)) * 180 / np.pi
            area_ratio = min(tri_areas[a], tri_areas[b]) / max(max(tri_areas[a], tri_areas[b]), 1e-12)
            shared = set(stl_faces[a]) & set(stl_faces[b])
            edge_len = 0.001
            if len(shared) >= 2:
                s = list(shared)[:2]
                edge_len = np.linalg.norm(stl_verts[s[0]] - stl_verts[s[1]])
            avg_side = np.sqrt(max(tri_areas[a], 1e-12) * 2 / np.sqrt(3))
            edge_ratio = edge_len / max(avg_side, 1e-6)
            # Triangle shape features
            def tri_shape(vi):
                pts = stl_verts[stl_faces[vi]]
                sides = [np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[1]-pts[2]), np.linalg.norm(pts[2]-pts[0])]
                sides.sort()
                aspect = sides[2] / max(sides[0], 1e-6)  # longest/shortest
                # max angle via law of cosines
                cos_max = (sides[0]**2 + sides[1]**2 - sides[2]**2) / max(2*sides[0]*sides[1], 1e-12)
                max_angle = np.arccos(np.clip(cos_max, -1, 1)) * 180 / np.pi
                return aspect, max_angle
            asp_a, ang_a = tri_shape(a); asp_b, ang_b = tri_shape(b)
            # Use the worse shape between the two triangles
            aspect_ratio_tri = max(asp_a, asp_b)
            max_angle_tri = max(ang_a, ang_b)
            X.append([dihedral, dihedral, area_ratio, edge_ratio, 0.0,
                      aspect_ratio_tri, max_angle_tri, min(asp_a, asp_b)])
            y.append(1 if stl_face_ids[a] == stl_face_ids[b] else 0)

        if len(X) < 10:
            return (stem, {"_too_few": 1})

        np.savez_compressed(out_path, X=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.int8))

        pos = sum(y); neg = len(y) - pos
        return (stem, {"pos": pos, "neg": neg})

    except Exception as e:
        return (stem, {"_err": str(e)[:100]})


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(LABEL_DIR) if f.endswith('.json') and f != 'distribution.json')
    n = max(1, cpu_count() - 1)
    print(f"Files: {len(files)}, Workers: {n}")

    ok = fail = 0; total_pos = 0; total_neg = 0; t0 = time.time()
    with Pool(n) as p:
        for stem, stats in p.imap_unordered(process_one, files, chunksize=20):
            if any(k.startswith('_') for k in stats):
                fail += 1
            else:
                ok += 1; total_pos += stats.get("pos", 0); total_neg += stats.get("neg", 0)
            total = ok + fail
            if total % 1000 == 0:
                print(f"  [{total}/{len(files)}] ok={ok} fail={fail}")

    print(f"\nDONE: ok={ok} fail={fail} in {time.time()-t0:.0f}s")
    total_edges = total_pos + total_neg
    print(f"Files: {ok}, Edges: {total_edges}")
    print(f"  Merge (pos): {total_pos} ({total_pos/max(total_edges,1)*100:.1f}%)")
    print(f"  Cut (neg):   {total_neg} ({total_neg/max(total_edges,1)*100:.1f}%)")
