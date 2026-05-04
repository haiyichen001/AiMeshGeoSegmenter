"""
从 STL 提取 GNN 图特征（训练用 STL 三角，与推理一致，零域偏移）

流程:
  1. 加载 label JSON（每面标签）
  2. 加载 STL 文件
  3. KD 树映射 STL 三角 → B-Rep 面
  4. 每面从 STL 三角提 18 维特征
  5. 从 STL 网格构建面邻接图
  6. 保存 NPZ
"""
import os, sys, json, time, collections
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count

ROOT = Path(r"D:\AiMeshGeoSegmenter")
STL_DIR = ROOT / "data" / "stl"
LABEL_DIR = ROOT / "data" / "labels"
GRAPH_DIR = ROOT / "data" / "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

LABEL_NAMES = ["plane", "cylinder", "sphere", "cone", "torus", "fillet", "chamfer", "freeform"]
LABEL_TO_IDX = {n: i for i, n in enumerate(LABEL_NAMES)}


def process_one(label_file):
    stem = label_file.replace('.json', '')
    label_path = LABEL_DIR / label_file
    stl_path = STL_DIR / f"{stem}.stl"
    out_path = GRAPH_DIR / f"{stem}.npz"

    if out_path.exists():
        return (stem, {"_skip": 1})

    if not stl_path.exists():
        return (stem, {"_no_stl": 1})

    try:
        with open(label_path) as f:
            label_data = json.load(f)
        faces = label_data["faces"]
        if not faces:
            return (stem, {"_empty": 1})

        # Load STL
        mesh = trimesh.load(str(stl_path))
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        stl_verts = np.array(mesh.vertices, dtype=np.float32)
        stl_faces = np.array(mesh.faces, dtype=np.int32)
        n_stl_tris = len(stl_faces)

        if n_stl_tris < 3:
            return (stem, {"_too_few_tris": 1})

        # Build KD-tree from label mesh vertices to map STL triangles to faces
        all_label_centers = []
        all_label_face_ids = []
        for face in faces:
            verts = face.get("vertices", [])
            tris = face.get("triangles", [])
            if len(verts) < 9 or len(tris) < 3:
                continue
            v = np.array(verts).reshape(-1, 3)
            t = np.array(tris).reshape(-1, 3)
            centers = v[t].mean(axis=1)
            all_label_centers.append(centers)
            all_label_face_ids.append(np.full(len(centers), face["id"], dtype=np.int32))

        if not all_label_centers:
            return (stem, {"_no_label_mesh": 1})

        all_centers = np.vstack(all_label_centers)
        all_ids = np.concatenate(all_label_face_ids)
        tree = KDTree(all_centers)

        # Map STL triangles to face IDs
        stl_centers = stl_verts[stl_faces].mean(axis=1)
        _, idx = tree.query(stl_centers)
        stl_face_ids = all_ids[idx]

        # Group STL triangles by face
        unique_faces = sorted(set(stl_face_ids))
        n_faces = len(unique_faces)
        if n_faces < 2:
            return (stem, {"_single_face": 1})

        face_id_to_idx = {fid: i for i, fid in enumerate(unique_faces)}

        # Build per-face normals and features from STL triangles
        stl_normals = mesh.face_normals
        stl_areas = mesh.area_faces
        total_area = stl_areas.sum()
        part_center = stl_verts.mean(axis=0)
        part_span = float(max(stl_verts.max(axis=0) - stl_verts.min(axis=0)))
        if part_span < 1e-6: part_span = 1.0

        # Adjacency between STL triangles
        stl_adj = mesh.face_adjacency

        # Build face-level adjacency: two faces are adjacent if any of their STL triangles are adjacent
        face_adj = {i: set() for i in range(n_faces)}
        for a, b in stl_adj:
            fa = stl_face_ids[a]
            fb = stl_face_ids[b]
            if fa != fb:
                ia = face_id_to_idx[fa]
                ib = face_id_to_idx[fb]
                face_adj[ia].add(ib)
                face_adj[ib].add(ia)

        # Per-face features
        features = np.zeros((n_faces, 26), dtype=np.float32)
        edge_list = [[], []]
        labels = np.zeros(n_faces, dtype=np.int64)

        # Face label mapping (from original label data)
        face_label_map = {f["id"]: f["label"] for f in faces}

        for fid in unique_faces:
            fi = face_id_to_idx[fid]
            tri_mask = stl_face_ids == fid
            tri_indices = np.where(tri_mask)[0]
            n_tris_face = len(tri_indices)
            if n_tris_face == 0:
                continue

            # Face-level stats from STL triangles
            face_area = stl_areas[tri_indices].sum()
            face_normals = stl_normals[tri_indices]
            mean_normal = face_normals.mean(axis=0)
            nr = np.linalg.norm(mean_normal)
            if nr > 1e-15: mean_normal /= nr

            normal_std = face_normals.std(axis=0).mean()
            face_centers = stl_centers[tri_indices].mean(axis=0)
            rel_center = (face_centers - part_center) / max(part_span, 1e-6)

            # Vertex stats
            face_verts_set = set()
            for ti in tri_indices:
                face_verts_set.update(stl_faces[ti])
            n_verts_face = len(face_verts_set)

            nbrs = sorted(face_adj.get(fi, set()))
            n_nbrs = len(nbrs)

            # Dihedral angles with neighbors
            dihedral = []
            for nb in nbrs:
                nb_fid = unique_faces[nb]
                nb_mask = stl_face_ids == nb_fid
                if nb_mask.sum() > 0:
                    nb_normal = stl_normals[nb_mask].mean(axis=0)
                    nbr_nr = np.linalg.norm(nb_normal)
                    if nbr_nr > 1e-15: nb_normal /= nbr_nr
                    dot = abs(np.dot(mean_normal, nb_normal))
                    dot = min(1.0, max(0.0, dot))
                    dihedral.append(np.arccos(dot) * 180 / np.pi)
            dih_mean = np.mean(dihedral) if dihedral else 0.0
            dih_std = np.std(dihedral) if dihedral else 0.0

            area_log = np.log10(max(face_area, 1e-6))
            area_ratio = face_area / max(total_area, 1e-6)
            n_tris_log = np.log10(max(n_tris_face, 1))
            vert_density_log = np.log10(max(n_verts_face / max(face_area, 1e-6), 1e-6))
            n_verts_log = np.log10(max(n_verts_face, 1))

            # Curvature features
            angle = np.arccos(np.clip(np.dot(face_normals, mean_normal), -1, 1)) * 180 / np.pi
            na_mean = float(angle.mean()); na_std = float(angle.std())
            na_max = float(angle.max()); na_range = float(angle.max() - angle.min())

            # BBox ratio
            if n_verts_face >= 3:
                fv = np.array([stl_verts[v] for v in face_verts_set])
                bs = fv.max(axis=0) - fv.min(axis=0)
                bs = bs / max(bs.max(), 1e-6)
            else:
                bs = np.array([1., 1., 1.])

            # Curvature features from normal covariance
            na_mean = na_std = na_max = na_range = 0.0
            cov_eig1 = cov_eig2 = cov_eig3 = 0.0
            if n_tris_face >= 2:
                fn_cov = np.cov(face_normals.T)
                eigvals = np.linalg.eigvalsh(fn_cov)
                cov_eig1, cov_eig2, cov_eig3 = float(eigvals[2]), float(eigvals[1]), float(eigvals[0])
                angles = np.arccos(np.clip(np.dot(face_normals, mean_normal), -1, 1)) * 180 / np.pi
                na_mean = float(angles.mean())
                na_std = float(angles.std())
                na_max = float(angles.max())
                na_range = float(angles.max() - angles.min())

            features[fi] = [
                area_log, mean_normal[0], mean_normal[1], mean_normal[2],
                normal_std, rel_center[0], rel_center[1], rel_center[2],
                n_tris_log, float(n_nbrs), bs[0], bs[1], bs[2],
                vert_density_log, area_ratio, dih_mean, dih_std, n_verts_log,
                na_mean, na_std, na_max, na_range,
                cov_eig1, cov_eig2, cov_eig3, np.log10(max(cov_eig1 + cov_eig2 + cov_eig3, 1e-12)),
            ]

            for nb in nbrs:
                edge_list[0].append(fi)
                edge_list[1].append(nb)

            label_name = face_label_map.get(fid, "freeform")
            labels[fi] = LABEL_TO_IDX.get(label_name, LABEL_TO_IDX["freeform"])

        np.savez_compressed(out_path,
                            x=features,
                            edge_index=np.array(edge_list, dtype=np.int64),
                            y=labels,
                            part_name=stem,
                            num_nodes=n_faces)

        label_counts = collections.Counter([LABEL_NAMES[int(l)] for l in labels])
        return (stem, dict(label_counts))

    except Exception as e:
        return (stem, {"_err": str(e)[:100]})


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(LABEL_DIR) if f.endswith('.json') and f != 'distribution.json')
    n = max(1, cpu_count() - 1)
    print(f"Files: {len(files)}, Workers: {n}")

    ok = 0; fail = 0; global_stats = collections.Counter(); t0 = time.time()
    with Pool(n) as p:
        for stem, stats in p.imap_unordered(process_one, files, chunksize=20):
            if any(k.startswith('_') for k in stats):
                fail += 1
            else:
                ok += 1; global_stats.update(stats)
            total = ok + fail
            if total % 500 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                eta = (len(files) - total) / rate / 60 if rate > 0 else 0
                print(f"  [{total}/{len(files)}] ok={ok} fail={fail} | {rate:.1f}/s | ETA {eta:.0f}min")

    elapsed = time.time() - t0
    total_faces = sum(global_stats.values())
    print(f"\nDONE: ok={ok} fail={fail} in {elapsed/60:.1f}min")
    print(f"Graphs: {ok}, Total faces: {total_faces}")
    for name in LABEL_NAMES:
        c = global_stats.get(name, 0)
        print(f"  {name:12s} {c:8d} ({c/max(total_faces,1)*100:5.1f}%)")
