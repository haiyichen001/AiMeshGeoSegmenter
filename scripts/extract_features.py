"""
从 labels JSON 提取 GNN 图数据集特征

每面特征 (18维):
  - area_log: log10(face area) [1]
  - normal: face平均法向量 [3]
  - normal_std: 法向量离散度 (0=平面, >0=曲面) [1]
  - center: 归一化面中心位置 [3]
  - n_triangles: 三角面片数 [1]
  - n_neighbors: 邻接面数 [1]
  - bbox_ratio: 包围盒高宽比 [3]
  - vertex_density: 顶点密度 [1]
  - area_ratio: 面面积占零件总比例 [1]
  - dihedral_mean/std: 与邻面的二面角均值/方差 [2]
  - occ_type_onehot: OCCT 类型 onehot [7] (只做辅助, 不作为最终特征)

输出: PyG Data 对象, 每个零件一个图
"""
import os, sys, json, collections, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count

ROOT = Path(r"D:\AiMeshGeoSegmenter")
LABEL_DIR = ROOT / "data" / "labels"
FEAT_DIR = ROOT / "data" / "graphs"
os.makedirs(FEAT_DIR, exist_ok=True)

OCC_TYPES = ["Plane", "Cylinder", "Cone", "Sphere", "Torus", "BSpline", "Bezier", "Extrusion", "Revolution", "Other"]
LABEL_NAMES = ["plane", "cylinder", "sphere", "cone", "torus", "fillet", "chamfer", "freeform"]
LABEL_TO_IDX = {n: i for i, n in enumerate(LABEL_NAMES)}


def compute_face_normals(vertices, triangles):
    """Compute per-triangle normals and area-weighted per-vertex normals."""
    v = np.array(vertices).reshape(-1, 3)
    t = np.array(triangles).reshape(-1, 3)

    # Per-triangle normals
    tri_normals = np.zeros((len(t), 3))
    tri_areas = np.zeros(len(t))
    for i, tri in enumerate(t):
        p0, p1, p2 = v[tri[0]], v[tri[1]], v[tri[2]]
        e1 = p1 - p0
        e2 = p2 - p0
        n = np.cross(e1, e2)
        area = np.linalg.norm(n)
        tri_areas[i] = area * 0.5
        if area > 1e-15:
            tri_normals[i] = n / area
        else:
            tri_normals[i] = np.array([0., 0., 1.])

    # Per-vertex normals from area-weighted triangle normals
    vert_normals = np.zeros((len(v), 3))
    for i, tri in enumerate(t):
        for j in tri:
            vert_normals[j] += tri_normals[i] * tri_areas[i]
    nrm = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    nrm[nrm < 1e-15] = 1.0
    vert_normals /= nrm

    return vert_normals, tri_normals, tri_areas


def process_one(label_file):
    stem = label_file.replace(".json", "")
    out_path = os.path.join(FEAT_DIR, f"{stem}.npz")

    with open(os.path.join(LABEL_DIR, label_file)) as f:
        data = json.load(f)

    faces = data["faces"]
    if not faces:
        return (stem, "empty")

    n_faces = len(faces)
    part_span = data.get("span", 1.0)
    part_center = np.array(data.get("center", [0., 0., 0.]))

    # Build face index map
    face_index = {f["id"]: idx for idx, f in enumerate(faces)}

    total_area = sum(f.get("area", 0) for f in faces)

    features = np.zeros((n_faces, 18), dtype=np.float32)
    edge_index = [[], []]
    labels = np.zeros(n_faces, dtype=np.int64)

    for idx, f in enumerate(faces):
        verts = f["vertices"]
        tris = f["triangles"]
        area = f.get("area", 0)
        nbrs = f.get("neighbors", [])
        n_nbrs = len(nbrs)

        if len(verts) >= 9 and len(tris) >= 3:
            v_normals, tri_normals, tri_areas = compute_face_normals(verts, tris)

            # Normal stats
            mean_normal = v_normals.mean(axis=0)
            nrm = np.linalg.norm(mean_normal)
            if nrm > 1e-15:
                mean_normal /= nrm

            # Normal variance (key curvature proxy)
            normal_std = v_normals.std(axis=0).mean()

            # Mesh info
            n_verts = len(verts) // 3
            n_tris = len(tris) // 3
        else:
            mean_normal = np.array([0., 0., 1.])
            normal_std = 0.0
            n_verts = 0
            n_tris = 0

        # Normalized center
        center = np.array(f.get("center", [0., 0., 0.]))
        rel_center = (center - part_center) / max(part_span, 1e-6)

        # Area features
        area_log = np.log10(max(area, 1e-6))
        area_ratio = area / max(total_area, 1e-6)

        # BBox ratio approximation (from vertex spread)
        if len(verts) >= 9:
            v_arr = np.array(verts).reshape(-1, 3)
            bbox_size = v_arr.max(axis=0) - v_arr.min(axis=0)
            bbox_size = bbox_size / max(bbox_size.max(), 1e-6)
        else:
            bbox_size = np.array([1., 1., 1.])

        # Vertex density
        vert_density = n_verts / max(area, 1e-6)

        # Dihedral angles with neighbors (computed from face normals)
        dihedral_angles = []
        for nbr_id in nbrs:
            nbr_idx = face_index.get(nbr_id)
            if nbr_idx is not None:
                nbr_f = faces[nbr_idx]
                nbr_verts = nbr_f.get("vertices", [])
                if len(nbr_verts) >= 9:
                    nbr_vn, _, _ = compute_face_normals(nbr_verts, nbr_f["triangles"])
                    nbr_mn = nbr_vn.mean(axis=0)
                    nrm2 = np.linalg.norm(nbr_mn)
                    if nrm2 > 1e-15:
                        nbr_mn /= nrm2
                    dot = abs(np.dot(mean_normal, nbr_mn))
                    dot = min(1.0, max(0.0, dot))
                    angle = np.arccos(dot) * 180 / np.pi
                    dihedral_angles.append(angle)

        if dihedral_angles:
            dihedral_mean = np.mean(dihedral_angles)
            dihedral_std = np.std(dihedral_angles)
        else:
            dihedral_mean = 0.0
            dihedral_std = 0.0

        # Feature vector
        ft = [
            area_log,                          # 0
            mean_normal[0], mean_normal[1], mean_normal[2],  # 1-3
            normal_std,                        # 4
            rel_center[0], rel_center[1], rel_center[2],    # 5-7
            np.log10(max(n_tris, 1)),          # 8
            float(n_nbrs),                     # 9
            bbox_size[0], bbox_size[1], bbox_size[2],       # 10-12
            np.log10(max(vert_density, 1e-6)), # 13
            area_ratio,                        # 14
            dihedral_mean,                     # 15
            dihedral_std,                      # 16
            np.log10(max(n_verts, 1)),         # 17
        ]
        features[idx] = np.array(ft, dtype=np.float32)

        # Edges
        for nbr_id in nbrs:
            nbr_idx = face_index.get(nbr_id)
            if nbr_idx is not None:
                edge_index[0].append(idx)
                edge_index[1].append(nbr_idx)

        # Label
        label_name = f.get("label", "freeform")
        labels[idx] = LABEL_TO_IDX.get(label_name, LABEL_TO_IDX["freeform"])

    # Save
    np.savez_compressed(out_path,
                        x=features,
                        edge_index=np.array(edge_index, dtype=np.int64),
                        y=labels,
                        part_name=stem,
                        num_nodes=n_faces)

    # Stats
    label_counts = collections.Counter()
    for l in labels:
        label_counts[LABEL_NAMES[l]] += 1
    return (stem, dict(label_counts))


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(LABEL_DIR)
                   if f.endswith('.json') and f != 'distribution.json')

    n = max(1, cpu_count() - 1)
    print(f"Files: {len(files)}, Workers: {n}")

    ok, fail = 0, 0
    global_stats = collections.Counter()
    t0 = time.time()

    with Pool(n) as p:
        for result in p.imap_unordered(process_one, files, chunksize=20):
            if result[1] == "empty":
                fail += 1
            else:
                ok += 1
                global_stats.update(result[1])
            total = ok + fail
            if total % 500 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                print(f"  [{total}/{len(files)}] ok={ok} fail={fail} | {rate:.1f} p/s")

    elapsed = time.time() - t0
    total_faces = sum(global_stats.values())
    print(f"\nDONE: ok={ok} fail={fail} in {elapsed:.0f}s")
    print(f"Graph files: {len(list(FEAT_DIR.glob('*.npz')))}")
    print(f"Total faces: {total_faces}")
    print(f"Label distribution: {dict(global_stats.most_common())}")
