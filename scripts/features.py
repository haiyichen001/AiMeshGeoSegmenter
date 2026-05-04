"""
共享特征提取: 训练和推理用同一套代码
"""
import numpy as np
from collections import defaultdict
import trimesh

LABEL_NAMES = ["plane","cylinder","sphere","cone","torus","fillet","chamfer","freeform"]


def compute_patch_features(vertices, faces, patch_labels):
    """从 STL 三角形 + 面片标签提取 26 维特征."""
    n_patches = patch_labels.max() + 1
    e1 = vertices[faces[:,1]] - vertices[faces[:,0]]
    e2 = vertices[faces[:,2]] - vertices[faces[:,0]]
    tri_normals = np.cross(e1, e2)
    tri_nrm = np.linalg.norm(tri_normals, axis=1, keepdims=True).clip(1e-15)
    face_areas = tri_nrm.flatten() * 0.5
    tri_normals /= tri_nrm
    tri_centers = vertices[faces].mean(axis=1)
    part_center = vertices.mean(axis=0)
    part_span = float(max(vertices.max(axis=0) - vertices.min(axis=0)))
    if part_span < 1e-6: part_span = 1.0

    patch_faces = [[] for _ in range(n_patches)]
    for fi, pid in enumerate(patch_labels): patch_faces[pid].append(fi)

    total_area = face_areas.sum()

    # Patch adjacency
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    adj = mesh.face_adjacency
    patch_adj = defaultdict(set)
    for a, b in adj:
        pa, pb = patch_labels[a], patch_labels[b]
        if pa != pb: patch_adj[pa].add(pb); patch_adj[pb].add(pa)

    features = np.zeros((n_patches, 26), dtype=np.float32)
    edge_index = [[], []]

    for pid in range(n_patches):
        tri_indices = patch_faces[pid]
        n_tris_face = len(tri_indices)
        if n_tris_face == 0: continue

        face_area = face_areas[tri_indices].sum()
        face_normals = tri_normals[tri_indices]
        mean_normal = face_normals.mean(axis=0)
        nr = np.linalg.norm(mean_normal)
        if nr > 1e-15: mean_normal /= nr
        normal_std = face_normals.std(axis=0).mean()
        face_centers = tri_centers[tri_indices].mean(axis=0)
        rel_center = (face_centers - part_center) / max(part_span, 1e-6)

        face_verts_set = set()
        for ti in tri_indices: face_verts_set.update(faces[ti])
        n_verts_face = len(face_verts_set)

        nbrs = sorted(patch_adj.get(pid, set())); n_nbrs = len(nbrs)

        dihedral = []
        for nb in nbrs:
            nb_mask = patch_faces[nb]
            if nb_mask:
                nb_normal = tri_normals[nb_mask].mean(axis=0)
                nbr_nr = np.linalg.norm(nb_normal)
                if nbr_nr > 1e-15: nb_normal /= nbr_nr
                dot = abs(np.dot(mean_normal, nb_normal))
                dot = min(1.0, max(0.0, dot))
                dihedral.append(np.arccos(dot)*180/np.pi)
        dih_mean = np.mean(dihedral) if dihedral else 0.0
        dih_std = np.std(dihedral) if dihedral else 0.0

        area_log = np.log10(max(face_area, 1e-6))
        area_ratio = face_area / max(total_area, 1e-6)
        n_tris_log = np.log10(max(n_tris_face, 1))
        vert_density_log = np.log10(max(n_verts_face / max(face_area, 1e-6), 1e-6))
        n_verts_log = np.log10(max(n_verts_face, 1))

        if n_verts_face >= 3:
            fv = np.array([vertices[v] for v in face_verts_set])
            bs = fv.max(axis=0) - fv.min(axis=0)
            bs = bs / max(bs.max(), 1e-6)
        else:
            bs = np.array([1., 1., 1.])

        na_mean = na_std = na_max = na_range = 0.0
        cov_eig1 = cov_eig2 = cov_eig3 = cov_sum_log = 0.0
        if n_tris_face >= 2:
            fn_cov = np.cov(face_normals.T)
            eigvals = np.linalg.eigvalsh(fn_cov) if len(face_normals) >= 3 else np.zeros(3)
            if len(eigvals) >= 3:
                cov_eig1, cov_eig2, cov_eig3 = float(eigvals[2]), float(eigvals[1]), float(eigvals[0])
            cov_sum_log = np.log10(max(cov_eig1+cov_eig2+cov_eig3, 1e-12))
            angles = np.arccos(np.clip(np.dot(face_normals, mean_normal), -1, 1))*180/np.pi
            na_mean = float(angles.mean()); na_std = float(angles.std())
            na_max = float(angles.max()); na_range = float(angles.max()-angles.min())

        features[pid] = [
            area_log, mean_normal[0], mean_normal[1], mean_normal[2],
            normal_std, rel_center[0], rel_center[1], rel_center[2],
            n_tris_log, float(n_nbrs), bs[0], bs[1], bs[2],
            vert_density_log, area_ratio, dih_mean, dih_std, n_verts_log,
            na_mean, na_std, na_max, na_range,
            cov_eig1, cov_eig2, cov_eig3, cov_sum_log,
        ]

        for nb in nbrs:
            edge_index[0].append(pid)
            edge_index[1].append(nb)

    return features, np.array(edge_index, dtype=np.int64), patch_faces
