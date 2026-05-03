"""
后处理: 从基础 OCCT 标签中检测 Fillet / Chamfer

Fillet = 圆柱面/环面 + 恰好2个邻接面 + 小面积 + 平滑过渡
Chamfer = 平面 + 至少2个邻接面 + 小面积 + 非零二面角

输出: 8 类标签 + 分布统计
"""
import os, json, collections
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count
import time

ROOT = Path(r"D:\AiMeshGeoSegmenter")
LABEL_DIR = ROOT / "data" / "labels"
os.makedirs(LABEL_DIR, exist_ok=True)

AREA_RATIO_THRESH = 0.15    # fillet: area / (area + neighbor area) max
PLANE_CHAMFER_RATIO = 0.05  # (unused, kept for reference)
CHAMFER_CONE_AREA_MAX = 500.0  # mm^2, cone chamfer absolute max area


def face_normal(vertices):
    """Estimate face normal from first 3 vertices."""
    if len(vertices) < 9:
        return None
    p0 = np.array(vertices[0:3])
    p1 = np.array(vertices[3:6])
    p2 = np.array(vertices[6:9])
    n = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return None
    return n / norm


def faces_angle(f1, f2):
    """Dihedral angle in degrees between two faces (0=parallel, 90=perpendicular)."""
    n1 = face_normal(f1.get("vertices", []))
    n2 = face_normal(f2.get("vertices", []))
    if n1 is None or n2 is None:
        return 0
    dot = abs(np.dot(n1, n2))
    dot = min(1.0, max(0.0, dot))
    return np.arccos(dot) * 180 / np.pi


def process_one(label_file):
    path = os.path.join(LABEL_DIR, label_file)
    with open(path) as f:
        data = json.load(f)

    faces = data["faces"]
    if not faces:
        return (label_file, "empty", {})

    # Build face lookup by id
    face_by_id = {f["id"]: f for f in faces}

    stats = collections.Counter()
    for f in faces:
        occ = f["occ_type"]
        nbrs = f.get("neighbors", [])
        n_nbrs = len(nbrs)
        area = f.get("area", 0)
        # Compute neighbor total area
        nbr_area = sum(face_by_id.get(n, {}).get("area", 0) for n in nbrs)
        total = area + nbr_area
        area_ratio = area / total if total > 0 else 1.0

        # --- Fillet detection (Cylinder/Torus with 2 neighbors, small, tangent) ---
        if occ in ("Cylinder", "Torus") and n_nbrs >= 2 and area_ratio < AREA_RATIO_THRESH:
            angles = [faces_angle(f, face_by_id.get(n, {})) for n in nbrs]
            angles = [a for a in angles if a >= 0]
            # Fillet connects neighbors via tangent (near-zero dihedral)
            if angles and all(a < 15 for a in angles):
                f["label"] = "fillet"
                stats["fillet"] += 1
                continue

        # --- Chamfer detection (small Cone only — planar chamfers stay as plane) ---
        if occ == "Cone" and n_nbrs >= 2 and area_ratio < AREA_RATIO_THRESH and area < CHAMFER_CONE_AREA_MAX:
            f["label"] = "chamfer"
            stats["chamfer"] += 1
            continue

        # No change — determine label from OCCT type
        if occ in ("Bezier", "BSpline", "Revolution", "Extrusion", "Other"):
            f["label"] = "freeform"
            stats["freeform"] += 1
        elif occ == "Plane":
            f["label"] = "plane"
            stats["plane"] += 1
        elif occ == "Cylinder":
            f["label"] = "cylinder"
            stats["cylinder"] += 1
        elif occ == "Sphere":
            f["label"] = "sphere"
            stats["sphere"] += 1
        elif occ == "Cone":
            f["label"] = "cone"
            stats["cone"] += 1
        elif occ == "Torus":
            f["label"] = "torus"
            stats["torus"] += 1
        else:
            f["label"] = "freeform"
            stats["freeform"] += 1

    # Update JSON
    data["label_distribution"] = dict(stats)
    with open(path, 'w') as f:
        json.dump(data, f)

    return (label_file, "ok", dict(stats))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default=None)
    args = parser.parse_args()

    if args.part:
        fn = f"{args.part}.json"
        name, status, s = process_one(fn)
        print(f"{name}: {status}")
        if s:
            for k, v in sorted(s.items()):
                print(f"  {k}: {v}")
    else:
        files = sorted(f for f in os.listdir(LABEL_DIR)
                       if f.endswith('.json') and f != 'distribution.json')

        n = max(1, cpu_count() - 1)
        print(f"Files: {len(files)}, Workers: {n}")

        global_stats = collections.Counter()
        ok = fail = 0
        t0 = time.time()

        with Pool(n) as p:
            for name, status, s in p.imap_unordered(process_one, files, chunksize=20):
                if status == "ok":
                    ok += 1
                    if s:
                        global_stats.update(s)
                else:
                    fail += 1
                if (ok + fail) % 500 == 0:
                    print(f"  ok={ok} fail={fail}")

        elapsed = time.time() - t0
        total_faces = sum(global_stats.values())
        print(f"\nDONE: ok={ok} fail={fail} in {elapsed:.0f}s")
        print(f"Total faces: {total_faces}")

        print(f"\n===== 8-Class Distribution =====")
        order = ["plane", "cylinder", "sphere", "cone", "torus", "fillet", "chamfer", "freeform"]
        for name in order:
            c = global_stats.get(name, 0)
            pct = c / total_faces * 100 if total_faces > 0 else 0
            print(f"  {name:12s}  {c:8d}  ({pct:5.1f}%)")

        # Save stats
        stats_path = LABEL_DIR / "distribution.json"
        new_data = {
            "total_parts": ok,
            "total_faces": total_faces,
            "distribution": dict(global_stats),
        }
        with open(stats_path, 'w') as f:
            json.dump(new_data, f, indent=2)
        print(f"\nStats saved to {stats_path}")
