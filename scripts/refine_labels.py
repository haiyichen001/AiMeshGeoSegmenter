"""
后处理: 半径/锥长加权评分检测 Fillet / Chamfer

Fillet  = 0.5*radius_score + 0.25*nbr_score + 0.25*area_score
Chamfer = 0.5*len_score   + 0.25*nbr_score + 0.25*area_score  (Cone)
        = 0.5*angle_score + 0.25*nbr_score + 0.25*area_score  (Plane)
"""
import os, json, collections, time, numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

ROOT = Path(r"D:\AiMeshGeoSegmenter")
LABEL_DIR = ROOT / "data" / "labels"

AREA_RATIO_THRESH = 0.15       # fillet + cone chamfer
PLANE_CHAMFER_RATIO = 0.05     # planar chamfer (stricter)
SPHERE_FIT_TOL = 0.02          # relative RMS error for sphere fitting


def faces_angle(f1, f2):
    verts1 = f1.get("vertices", [])
    verts2 = f2.get("vertices", [])
    if len(verts1) < 9 or len(verts2) < 9:
        return 0
    p0 = np.array(verts1[0:3]); p1 = np.array(verts1[3:6]); p2 = np.array(verts1[6:9])
    n1 = np.cross(p1 - p0, p2 - p0)
    nr = np.linalg.norm(n1); n1 = n1 / nr if nr > 1e-12 else np.array([0,0,1])
    p0 = np.array(verts2[0:3]); p1 = np.array(verts2[3:6]); p2 = np.array(verts2[6:9])
    n2 = np.cross(p1 - p0, p2 - p0)
    nr = np.linalg.norm(n2); n2 = n2 / nr if nr > 1e-12 else np.array([0,0,1])
    dot = min(1.0, max(0.0, abs(np.dot(n1, n2))))
    return float(np.arccos(dot) * 180 / np.pi)


def try_fit_sphere(verts_list):
    """Try to fit a sphere to face vertices. Returns (success, center, radius, rms_error)."""
    if len(verts_list) < 30:  # need enough points
        return False, None, None, 1.0
    pts = np.array(verts_list).reshape(-1, 3)
    # Sub-sample for speed
    if len(pts) > 500:
        idx = np.random.choice(len(pts), 500, replace=False)
        pts = pts[idx]
    # Least-squares sphere: |p-c|^2 = r^2  =>  2c·p + (r^2 - |c|^2) = |p|^2
    A = np.column_stack([2*pts, np.ones(len(pts))])
    b = (pts**2).sum(axis=1)
    try:
        x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        c = x[:3]
        r2 = x[3] + np.dot(c, c)
        if r2 <= 0:
            return False, None, None, 1.0
        r = np.sqrt(r2)
        dists = np.abs(np.linalg.norm(pts - c, axis=1) - r)
        rms = np.sqrt((dists**2).mean())
        rel_err = rms / max(r, 1e-6)
        return True, c.tolist(), float(r), float(rel_err)
    except:
        return False, None, None, 1.0


def process_one(label_file):
    path = os.path.join(LABEL_DIR, label_file)
    with open(path) as f:
        data = json.load(f)
    faces = data["faces"]
    if not faces:
        return (label_file, {"_empty": 1})
    diagonal = data.get("diagonal", 1.0)
    face_by_id = {f["id"]: f for f in faces}
    stats = collections.Counter()

    for f in faces:
        occ = f["occ_type"]
        nbrs = f.get("neighbors", [])
        n_nbrs = len(nbrs)
        area = f.get("area", 0)
        nbr_area = sum(face_by_id.get(n, {}).get("area", 0) for n in nbrs)
        total = area + nbr_area
        area_ratio = area / total if total > 0 else 1.0
        radius = f.get("radius", 0)
        cone_half = f.get("cone_half_len", 0)

        # --- Fillet: 0.5*radius + 0.25*nbr + 0.25*area ---
        if occ in ("Cylinder", "Torus") and n_nbrs == 2:
            r_score = 1.0 if radius > 0 and radius < diagonal * 0.05 else 0.0
            n_score = 1.0
            a_score = 1.0 if area_ratio < AREA_RATIO_THRESH else 0.0
            score = 0.5 * r_score + 0.25 * n_score + 0.25 * a_score
            if score > 0.5:
                f["label"] = "fillet"; stats["fillet"] += 1; continue

        # --- Chamfer (Cone): 0.5*len + 0.25*nbr + 0.25*area ---
        if occ == "Cone" and n_nbrs == 2:
            l_score = 1.0 if cone_half > 0 and cone_half < diagonal * 0.05 else 0.0
            n_score = 1.0
            a_score = 1.0 if area_ratio < AREA_RATIO_THRESH else 0.0
            score = 0.5 * l_score + 0.25 * n_score + 0.25 * a_score
            if score > 0.5:
                f["label"] = "chamfer"; stats["chamfer"] += 1; continue

        # --- Chamfer (Plane): 0.5*angle + 0.25*nbr + 0.25*area ---
        if occ == "Plane" and n_nbrs >= 2:
            angles = [faces_angle(f, face_by_id.get(n, {})) for n in nbrs]
            good = [a for a in angles if a > 15]
            ang_score = len(good) / max(len(angles), 1) if angles else 0
            n_score = 1.0 if n_nbrs == 2 else 0.5
            a_score = 1.0 if area_ratio < PLANE_CHAMFER_RATIO else 0.0
            score = 0.5 * ang_score + 0.25 * n_score + 0.25 * a_score
            if score > 0.5:
                f["label"] = "chamfer"; stats["chamfer"] += 1; continue

        # --- Sphere recovery: fit sphere to freeform faces ---
        if f["label"] == "freeform" or occ in ("Bezier", "BSpline", "Revolution", "Extrusion", "Other"):
            verts = f.get("vertices", [])
            ok, c, r, rel_err = try_fit_sphere(verts)
            if ok and rel_err < SPHERE_FIT_TOL:
                f["label"] = "sphere"; stats["sphere"] += 1; continue

        # --- Default ---
        if occ in ("Bezier", "BSpline", "Revolution", "Extrusion", "Other"):
            f["label"] = "freeform"; stats["freeform"] += 1
        elif occ == "Plane": f["label"] = "plane"; stats["plane"] += 1
        elif occ == "Cylinder": f["label"] = "cylinder"; stats["cylinder"] += 1
        elif occ == "Sphere": f["label"] = "sphere"; stats["sphere"] += 1
        elif occ == "Cone": f["label"] = "cone"; stats["cone"] += 1
        elif occ == "Torus": f["label"] = "torus"; stats["torus"] += 1
        else: f["label"] = "freeform"; stats["freeform"] += 1

    data["label_distribution"] = dict(stats)
    with open(path, 'w') as f:
        json.dump(data, f)
    return (label_file, dict(stats))


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(LABEL_DIR) if f.endswith('.json') and f != 'distribution.json')
    n = max(1, cpu_count() - 1)
    print(f"Files: {len(files)}, Workers: {n}")
    ok = 0; global_stats = collections.Counter(); t0 = time.time()
    with Pool(n) as p:
        for name, stats in p.imap_unordered(process_one, files, chunksize=20):
            ok += 1; global_stats.update(stats)
            if ok % 2000 == 0: print(f"  ok={ok}")
    total = sum(global_stats.values())
    print(f"\nDONE: ok={ok} in {time.time()-t0:.0f}s, {total} faces")
    order = ["plane", "cylinder", "sphere", "cone", "torus", "fillet", "chamfer", "freeform"]
    for name in order:
        c = global_stats.get(name, 0)
        print(f"  {name:12s} {c:8d} ({c/max(total,1)*100:5.1f}%)")
