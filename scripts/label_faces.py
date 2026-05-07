"""
STEP -> Per-Face Type Labels (9-class taxonomy)

从 STEP B-Rep 提取每个面的 OCCT 类型 + 几何特征 + 邻接关系。
OCCT 基础类型 -> 9类映射 (Fillet/Chamfer/Gear 需后处理细化)。

Usage:
    python scripts/label_faces.py                    # 处理全部
    python scripts/label_faces.py --part HexagonNut  # 单个测试
"""
import os, sys, json, collections, time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import math

ROOT = Path(__file__).parent.parent
STEP_DIR = ROOT / "data" / "step"
LABEL_DIR = ROOT / "data" / "labels"
os.makedirs(LABEL_DIR, exist_ok=True)

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, GeomAbs_OtherSurface
)
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.gp import gp_Pnt, gp_Vec

# OCCT base type -> our label (initial pass)
TYPE_MAP = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BezierSurface: "freeform",
    GeomAbs_BSplineSurface: "freeform",
    GeomAbs_SurfaceOfRevolution: "freeform",
    GeomAbs_SurfaceOfExtrusion: "freeform",
    GeomAbs_OtherSurface: "freeform",
}

OCCT_NAMES = {
    GeomAbs_Plane: "Plane",
    GeomAbs_Cylinder: "Cylinder",
    GeomAbs_Cone: "Cone",
    GeomAbs_Sphere: "Sphere",
    GeomAbs_Torus: "Torus",
    GeomAbs_BezierSurface: "Bezier",
    GeomAbs_BSplineSurface: "BSpline",
    GeomAbs_SurfaceOfRevolution: "Revolution",
    GeomAbs_SurfaceOfExtrusion: "Extrusion",
    GeomAbs_OtherSurface: "Other",
}

RADIUS_RATIO = 0.10
SPHERE_FIT_TOL = 0.005
PLANE_FIT_TOL = 1e-4  # relative RMS error for plane fitting


def face_norm_from_verts(verts):
    """Compute face normal from first 3 vertices."""
    if len(verts) < 9:
        return None
    p0 = np.array(verts[0:3]); p1 = np.array(verts[3:6]); p2 = np.array(verts[6:9])
    n = np.cross(p1 - p0, p2 - p0)
    nr = np.linalg.norm(n)
    return n / nr if nr > 1e-12 else np.array([0, 0, 1])


def get_face_direction(f):
    """Representative direction: axis for curved faces, vertex normal for planar."""
    axis = f.get("axis")
    if axis is not None and any(a != 0 for a in axis):
        return np.array(axis)
    return face_norm_from_verts(f.get("vertices", []))


def angle_between_faces(f1, f2):
    """Dihedral angle in degrees between faces. Uses axis for curved faces."""
    d1 = get_face_direction(f1)
    d2 = get_face_direction(f2)
    if d1 is None or d2 is None:
        return None

    has_axis1 = f1.get("axis") is not None and any(a != 0 for a in f1["axis"])
    has_axis2 = f2.get("axis") is not None and any(a != 0 for a in f2["axis"])

    if has_axis1 and has_axis2:
        # Both curved: angle between axes
        dot = min(1.0, max(0.0, abs(np.dot(d1, d2))))
        return float(np.arccos(dot) * 180 / np.pi)

    if has_axis1 or has_axis2:
        # One curved + one planar: face angle = |90 - angle(plane_normal, axis)|
        axis = d1 if has_axis1 else d2
        plane_n = d2 if has_axis1 else d1
        cos_a = min(1.0, max(0.0, abs(np.dot(plane_n, axis))))
        alpha = float(np.arccos(cos_a) * 180 / np.pi)
        return abs(90.0 - alpha)

    # Both planar: standard normal angle
    dot = min(1.0, max(0.0, abs(np.dot(d1, d2))))
    return float(np.arccos(dot) * 180 / np.pi)


def try_fit_plane(verts_list):
    """Check if vertices lie on a plane. Returns (ok, normal, rms_error/span)."""
    if len(verts_list) < 9:
        return False, None, 1.0
    pts = np.array(verts_list).reshape(-1, 3)
    if len(pts) > 500:
        idx = np.random.choice(len(pts), 500, replace=False)
        pts = pts[idx]
    c = pts.mean(axis=0)
    u, s, vh = np.linalg.svd(pts - c)
    normal = vh[2]  # smallest singular vector = plane normal
    dists = np.abs(np.dot(pts - c, normal))
    rms = np.sqrt((dists**2).mean())
    span = float(np.sqrt(((pts.max(axis=0) - pts.min(axis=0))**2).sum()))
    rel_err = rms / max(span, 1e-6)
    return True, normal.tolist(), float(rel_err)


def try_fit_sphere(verts_list, tri_list=None):
    """Least-squares sphere fit. Subdivides if < 100 vertices to avoid overfitting."""
    if len(verts_list) < 30:
        return False, None, None, 1.0
    pts = np.array(verts_list).reshape(-1, 3)
    # If too few vertices, subdivide triangle mesh to get more samples
    if len(pts) < 100 and tri_list is not None:
        tris = np.array(tri_list).reshape(-1, 3)
        for _ in range(3):  # 3 iterations of subdivision
            new_pts = list(pts)
            new_tris = []
            edge_mid = {}
            for t in tris:
                new_v = []
                for j in range(3):
                    a, b = int(t[j]), int(t[(j+1)%3])
                    key = tuple(sorted([a, b]))
                    if key not in edge_mid:
                        mid = (pts[a] + pts[b]) / 2
                        edge_mid[key] = len(new_pts)
                        new_pts.append(mid)
                    new_v.append(edge_mid[key])
                new_tris.append([t[0], new_v[0], new_v[2]])
                new_tris.append([new_v[0], t[1], new_v[1]])
                new_tris.append([new_v[2], new_v[1], t[2]])
                new_tris.append([new_v[0], new_v[1], new_v[2]])
            pts = np.array(new_pts)
            tris = np.array(new_tris)
            if len(pts) >= 500: break
    if len(pts) > 500:
        idx = np.random.choice(len(pts), 500, replace=False)
        pts = pts[idx]
    A = np.column_stack([2*pts, np.ones(len(pts))])
    b = (pts**2).sum(axis=1)
    try:
        x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        c = x[:3]; r2 = x[3] + np.dot(c, c)
        if r2 <= 0:
            return False, None, None, 1.0
        r = np.sqrt(r2)
        dists = np.abs(np.linalg.norm(pts - c, axis=1) - r)
        rel = np.sqrt((dists**2).mean()) / max(r, 1e-6)
        return True, c.tolist(), float(r), float(rel)
    except:
        return False, None, None, 1.0


def face_area(face):
    """Compute approximate face area from its triangulation."""
    loc = TopLoc_Location()
    tri = BRep_Tool().Triangulation(face, loc)
    if tri is None:
        return 0.0
    trsf = loc.Transformation()
    area = 0.0
    for i in range(1, tri.NbTriangles() + 1):
        t = tri.Triangle(i)
        p1 = tri.Node(t.Value(1)); p1.Transform(trsf)
        p2 = tri.Node(t.Value(2)); p2.Transform(trsf)
        p3 = tri.Node(t.Value(3)); p3.Transform(trsf)
        v1 = np.array([p2.X()-p1.X(), p2.Y()-p1.Y(), p2.Z()-p1.Z()])
        v2 = np.array([p3.X()-p1.X(), p3.Y()-p1.Y(), p3.Z()-p1.Z()])
        area += 0.5 * np.linalg.norm(np.cross(v1, v2))
    return area


def process_one(step_path):
    stem = Path(step_path).stem
    out_path = os.path.join(LABEL_DIR, f"{stem}.json")
    if os.path.exists(out_path):
        return (stem, "skip", None)

    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
            return (stem, "read_fail", None)
        reader.TransferRoots()
        shape = reader.OneShape()

        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        x1, y1, z1, x2, y2, z2 = bbox.Get()
        span = max(x2 - x1, y2 - y1, z2 - z1)

        BRepMesh_IncrementalMesh(shape, span / 100.0).Perform()

        # Build face index map for adjacency
        face_map = TopTools_IndexedMapOfShape()
        exp_faces = TopExp_Explorer(shape, TopAbs_FACE)
        while exp_faces.More():
            face_map.Add(exp_faces.Current())
            exp_faces.Next()

        # Build face -> neighbor map via shared edges
        face_neighbors = {i: set() for i in range(1, face_map.Size() + 1)}
        edge_exp = TopExp_Explorer(shape, TopAbs_EDGE)
        while edge_exp.More():
            edge = edge_exp.Current()
            edge_faces = []
            for i in range(1, face_map.Size() + 1):
                face = face_map.FindKey(i)
                f_exp = TopExp_Explorer(face, TopAbs_EDGE)
                while f_exp.More():
                    if f_exp.Current().IsSame(edge):
                        edge_faces.append(i)
                        break
                    f_exp.Next()
            for a in edge_faces:
                for b in edge_faces:
                    if a != b:
                        face_neighbors[a].add(b)
                        face_neighbors[b].add(a)
            edge_exp.Next()

        faces_data = []
        occ_types = collections.Counter()

        for i in range(1, face_map.Size() + 1):
            face = face_map.FindKey(i)
            adapt = BRepAdaptor_Surface(face, True)
            st = adapt.GetType()
            loc = TopLoc_Location()
            tri = BRep_Tool().Triangulation(face, loc)
            if tri is None:
                continue

            trsf = loc.Transformation()
            nv = tri.NbNodes()
            nt = tri.NbTriangles()
            verts = []
            tris_idx = []
            vi = 0
            local_idx = {}
            for j in range(1, nv + 1):
                p = tri.Node(j)
                p.Transform(trsf)
                verts.extend([p.X(), p.Y(), p.Z()])
                local_idx[j] = vi
                vi += 1
            for j in range(1, nt + 1):
                t = tri.Triangle(j)
                tris_idx.extend([
                    local_idx[t.Value(1)],
                    local_idx[t.Value(2)],
                    local_idx[t.Value(3)]
                ])

            cx = sum(verts[0::3]) / vi
            cy = sum(verts[1::3]) / vi
            cz = sum(verts[2::3]) / vi
            area = face_area(face)

            # Extract geometric parameters for fillet/chamfer detection
            radius = 0.0
            cone_half_len = 0.0
            axis_dir = [0.0, 0.0, 0.0]
            if st == GeomAbs_Cylinder:
                try:
                    cyl = adapt.Cylinder()
                    radius = cyl.Radius()
                    ax = cyl.Position().Axis().Direction()
                    axis_dir = [ax.X(), ax.Y(), ax.Z()]
                except: pass
            elif st == GeomAbs_Torus:
                try:
                    tor = adapt.Torus()
                    radius = tor.MinorRadius()
                    ax = tor.Position().Axis().Direction()
                    axis_dir = [ax.X(), ax.Y(), ax.Z()]
                except: pass
            elif st == GeomAbs_Cone:
                try:
                    cone = adapt.Cone()
                    ax = cone.Position().Axis().Direction()
                    axis_dir = [ax.X(), ax.Y(), ax.Z()]
                except: pass
            # Check if small cylinder is a full circle (hole) vs partial arc (fillet)
            arc_deg = 360.0
            if st in (GeomAbs_Cylinder, GeomAbs_Torus):
                try:
                    u1 = adapt.FirstUParameter(); u2 = adapt.LastUParameter()
                    arc_deg = abs(u2 - u1) * 180 / np.pi
                except: pass

            elif st == GeomAbs_Cone:
                try:
                    cone = adapt.Cone()
                    sa = cone.SemiAngle()
                    v1 = adapt.FirstVParameter()
                    v2 = adapt.LastVParameter()
                    cone_half_len = abs(v2 - v1) * np.sin(float(sa)) / 2.0
                except: pass

            occ_type_name = OCCT_NAMES.get(st, "Other")
            label = TYPE_MAP.get(st, "freeform")
            occ_types[occ_type_name] += 1

            # Convexity via exact surface normal (D1 at face midpoint)
            is_convex = True
            try:
                u = (adapt.FirstUParameter() + adapt.LastUParameter()) / 2.0
                v = (adapt.FirstVParameter() + adapt.LastVParameter()) / 2.0
                pt = gp_Pnt(); d1u = gp_Vec(); d1v = gp_Vec()
                adapt.D1(u, v, pt, d1u, d1v)
                surf_n = np.cross(
                    [d1u.X(), d1u.Y(), d1u.Z()],
                    [d1v.X(), d1v.Y(), d1v.Z()])
                nr = np.linalg.norm(surf_n)
                surf_n = surf_n / nr if nr > 1e-12 else np.array([0, 0, 1])

                if st in (GeomAbs_Cylinder, GeomAbs_Torus):
                    ax = adapt.Cylinder() if st == GeomAbs_Cylinder else adapt.Torus()
                    axis = ax.Position().Axis()
                    o = np.array([axis.Location().X(), axis.Location().Y(), axis.Location().Z()])
                    d = np.array([axis.Direction().X(), axis.Direction().Y(), axis.Direction().Z()])
                    fc = np.array([cx, cy, cz])
                    proj = o + np.dot(fc - o, d) * d
                    radial = fc - proj
                    nr_radial = np.linalg.norm(radial)
                    if nr_radial > 1e-6:
                        radial = radial / nr_radial
                        is_convex = bool(np.dot(surf_n, radial) > 0)
                elif st == GeomAbs_Cone:
                    apex = adapt.Cone().Apex()
                    fc = np.array([cx, cy, cz])
                    to_apex = np.array([apex.X() - cx, apex.Y() - cy, apex.Z() - cz])
                    da = np.linalg.norm(to_apex)
                    if da > 1e-6:
                        is_convex = bool(np.dot(surf_n, to_apex / da) > 0)
                elif st == GeomAbs_Plane:
                    fc = np.array([cx, cy, cz]); off = span * 0.001
                    clf = BRepClass3d_SolidClassifier(shape)
                    clf.Perform(gp_Pnt(fc[0] + surf_n[0]*off, fc[1] + surf_n[1]*off, fc[2] + surf_n[2]*off), 1e-4)
                    is_convex = bool(clf.State() != 3)
            except:
                pass

            faces_data.append({
                "id": i,
                "vertices": verts,
                "triangles": tris_idx,
                "center": [cx, cy, cz],
                "area": round(area, 6),
                "occ_type": occ_type_name,
                "label": label,
                "neighbors": sorted(face_neighbors.get(i, [])),
                "smooth_edges": 0,
                "total_edges": len(face_neighbors.get(i, [])),
                "radius": round(radius, 4),
                "cone_half_len": round(cone_half_len, 4),
                "arc_deg": round(arc_deg, 1),
                "is_convex": is_convex,
                "axis": [round(x, 6) for x in axis_dir],
            })

        diagonal = float(np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2))

        # --- Recovery: detect planes then spheres from freeform faces ---
        label_dist = collections.Counter()
        for f in faces_data:
            occ = f["occ_type"]
            if f["label"] == "freeform" or occ in ("Bezier", "BSpline", "Revolution", "Extrusion", "Other"):
                verts = f.get("vertices", [])

                # 1. Check if actually planar
                ok_pl, n_pl, rel_pl = try_fit_plane(verts)
                if ok_pl and rel_pl < PLANE_FIT_TOL:
                    f["label"] = "plane"; label_dist["plane"] += 1; continue

                # 2. Check if actually spherical (skip Extrusion - can never be a sphere)
                if occ != "Extrusion":
                    ok_sp, c_sp, r_sp, rel_sp = try_fit_sphere(verts, f.get("triangles"))
                    if ok_sp and rel_sp < SPHERE_FIT_TOL:
                        f["label"] = "sphere"; label_dist["sphere"] += 1; continue

                f["label"] = "freeform"; label_dist["freeform"] += 1
            else:
                label_dist[f["label"]] += 1

        with open(out_path, 'w') as f:
            json.dump({
                "part": stem,
                "num_faces": len(faces_data),
                "center": [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2],
                "span": span,
                "diagonal": diagonal,
                "occ_distribution": dict(occ_types),
                "label_distribution": dict(label_dist),
                "faces": faces_data,
            }, f)

        return (stem, "ok", dict(occ_types))

    except Exception as e:
        return (stem, f"err:{e}", None)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default=None)
    args = parser.parse_args()

    if args.part:
        step_file = os.path.join(STEP_DIR, f"{args.part}.step")
        stem, status, _ = process_one(step_file)
        print(f"{stem}: {status}")
    else:
        files = sorted(Path(STEP_DIR).glob("*.step"))
        todo = [str(f) for f in files
                if not os.path.exists(os.path.join(LABEL_DIR, f"{f.stem}.json"))]
        n = max(1, cpu_count() - 1)
        print(f"Total: {len(files)}, To do: {len(todo)}, Workers: {n}")

        ok = fail = skip = 0
        global_stats = collections.Counter()
        global_faces = 0
        t0 = time.time()

        with Pool(n) as p:
            for stem, status, occ in p.imap_unordered(process_one, todo, chunksize=10):
                if status == "ok":
                    ok += 1
                    if occ:
                        global_stats.update(occ)
                        global_faces += sum(occ.values())
                elif status == "skip":
                    skip += 1
                else:
                    fail += 1
                total = ok + fail
                if total % 200 == 0:
                    elapsed = time.time() - t0
                    rate = total / elapsed if elapsed > 0 else 0
                    eta = (len(todo) - total) / rate / 60 if rate > 0 else 0
                    print(f"  [{total}/{len(todo)}] ok={ok} fail={fail} skip={skip} | {rate:.1f} p/s | ETA {eta:.0f}min")

        elapsed = time.time() - t0
        print(f"\nDONE: ok={ok} fail={fail} skip={skip} in {elapsed/60:.1f}min")
        print(f"Label files: {len(list(Path(LABEL_DIR).glob('*.json')))}")

        # Stats summary
        print(f"\n===== Type Distribution (OCCT base, {global_faces} faces from {ok} parts) =====")
        for t, c in global_stats.most_common():
            pct = c / global_faces * 100 if global_faces > 0 else 0
            print(f"  {t:15s}  {c:8d}  ({pct:5.1f}%)")

        # 9-class summary (initial mapping)
        label_map = {
            "Plane": "plane", "Cylinder": "cylinder", "Sphere": "sphere",
            "Cone": "cone", "Torus": "torus",
            "Bezier": "freeform", "BSpline": "freeform",
            "Revolution": "freeform", "Extrusion": "freeform", "Other": "freeform",
        }
        label_stats = collections.Counter()
        for t, c in global_stats.items():
            label_stats[label_map.get(t, "freeform")] += c
        print(f"\n===== 9-Class Summary (initial, fillet/chamfer/gear need post-processing) =====")
        for name in ["plane", "cylinder", "sphere", "cone", "torus", "freeform"]:
            c = label_stats.get(name, 0)
            pct = c / global_faces * 100 if global_faces > 0 else 0
            print(f"  {name:15s}  {c:8d}  ({pct:5.1f}%)")

        # Save stats file
        stats_path = LABEL_DIR / "distribution.json"
        with open(stats_path, 'w') as f:
            json.dump({
                "total_parts": ok,
                "total_faces": global_faces,
                "occ_distribution": dict(global_stats),
                "label_distribution": dict(label_stats),
            }, f, indent=2)
        print(f"\nStats saved to {stats_path}")
