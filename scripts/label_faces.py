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
from OCC.Core.gp import gp_Pnt

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
            if st == GeomAbs_Cylinder:
                try:
                    radius = adapt.Cylinder().Radius()
                except: pass
            elif st == GeomAbs_Torus:
                try:
                    radius = adapt.Torus().MinorRadius()
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

            # Convexity: offset face center along normal, check if outside solid
            is_convex = True
            try:
                if len(verts) >= 9:
                    p0 = np.array(verts[0:3]); p1 = np.array(verts[3:6]); p2 = np.array(verts[6:9])
                    fn = np.cross(p1-p0, p2-p0); nr = np.linalg.norm(fn)
                    if nr > 1e-12:
                        fn /= nr
                        off = span * 0.001
                        clf = BRepClass3d_SolidClassifier(shape)
                        clf.Perform(gp_Pnt(cx + fn[0]*off, cy + fn[1]*off, cz + fn[2]*off), 1e-4)
                        is_convex = (clf.State() != 3)  # 3=TopAbs_IN (inside solid)
            except: pass

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
            })

        with open(out_path, 'w') as f:
            json.dump({
                "part": stem,
                "num_faces": len(faces_data),
                "center": [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2],
                "span": span,
                "diagonal": float(np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)),
                "occ_distribution": dict(occ_types),
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
