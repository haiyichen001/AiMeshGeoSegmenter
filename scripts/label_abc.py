"""
ABC 数据集: STEP -> 图特征 (跳过大JSON)

直接读取 STEP B-Rep -> 提取面类型/面积/邻接/法向 -> 保存 compact NPZ
"""
import os, sys, re, time, collections, tempfile, shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np

ROOT = Path(r"D:\AiMeshGeoSegmenter")
ABC_STEP_DIR = ROOT / "data" / "abc_step"
GRAPH_DIR = ROOT / "data" / "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
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

LABEL_NAMES = ["plane", "cylinder", "sphere", "cone", "torus", "fillet", "chamfer", "freeform"]
LABEL_TO_IDX = {n: i for i, n in enumerate(LABEL_NAMES)}

OCC_TO_BASE = {
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

AREA_RATIO_THRESH = 0.15
CHAMFER_CONE_AREA_MAX = 500.0


def face_area(face):
    loc = TopLoc_Location()
    tri = BRep_Tool().Triangulation(face, loc)
    if tri is None:
        return 0.0, 0
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
    return area, tri.NbNodes()


def face_normal_std(face):
    """Compute normal variance from face triangulation (curvature proxy)."""
    loc = TopLoc_Location()
    tri = BRep_Tool().Triangulation(face, loc)
    if tri is None or tri.NbTriangles() < 1:
        return np.array([0., 0., 1.]), 0.0
    trsf = loc.Transformation()
    norms = []
    for i in range(1, tri.NbTriangles() + 1):
        t = tri.Triangle(i)
        p1 = tri.Node(t.Value(1)); p1.Transform(trsf)
        p2 = tri.Node(t.Value(2)); p2.Transform(trsf)
        p3 = tri.Node(t.Value(3)); p3.Transform(trsf)
        v1 = np.array([p2.X()-p1.X(), p2.Y()-p1.Y(), p2.Z()-p1.Z()])
        v2 = np.array([p3.X()-p1.X(), p3.Y()-p1.Y(), p3.Z()-p1.Z()])
        n = np.cross(v1, v2)
        nr = np.linalg.norm(n)
        if nr > 1e-15:
            norms.append(n / nr)
    if not norms:
        return np.array([0., 0., 1.]), 0.0
    norms = np.array(norms)
    mean_n = norms.mean(axis=0)
    nr = np.linalg.norm(mean_n)
    if nr > 1e-15:
        mean_n /= nr
    return mean_n, float(norms.std(axis=0).mean())


def process_one(step_path_str):
    step_path = Path(step_path_str)
    stem = step_path.stem
    parent = step_path.parent.name  # model_id like "00000002"
    graph_name = f"abc_{parent}_{stem}"
    out_path = GRAPH_DIR / f"{graph_name}.npz"
    if out_path.exists():
        return (graph_name, {"_skip": 1})

    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
            return (graph_name, {"_read_fail": 1})
        reader.TransferRoots()
        shape = reader.OneShape()

        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        x1, y1, z1, x2, y2, z2 = bbox.Get()
        span = max(x2 - x1, y2 - y1, z2 - z1)
        part_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2])

        BRepMesh_IncrementalMesh(shape, span / 100.0).Perform()

        # Face map + neighbors
        face_map = TopTools_IndexedMapOfShape()
        exp_f = TopExp_Explorer(shape, TopAbs_FACE)
        while exp_f.More():
            face_map.Add(exp_f.Current())
            exp_f.Next()
        n_faces = face_map.Size()
        if n_faces < 2:
            return (graph_name, {"_too_few_faces": 1})

        neighbors = {i: set() for i in range(1, n_faces + 1)}
        exp_e = TopExp_Explorer(shape, TopAbs_EDGE)
        while exp_e.More():
            edge = exp_e.Current()
            edge_faces = []
            for i in range(1, n_faces + 1):
                face = face_map.FindKey(i)
                fe = TopExp_Explorer(face, TopAbs_EDGE)
                while fe.More():
                    if fe.Current().IsSame(edge):
                        edge_faces.append(i)
                        break
                    fe.Next()
            for a in edge_faces:
                for b in edge_faces:
                    if a != b:
                        neighbors[a].add(b)
                        neighbors[b].add(a)
            exp_e.Next()

        # Per-face data
        face_areas = {}
        face_types = {}
        face_normals = {}
        face_normal_stds = {}
        total_area = 0.0

        for i in range(1, n_faces + 1):
            face = face_map.FindKey(i)
            adapt = BRepAdaptor_Surface(face, True)
            occ_type = adapt.GetType()
            area, n_verts = face_area(face)
            face_areas[i] = area
            face_types[i] = occ_type
            total_area += area
            mean_n, nstd = face_normal_std(face)
            face_normals[i] = mean_n
            face_normal_stds[i] = nstd

        # Classify (with fillet/chamfer refinement)
        features = np.zeros((n_faces, 18), dtype=np.float32)
        edge_list = [[], []]
        labels = np.zeros(n_faces, dtype=np.int64)

        for i in range(1, n_faces + 1):
            occ_type = face_types[i]
            area = face_areas[i]
            nbrs = sorted(neighbors[i])
            n_nbrs = len(nbrs)
            base_label = OCC_TO_BASE.get(occ_type, "freeform")
            mean_normal = face_normals[i]
            normal_std = face_normal_stds[i]
            center = np.array([0., 0., 0.])  # approximate
            rel_center = (center - part_center) / max(span, 1e-6)

            # Neighbor area
            nbr_area = sum(face_areas.get(n, 0) for n in nbrs)
            total_nbr = area + nbr_area
            area_ratio = area / max(total_nbr, 1e-6)

            # Fillet detection
            if occ_type in (GeomAbs_Cylinder, GeomAbs_Torus) and n_nbrs >= 2 and area_ratio < AREA_RATIO_THRESH:
                base_label = "fillet"

            # Chamfer detection
            if occ_type == GeomAbs_Cone and n_nbrs >= 2 and area_ratio < AREA_RATIO_THRESH and area < CHAMFER_CONE_AREA_MAX:
                base_label = "chamfer"

            # Features
            area_log = np.log10(max(area, 1e-6))
            area_r = area / max(total_area, 1e-6)
            n_tris_log = np.log10(max(n_verts // 3, 1))
            vert_density_log = np.log10(max(n_verts / max(area, 1e-6), 1e-6))
            n_verts_log = np.log10(max(n_verts, 1))

            features[i - 1] = [
                area_log, mean_normal[0], mean_normal[1], mean_normal[2],
                normal_std, rel_center[0], rel_center[1], rel_center[2],
                n_tris_log, float(n_nbrs), 1.0, 1.0, 1.0,
                vert_density_log, area_r, 0.0, 0.0, n_verts_log,
            ]

            # Edges
            for nb in nbrs:
                edge_list[0].append(i - 1)
                edge_list[1].append(nb - 1)

            labels[i - 1] = LABEL_TO_IDX.get(base_label, LABEL_TO_IDX["freeform"])

        np.savez_compressed(out_path,
                            x=features,
                            edge_index=np.array(edge_list, dtype=np.int64),
                            y=labels,
                            part_name=graph_name,
                            num_nodes=n_faces)
        label_dist = collections.Counter([LABEL_NAMES[int(l)] for l in labels])
        return (graph_name, dict(label_dist))

    except Exception as e:
        return (graph_name, {"_err": str(e)[:100]})


if __name__ == "__main__":
    # Find all .step files recursively
    all_steps = list(ABC_STEP_DIR.rglob("*.step"))
    print(f"ABC STEP files found: {len(all_steps)}")

    existing = set(f.stem for f in GRAPH_DIR.glob("abc_*.npz"))
    todo = [str(f) for f in all_steps if f"abc_{f.parent.name}_{f.stem}" not in existing]
    print(f"Already done: {len(all_steps) - len(todo)}, To do: {len(todo)}")

    n = max(1, cpu_count() - 1)
    print(f"Workers: {n}")

    ok, fail, skip = 0, 0, 0
    global_stats = collections.Counter()
    t0 = time.time()

    with Pool(n) as p:
        for name, stats in p.imap_unordered(process_one, todo, chunksize=20):
            if "_err" in stats:
                fail += 1
            elif any(k.startswith("_") for k in stats):
                skip += 1  # _skip, _too_few_faces, _read_fail
            else:
                ok += 1
                global_stats.update(stats)
            total = ok + fail + skip
            if total % 500 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                eta = (len(todo) - total) / rate / 60 if rate > 0 else 0
                print(f"  [{total}/{len(todo)}] ok={ok} fail={fail} | {rate:.1f} p/s | ETA {eta:.0f}min")

    elapsed = time.time() - t0
    print(f"\nDONE: ok={ok} fail={fail} in {elapsed/60:.1f}min")
    total_faces = sum(global_stats.values())
    print(f"Graph files: {ok}")
    print(f"Total faces: {total_faces}")
    print(f"Label distribution:")
    for name in LABEL_NAMES:
        c = global_stats.get(name, 0)
        print(f"  {name:12s} {c:8d} ({c/max(total_faces,1)*100:5.1f}%)")
