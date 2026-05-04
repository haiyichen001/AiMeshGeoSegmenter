"""
答案库: STEP -> 统一三角网格 + per-triangle 面标签 (含 fillet/chamfer/sphere 检测)
"""
import os, sys, json, time, random, collections, numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

ROOT = Path(r"D:\AiMeshGeoSegmenter")
STEP_DIR = ROOT / "data" / "step"
ANSWERS_DIR = ROOT / "answers"
os.makedirs(ANSWERS_DIR, exist_ok=True)

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
L2I = {n: i for i, n in enumerate(LABEL_NAMES)}
OCC2BASE = {
    GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus",
    GeomAbs_BezierSurface: "freeform", GeomAbs_BSplineSurface: "freeform",
    GeomAbs_SurfaceOfRevolution: "freeform", GeomAbs_SurfaceOfExtrusion: "freeform",
    GeomAbs_OtherSurface: "freeform",
}
AREA_THRESH = 0.15; PLANE_THRESH = 0.05


def face_area(face):
    loc = TopLoc_Location(); tri = BRep_Tool().Triangulation(face, loc)
    if tri is None: return 0.0
    trsf = loc.Transformation(); area = 0.0
    for i in range(1, tri.NbTriangles()+1):
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
    out = ANSWERS_DIR / f"{stem}.npz"
    if out.exists(): return (stem, "skip", None)

    try:
        sr = STEPControl_Reader()
        if sr.ReadFile(str(step_path)) != IFSelect_RetDone: return (stem, "read_fail", None)
        sr.TransferRoots(); shape = sr.OneShape()
        bbox = Bnd_Box(); brepbndlib.Add(shape, bbox)
        x1,y1,z1,x2,y2,z2 = bbox.Get()
        diag = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
        BRepMesh_IncrementalMesh(shape, diag/100.0).Perform()

        fm = TopTools_IndexedMapOfShape()
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More(): fm.Add(exp.Current()); exp.Next()
        nf = fm.Size()
        if nf < 3: return (stem, {"_few_faces":1})

        # Neighbors + face data
        neighbors = {i: set() for i in range(1, nf+1)}
        exp_e = TopExp_Explorer(shape, TopAbs_EDGE)
        while exp_e.More():
            e = exp_e.Current(); ef = []
            for i in range(1, nf+1):
                fe = TopExp_Explorer(fm.FindKey(i), TopAbs_EDGE)
                while fe.More():
                    if fe.Current().IsSame(e): ef.append(i); break
                    fe.Next()
            for a in ef:
                for b in ef:
                    if a != b: neighbors[a].add(b); neighbors[b].add(a)
            exp_e.Next()

        face_areas = {}; face_occ = {}
        for i in range(1, nf+1):
            adapt = BRepAdaptor_Surface(fm.FindKey(i), True)
            face_occ[i] = adapt.GetType()
            face_areas[i] = face_area(fm.FindKey(i))

        # Classify faces (same logic as refine_labels.py)
        face_labels = {}
        for i in range(1, nf+1):
            occ = face_occ[i]; area = face_areas[i]; nbrs = neighbors[i]
            nn = len(nbrs); base = OCC2BASE.get(occ, "freeform")
            nbr_area = sum(face_areas.get(n,0) for n in nbrs)
            ar = area / max(area + nbr_area, 1e-6)
            radius = 0; cone_half = 0
            if occ == GeomAbs_Cylinder:
                try: radius = adapt.Cylinder().Radius()
                except: pass
            elif occ == GeomAbs_Torus:
                try: radius = adapt.Torus().MinorRadius()
                except: pass
            elif occ == GeomAbs_Cone:
                try:
                    cone = adapt.Cone(); sa = cone.SemiAngle()
                    v1 = adapt.FirstVParameter(); v2 = adapt.LastVParameter()
                    cone_half = abs(v2-v1)*np.sin(float(sa))/2.0
                except: pass

            # Fillet
            if occ in (GeomAbs_Cylinder, GeomAbs_Torus) and nn == 2 and ar < AREA_THRESH:
                if radius > 0 and radius < diag * 0.05: base = "fillet"
            # Chamfer (Cone)
            if occ == GeomAbs_Cone and nn == 2 and ar < AREA_THRESH:
                if cone_half > 0 and cone_half < diag * 0.05: base = "chamfer"
            # Chamfer (Plane): skip, unreliable

            face_labels[i] = L2I[base]

        # Unified mesh
        all_verts, all_tris, all_fids = [], [], []
        for i in range(1, nf+1):
            loc = TopLoc_Location()
            tri = BRep_Tool().Triangulation(fm.FindKey(i), loc)
            if tri is None: continue
            trsf = loc.Transformation(); nv = tri.NbNodes(); nt = tri.NbTriangles()
            lidx = {}
            for j in range(1, nv+1):
                p = tri.Node(j); p.Transform(trsf)
                all_verts.append([p.X(), p.Y(), p.Z()]); lidx[j] = len(all_verts)-1
            for j in range(1, nt+1):
                t = tri.Triangle(j)
                all_tris.append([lidx[t.Value(1)], lidx[t.Value(2)], lidx[t.Value(3)]])
                all_fids.append(i)

        va = np.array(all_verts, dtype=np.float32); ta = np.array(all_tris, dtype=np.int32)
        fa = np.array(all_fids, dtype=np.int32)
        uniq, inv = np.unique(va, axis=0, return_inverse=True)
        tu = inv[ta.flatten()].reshape(-1,3)
        ft = np.array([face_labels[i] for i in range(1, nf+1)], dtype=np.int32)

        np.savez_compressed(out, vertices=uniq, triangles=tu, tri_face_ids=fa, face_types=ft, label_names=LABEL_NAMES)
        return (stem, dict(collections.Counter([LABEL_NAMES[face_labels[i]] for i in range(1, nf+1)])))
    except Exception as e:
        return (stem, {"_err": str(e)[:100]})


if __name__ == "__main__":
    all_files = sorted(STEP_DIR.glob("*.step"))
    random.seed(42); samples = random.sample(all_files, 1000)
    n = max(1, cpu_count()-1)
    print(f"Building answers for {len(samples)} parts, Workers: {n}")
    ok = fail = skip = 0; stats = collections.Counter(); t0 = time.time()
    todo = [str(f) for f in samples if not (ANSWERS_DIR / f"{f.stem}.npz").exists()]
    with Pool(n) as p:
        for stem, s in p.imap_unordered(process_one, todo, chunksize=10):
            if s is None: skip += 1
            elif "_err" in s or "_few" in s: fail += 1
            else: ok += 1; stats.update(s)
            if (ok+fail) % 100 == 0: print(f"  ok={ok} fail={fail}")
    print(f"\nDONE: ok={ok} fail={fail} in {time.time()-t0:.0f}s")
    total = sum(stats.values())
    for name in LABEL_NAMES: print(f"  {name:12s} {stats.get(name,0):8d} ({stats.get(name,0)/max(total,1)*100:5.1f}%)")
