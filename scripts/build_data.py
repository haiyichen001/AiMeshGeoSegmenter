"""
构建 GAT 训练数据: STEP 统一三角剖分 -> per-triangle 面标签 (100% 准确, 无 KD 树映射)
"""
import os, time, collections, numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

ROOT = Path(r"D:\AiMeshGeoSegmenter")
STEP_DIR = ROOT / "data" / "step"
DATA_DIR = ROOT / "data" / "patches"
os.makedirs(DATA_DIR, exist_ok=True)

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
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier

LABEL_NAMES = ["plane","cylinder","sphere","cone","torus","fillet","chamfer","freeform"]
L2I = {n:i for i,n in enumerate(LABEL_NAMES)}
O2B = {GeomAbs_Plane:"plane",GeomAbs_Cylinder:"cylinder",GeomAbs_Cone:"cone",
       GeomAbs_Sphere:"sphere",GeomAbs_Torus:"torus",
       GeomAbs_BezierSurface:"freeform",GeomAbs_BSplineSurface:"freeform",
       GeomAbs_SurfaceOfRevolution:"freeform",GeomAbs_SurfaceOfExtrusion:"freeform",
       GeomAbs_OtherSurface:"freeform"}
AREA_THRESH = 0.15


def face_area(face):
    loc = TopLoc_Location(); tri = BRep_Tool().Triangulation(face, loc)
    if tri is None: return 0.0
    trsf = loc.Transformation(); area = 0.0
    for i in range(1, tri.NbTriangles()+1):
        t = tri.Triangle(i); p1=tri.Node(t.Value(1));p1.Transform(trsf)
        p2=tri.Node(t.Value(2));p2.Transform(trsf);p3=tri.Node(t.Value(3));p3.Transform(trsf)
        v1=np.array([p2.X()-p1.X(),p2.Y()-p1.Y(),p2.Z()-p1.Z()])
        v2=np.array([p3.X()-p1.X(),p3.Y()-p1.Y(),p3.Z()-p1.Z()])
        area += 0.5*np.linalg.norm(np.cross(v1,v2))
    return area


def process_one(step_path):
    stem = Path(step_path).stem; out = DATA_DIR / f"{stem}.npz"
    if out.exists(): return (stem, {"_skip":1})

    try:
        sr = STEPControl_Reader()
        if sr.ReadFile(str(step_path)) != IFSelect_RetDone: return (stem, {"_read":1})
        sr.TransferRoots(); shape = sr.OneShape()
        bbox = Bnd_Box(); brepbndlib.Add(shape,bbox)
        x1,y1,z1,x2,y2,z2 = bbox.Get()
        diag = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
        BRepMesh_IncrementalMesh(shape, diag/100.0).Perform()

        fm = TopTools_IndexedMapOfShape()
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More(): fm.Add(exp.Current()); exp.Next()
        nf = fm.Size()
        if nf < 3: return (stem, {"_few":1})

        # Neighbors + face attributes
        neighbors = {i:set() for i in range(1,nf+1)}
        face_areas={}; face_occ={}
        for i in range(1,nf+1):
            adapt = BRepAdaptor_Surface(fm.FindKey(i),True)
            face_occ[i]=adapt.GetType(); face_areas[i]=face_area(fm.FindKey(i))

        exp_e = TopExp_Explorer(shape, TopAbs_EDGE)
        while exp_e.More():
            e=exp_e.Current(); ef=[]
            for i in range(1,nf+1):
                fe=TopExp_Explorer(fm.FindKey(i), TopAbs_EDGE)
                while fe.More():
                    if fe.Current().IsSame(e): ef.append(i); break
                    fe.Next()
            for a in ef:
                for b in ef:
                    if a!=b: neighbors[a].add(b); neighbors[b].add(a)
            exp_e.Next()

        # Classify faces
        face_labels = {}
        for i in range(1,nf+1):
            occ=face_occ[i]; area=face_areas[i]; nbrs=neighbors[i]; nn=len(nbrs)
            base = O2B.get(occ,"freeform")
            nbr_area = sum(face_areas.get(n,0) for n in nbrs)
            ar = area/max(area+nbr_area,1e-6)
            radius=0; cone_half=0
            if occ==GeomAbs_Cylinder:
                try: radius=adapt.Cylinder().Radius()
                except: pass
            elif occ==GeomAbs_Torus:
                try: radius=adapt.Torus().MinorRadius()
                except: pass
            elif occ==GeomAbs_Cone:
                try:
                    cone=adapt.Cone(); sa=cone.SemiAngle()
                    v1=adapt.FirstVParameter(); v2=adapt.LastVParameter()
                    cone_half=abs(v2-v1)*np.sin(float(sa))/2.0
                except: pass
            # Compute neighbor face angle (for fillet/chamfer detection)
            nbr_angle = None
            if nn == 2:
                n1 = list(nbrs)[0]; n2 = list(nbrs)[1]
                # Get face normals from first 3 vertices of each neighbor
                def face_norm(fid):
                    loc = TopLoc_Location()
                    tri = BRep_Tool().Triangulation(fm.FindKey(fid), loc)
                    if tri is None or tri.NbNodes() < 3: return None
                    trsf = loc.Transformation()
                    p1 = tri.Node(1); p1.Transform(trsf)
                    p2 = tri.Node(2); p2.Transform(trsf)
                    p3 = tri.Node(3); p3.Transform(trsf)
                    n = np.cross(np.array([p2.X()-p1.X(),p2.Y()-p1.Y(),p2.Z()-p1.Z()]),
                                 np.array([p3.X()-p1.X(),p3.Y()-p1.Y(),p3.Z()-p1.Z()]))
                    nr = np.linalg.norm(n)
                    return n/nr if nr > 1e-12 else np.array([0,0,1])
                fn1 = face_norm(n1); fn2 = face_norm(n2)
                if fn1 is not None and fn2 is not None:
                    dot = min(1.0, max(0.0, abs(np.dot(fn1, fn2))))
                    nbr_angle = float(np.arccos(dot) * 180 / np.pi)

            # Convexity check for fillet/chamfer
            is_convex = True
            try:
                loc = TopLoc_Location(); tri = BRep_Tool().Triangulation(fm.FindKey(i), loc)
                if tri is not None and tri.NbNodes() >= 3:
                    trsf = loc.Transformation()
                    p0 = tri.Node(1); p0.Transform(trsf); p1 = tri.Node(2); p1.Transform(trsf); p2 = tri.Node(3); p2.Transform(trsf)
                    fn = np.cross(np.array([p1.X()-p0.X(),p1.Y()-p0.Y(),p1.Z()-p0.Z()]),
                                  np.array([p2.X()-p0.X(),p2.Y()-p0.Y(),p2.Z()-p0.Z()]))
                    nr = np.linalg.norm(fn)
                    if nr > 1e-12: fn /= nr
                    off = diag * 0.001
                    fc = np.array([(p0.X()+p1.X()+p2.X())/3, (p0.Y()+p1.Y()+p2.Y())/3, (p0.Z()+p1.Z()+p2.Z())/3])
                    pt = gp_Pnt(fc[0]+fn[0]*off, fc[1]+fn[1]*off, fc[2]+fn[2]*off)
                    clf = BRepClass3d_SolidClassifier(shape)
                    clf.Perform(pt, 1e-4)
                    is_convex = (clf.State() != 3)
            except: pass

            # Fillet: Cylinder/Torus + 2 neighbors + radius < 10% + convex
            if occ in (GeomAbs_Cylinder,GeomAbs_Torus) and nn==2:
                if radius>0 and radius<diag*0.10 and is_convex:
                    base="fillet"
            # Chamfer (Cone): 2 neighbors + half-length < 10% + angle 85-95 + convex
            if occ==GeomAbs_Cone and nn==2:
                if cone_half>0 and cone_half<diag*0.10 and is_convex:
                    if nbr_angle is not None and 85<nbr_angle<95:
                        base="chamfer"
            # Chamfer (Plane): 2 neighbors + half-length < 10% + angle 85-95 + convex
            if occ==GeomAbs_Plane and nn==2:
                plane_half = np.sqrt(max(face_areas[i], 1e-6)) / 2.0
                if plane_half < diag*0.10 and nbr_angle is not None and 85<nbr_angle<95 and is_convex:
                    base="chamfer"
            # Sphere recovery from BSpline
            if base=="freeform":
                loc=TopLoc_Location(); tri=BRep_Tool().Triangulation(fm.FindKey(i),loc)
                if tri is not None:
                    pts=[]; trsf=loc.Transformation()
                    for j in range(1, min(tri.NbNodes(),500)+1):
                        p=tri.Node(j); p.Transform(trsf); pts.append([p.X(),p.Y(),p.Z()])
                    pts=np.array(pts)
                    if len(pts)>=30:
                        A=np.column_stack([2*pts, np.ones(len(pts))])
                        b=(pts**2).sum(axis=1)
                        try:
                            x,_,_,_=np.linalg.lstsq(A,b,rcond=None)
                            c=x[:3]; r2=x[3]+np.dot(c,c)
                            if r2>0:
                                r=np.sqrt(r2); dists=np.abs(np.linalg.norm(pts-c,axis=1)-r)
                                if np.sqrt((dists**2).mean())/max(r,1e-6)<0.02: base="sphere"
                        except: pass
            face_labels[i]=L2I[base]

        # Unified mesh: collect all triangles with face IDs
        all_verts=[]; all_tris=[]; all_fids=[]
        for i in range(1,nf+1):
            loc=TopLoc_Location(); tri=BRep_Tool().Triangulation(fm.FindKey(i),loc)
            if tri is None: continue
            trsf=loc.Transformation(); nv=tri.NbNodes(); nt=tri.NbTriangles()
            lidx={}
            for j in range(1,nv+1):
                p=tri.Node(j); p.Transform(trsf)
                all_verts.append([p.X(),p.Y(),p.Z()]); lidx[j]=len(all_verts)-1
            for j in range(1,nt+1):
                t=tri.Triangle(j); all_tris.append([lidx[t.Value(1)],lidx[t.Value(2)],lidx[t.Value(3)]])
                all_fids.append(i)

        va=np.array(all_verts,dtype=np.float32); ta=np.array(all_tris,dtype=np.int32)
        fa=np.array(all_fids,dtype=np.int32)
        # Deduplicate vertices
        uniq,inv=np.unique(va,axis=0,return_inverse=True)
        tu=inv[ta.flatten()].reshape(-1,3)
        per_tri_label=np.array([face_labels[min(fid,nf-1)] for fid in fa], dtype=np.int8)

        # Per-triangle normals
        e1=uniq[tu[:,1]]-uniq[tu[:,0]]; e2=uniq[tu[:,2]]-uniq[tu[:,0]]
        tri_normals=np.cross(e1,e2); nrm=np.linalg.norm(tri_normals,axis=1,keepdims=True).clip(1e-15)
        tri_normals/=nrm; tri_areas=nrm.flatten()*0.5

        np.savez_compressed(out, vertices=uniq, faces=tu, tri_normals=tri_normals,
                            tri_areas=tri_areas, tri_labels=per_tri_label, part_name=stem)
        dist=collections.Counter([LABEL_NAMES[l] for l in per_tri_label])
        return (stem, dict(dist))
    except Exception as e:
        return (stem, {"_err":str(e)[:100]})


if __name__ == "__main__":
    files = sorted(STEP_DIR.glob("*.step"))
    n = max(1, cpu_count()-1)
    print(f"Files: {len(files)}, Workers: {n}")
    ok=fail=0; stats=collections.Counter(); t0=time.time()
    todo = [str(f) for f in files if not (DATA_DIR/f"{f.stem}.npz").exists()]
    print(f"To do: {len(todo)}")
    with Pool(n) as p:
        for stem, s in p.imap_unordered(process_one, todo, chunksize=10):
            if "_err" in s or "_read" in s or "_few" in s: fail+=1
            else: ok+=1; stats.update(s)
            if (ok+fail)%500==0: print(f"  ok={ok} fail={fail}")
    print(f"\nDONE: ok={ok} fail={fail} in {(time.time()-t0)/60:.1f}min")
    total = sum(stats.values())
    for name in LABEL_NAMES: print(f"  {name:12s} {stats.get(name,0):10d} ({stats.get(name,0)/max(total,1)*100:.1f}%)")
