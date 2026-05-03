"""STEP -> STL batch conversion with random deflection"""
import os, math, random, json, time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

ROOT = Path(__file__).parent.parent
STEP_DIR = ROOT / "data" / "step"
STL_DIR = ROOT / "data" / "stl"
MANIFEST = STL_DIR / "_deflections.json"
os.makedirs(STL_DIR, exist_ok=True)


def process_one(step_path_str):
    step_path = Path(step_path_str)
    stem = step_path.stem
    out_path = os.path.join(STL_DIR, f"{stem}.stl")
    if os.path.exists(out_path):
        return (stem, "skip", 0.0)
    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
            return (stem, "read_fail", 0.0)
        reader.TransferRoots()
        shape = reader.OneShape()
        # Bounding box diagonal
        bbox = Bnd_Box(); brepbndlib.Add(shape, bbox)
        x1,y1,z1,x2,y2,z2 = bbox.Get()
        diag = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        # Random deflection: 0.0005~0.02 * diagonal
        ratio = 10 ** random.uniform(-3.3, -1.7)  # ~0.0005 to ~0.02
        defelection = max(0.001, diag * ratio)
        mesh = BRepMesh_IncrementalMesh(shape, defelection)
        mesh.Perform()
        writer = StlAPI_Writer()
        writer.Write(shape, out_path)
        return (stem, "ok", round(defelection, 4))
    except Exception as e:
        return (stem, f"err:{e}", 0.0)


if __name__ == "__main__":
    # Load previous deflections if exists
    deflections = {}
    if MANIFEST.exists():
        with open(MANIFEST) as f:
            deflections = json.load(f)

    all_files = sorted(Path(STEP_DIR).glob("*.step"))
    to_do = [str(f) for f in all_files if f"{f.stem}" not in deflections]

    n = max(1, cpu_count() - 1)
    print(f"Total: {len(all_files)}, To do: {len(to_do)}, Workers: {n}")

    ok = fail = skip = 0
    with Pool(n) as p:
        for stem, status, defl in p.imap_unordered(process_one, to_do, chunksize=20):
            if status == "ok": ok += 1; deflections[stem] = defl
            elif status == "skip": skip += 1; deflections[stem] = deflections.get(stem, 0)
            else: fail += 1
            if (ok + fail) % 500 == 0:
                print(f"  ok={ok} fail={fail} skip={skip}")

    # Save deflection manifest
    with open(MANIFEST, 'w') as f:
        json.dump(deflections, f, indent=2)

    vals = [v for v in deflections.values() if v > 0]
    print(f"\nDONE: ok={ok} fail={fail} skip={skip}")
    print(f"STL files: {sum(1 for _ in STL_DIR.glob('*.stl'))}")
    print(f"Deflection range: {min(vals):.4f} ~ {max(vals):.4f} mm")
