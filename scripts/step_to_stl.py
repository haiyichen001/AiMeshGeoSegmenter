"""STEP → STL 批量转换 (pythonOCC BRepMesh)"""
import os, math
from pathlib import Path
from multiprocessing import Pool, cpu_count
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

ROOT = Path(__file__).parent.parent
STEP_DIR = ROOT / "step"
STL_DIR = ROOT / "stl"
os.makedirs(STL_DIR, exist_ok=True)

DEFLECTION = 0.1  # 三角化精度 (mm)

def process_one(step_path_str):
    step_path = Path(step_path_str)
    stem = step_path.stem
    out_path = os.path.join(STL_DIR, f"{stem}.stl")
    if os.path.exists(out_path):
        return (stem, "skip")
    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
            return (stem, "read_fail")
        reader.TransferRoots()
        shape = reader.OneShape()
        mesh = BRepMesh_IncrementalMesh(shape, DEFLECTION)
        mesh.Perform()
        writer = StlAPI_Writer()
        writer.Write(shape, out_path)
        return (stem, "ok")
    except Exception as e:
        return (stem, f"err:{e}")

if __name__ == "__main__":
    all_files = sorted(Path(STEP_DIR).glob("*.step"))
    to_do = [str(f) for f in all_files if not os.path.exists(os.path.join(STL_DIR, f"{f.stem}.stl"))]
    print(f"Total: {len(all_files)}, to do: {len(to_do)}, deflection={DEFLECTION}mm")
    ok = fail = skip = 0
    n = max(1, cpu_count() - 1)
    print(f"Workers: {n}")
    with Pool(n) as p:
        for stem, status in p.imap_unordered(process_one, to_do, chunksize=20):
            if status == "ok": ok += 1
            elif status == "skip": skip += 1
            else: fail += 1
            if (ok + fail) % 100 == 0:
                print(f"  ok={ok} fail={fail} skip={skip}")
    print(f"\nDONE: ok={ok} fail={fail} skip={skip}")
    print(f"STL files: {len(list(Path(STL_DIR).glob('*.stl')))}")
