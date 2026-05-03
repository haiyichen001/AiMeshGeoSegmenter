"""
ABC 单实体过滤 + STEP 复制

解压 stat 7z → 解析 #parts → 复制 STEP 到项目目录
"""
import os, sys, re, time, shutil, tempfile, collections
from pathlib import Path
import py7zr

ROOT = Path(r"D:\AiMeshGeoSegmenter")
ABC_STAT_DIR = Path(r"F:\abc-dataset\stat")
ABC_STEP_DIR = Path(r"F:\abc-dataset\step_extracted")
OUT_DIR = ROOT / "data" / "abc_step"
os.makedirs(OUT_DIR, exist_ok=True)

CHUNKS = [f"abc_{i:04d}" for i in range(10)]


def process_chunk(chunk_name):
    stat_7z = ABC_STAT_DIR / f"{chunk_name}_stat_v00.7z"
    if not stat_7z.exists():
        print(f"  {chunk_name}: stat 7z not found")
        return []

    step_chunk_dir = ABC_STEP_DIR / chunk_name
    if not step_chunk_dir.exists():
        print(f"  {chunk_name}: step dir not found")
        return []

    single_list = []
    total, multi = 0, 0

    tmp_dir = tempfile.mkdtemp(prefix=f"abs_{chunk_name}_")
    try:
        with py7zr.SevenZipFile(str(stat_7z), 'r') as z:
            z.extractall(tmp_dir)

        for root, dirs, files in os.walk(tmp_dir):
            for fn in files:
                if not fn.endswith('.yml'):
                    continue
                fpath = os.path.join(root, fn)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as fh:
                    text = fh.read(600)

                m = re.search(r"'?#parts'?:\s*(\d+)", text)
                if not m:
                    continue
                parts = int(m.group(1))
                total += 1
                if parts == 1:
                    rel = os.path.relpath(fpath, tmp_dir)
                    # rel = "00000002\00000002_xxx_stats_001.yml"
                    # step = "abc_0000/00000002/00000002_xxx_step_001.step"
                    step_name = f"{chunk_name}/{rel}".replace('_stats_', '_step_').replace('.yml', '.step')
                    single_list.append(step_name)
                else:
                    multi += 1
    except Exception as e:
        print(f"  {chunk_name}: ERROR {e}")
        return []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"  {chunk_name}: total={total} single={len(single_list)} multi={multi}")
    return single_list


def copy_step_files(chunk_name, step_paths):
    step_chunk_dir = ABC_STEP_DIR / chunk_name
    ok, fail = 0, 0
    for rel_path in step_paths:
        rel_path = rel_path.replace('\\', '/')
        # rel_path = "abc_0000/00000002/00000002_xxx_step_001.step"
        inner = rel_path.replace(chunk_name + '/', '', 1)  # "00000002/00000002_xxx_step_001.step"
        parts = inner.split('/')
        if len(parts) < 2:
            fail += 1
            continue
        src = step_chunk_dir / inner
        if not src.exists():
            fail += 1
            continue
        model_id = parts[0]   # "00000002"
        model_fn = parts[1]   # "00000002_xxx_step_001.step"
        dest_dir = OUT_DIR / model_id
        os.makedirs(dest_dir, exist_ok=True)
        dest = dest_dir / model_fn
        if not dest.exists():
            shutil.copy2(str(src), str(dest))
        ok += 1
    return ok, fail


if __name__ == "__main__":
    print(f"Source: {ABC_STEP_DIR}")
    print(f"Target: {OUT_DIR}")
    print(f"Chunks: {CHUNKS[0]} ... {CHUNKS[-1]}\n")

    all_single = []
    t0 = time.time()

    for chunk in CHUNKS:
        t_chunk = time.time()
        single = process_chunk(chunk)
        all_single.extend(single)
        print(f"    done in {time.time()-t_chunk:.0f}s")

    print(f"\nTotal single-solid: {len(all_single)}")
    print(f"Filter time: {time.time()-t0:.0f}s")

    # Copy
    print(f"\nCopying STEP files to {OUT_DIR}...")
    t_copy = time.time()
    chunk_groups = collections.defaultdict(list)
    for sp in all_single:
        sp_norm = sp.replace('\\', '/')
        chunk_groups[sp_norm.split('/')[0]].append(sp_norm)

    total_ok, total_fail = 0, 0
    for chunk in CHUNKS:
        paths = chunk_groups.get(chunk, [])
        if not paths:
            continue
        ok, fail = copy_step_files(chunk, paths)
        total_ok += ok
        total_fail += fail
        print(f"  {chunk}: copied={ok} missing={fail}")

    elapsed = time.time() - t_copy
    print(f"\nCopied: {total_ok}, Missing: {total_fail}")
    step_count = sum(1 for _ in OUT_DIR.rglob('*.step'))
    print(f"Files in target: {step_count}")
    print(f"Copy time: {elapsed/60:.0f}min")
    total_time = time.time() - t0
    print(f"Total time: {total_time/60:.0f}min")
