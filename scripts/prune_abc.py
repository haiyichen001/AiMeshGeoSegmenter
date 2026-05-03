"""
ABC 数据集瘦身：统计每零件 #surfs 面数，保留最简单的 20,000 个，删除其余
"""
import os, re, time, shutil, tempfile, collections
from pathlib import Path
import py7zr

ROOT = Path(r"D:\AiMeshGeoSegmenter")
ABC_STAT_DIR = Path(r"F:\abc-dataset\stat")
ABC_STEP_DIR = ROOT / "data" / "abc_step"
CHUNKS = [f"abc_{i:04d}" for i in range(10)]
KEEP_N = 20000


def main():
    # Step 1: Collect surfs count for all single-solid parts
    all_parts = []  # [(surfs, model_id, step_rel_path)]

    for chunk in CHUNKS:
        stat_7z = ABC_STAT_DIR / f"{chunk}_stat_v00.7z"
        if not stat_7z.exists():
            print(f"  {chunk}: stat not found, skip")
            continue

        tmp_dir = tempfile.mkdtemp(prefix=f"prune_{chunk}_")
        try:
            with py7zr.SevenZipFile(str(stat_7z), 'r') as z:
                z.extractall(tmp_dir)

            count = 0
            for root, dirs, files in os.walk(tmp_dir):
                for fn in files:
                    if not fn.endswith('.yml'):
                        continue
                    fpath = os.path.join(root, fn)
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as fh:
                        text = fh.read(600)

                    m_parts = re.search(r"'?#parts'?:\s*(\d+)", text)
                    m_surfs = re.search(r"'?#surfs'?:\s*(\d+)", text)
                    if not m_parts or not m_surfs:
                        continue
                    if int(m_parts.group(1)) != 1:
                        continue

                    surfs = int(m_surfs.group(1))
                    rel = os.path.relpath(fpath, tmp_dir)
                    step_name = f"{chunk}/{rel}".replace('_stats_', '_step_').replace('.yml', '.step')
                    model_id = rel.split(os.sep)[0]
                    all_parts.append((surfs, model_id, step_name))
                    count += 1

            print(f"  {chunk}: {count} single-solid candidates")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nTotal single-solid: {len(all_parts)}")

    # Step 2: Sort by surfs (simpler first), keep top N
    all_parts.sort(key=lambda x: x[0])
    keep = all_parts[:KEEP_N]
    to_delete = set()
    for _, mid, step_path in all_parts[KEEP_N:]:
        fn = os.path.basename(step_path)
        to_delete.add((mid, fn))

    keep_surfs = [p[0] for p in keep]
    del_surfs_min = all_parts[KEEP_N][0] if len(all_parts) > KEEP_N else 9999
    print(f"Keep: {len(keep)} (surfs: {keep_surfs[0]}-{keep_surfs[-1]})")
    print(f"Delete: {len(to_delete)} (surfs >= {del_surfs_min})")

    # Step 3: Delete
    deleted = 0
    for mid, fn in to_delete:
        target = ABC_STEP_DIR / mid / fn
        if target.exists():
            target.unlink()
            deleted += 1
            # Remove empty directory
            parent = target.parent
            if not any(parent.iterdir()):
                parent.rmdir()

    print(f"\nDeleted: {deleted} files")
    remaining = sum(1 for _ in ABC_STEP_DIR.rglob("*.step"))
    print(f"Remaining: {remaining} STEP files")

    # Distribution summary
    print(f"\nSurfs distribution in kept parts:")
    ranges = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 200), (200, 9999)]
    for lo, hi in ranges:
        c = sum(1 for p in keep if lo <= p[0] < hi)
        if c:
            print(f"  surfs {lo:3d}-{hi:3d}: {c:5d}")


if __name__ == "__main__":
    main()
