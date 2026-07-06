from pathlib import Path
from typing import List, Dict


def find_paired_files(data_dir: str) -> List[Dict[str, str]]:
    """Find files with `_train` and `_exam` pairs in data_dir.

    Returns list of dicts: {'name': base, 'train': path, 'exam': path}
    """
    p = Path(data_dir)
    pairs = {}
    for f in p.glob("**/*"):
        if not f.is_file():
            continue
        name = f.stem
        if name.endswith("_train"):
            base = name[: -6]
            pairs.setdefault(base, {})["train"] = str(f)
        elif name.endswith("_exam"):
            base = name[: -5]
            pairs.setdefault(base, {})["exam"] = str(f)
    results = []
    for k, v in pairs.items():
        results.append({"name": k, "train": v.get("train"), "exam": v.get("exam")})
    return results


def ingest_uploaded_files(upload_dir: str, target_dir: str) -> None:
    """Move files from upload_dir to target_dir. Stub for Colab uploads."""
    src = Path(upload_dir)
    dst = Path(target_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            f.rename(dst / f.name)
