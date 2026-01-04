import hashlib
import json
from pathlib import Path
from datetime import datetime


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def record_dataset(
    dataset_path: Path,
    source: str,
    params: dict,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_path": str(dataset_path),
        "hash": hash_file(dataset_path),
        "source": source,
        "params": params,
    }

    record_path = output_dir / f"{dataset_path.stem}_lineage.json"
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)

    return record_path
