"""Small artifact helpers; completed manifests are the commit marker for runs."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess

import numpy as np

from .paths import REPO_ROOT, WORKSTREAM_ROOT


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_run(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def provenance():
    root = REPO_ROOT
    def git(*args):
        p = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    packages = {}
    for name in ("numpy", "torch", "transformers", "accelerate", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    sources = {str(p.relative_to(root)): sha256(p)
               for p in sorted((WORKSTREAM_ROOT / "attentionseekers").glob("*.py"))}
    return {"utc": datetime.now(timezone.utc).isoformat(), "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")), "packages": packages,
            "source_hashes": sources}


def finish_run(path, metadata):
    path = Path(path)
    files = {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob("*"))
             if p.is_file() and p != path / "manifest.json"}
    write_json(path / "manifest.json", {"schema_version": 1, "status": "complete",
                                      **metadata, "files": files})


def load_manifest(path, verify=True):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("schema_version") != 1:
        raise ValueError("Run is incomplete or has an unsupported schema version")
    if verify:
        for name, digest in manifest["files"].items():
            if not (path / name).is_file() or sha256(path / name) != digest:
                raise ValueError(f"Artifact missing or changed: {path / name}")
    return manifest


def load_array(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)
