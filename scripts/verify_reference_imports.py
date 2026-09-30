"""Compare imported files to upstream Git blobs and reject nested Git metadata."""
from pathlib import Path
import argparse
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "style-modulation-head": "532da0151b319efa99145cdad88035c889d72ef3",
    "persona_vectors": "b8e0f044fe2410a6fad579f38324f03f13b4e917",
}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()


def verify(name, source, expected):
    target = ROOT / "reference_repos" / name
    if git(source, "rev-parse", "HEAD") != expected:
        raise ValueError(f"{name}: upstream commit changed")
    files = git(source, "ls-files", "-z").split("\x00")[:-1]
    actual = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
    if actual != set(files):
        raise ValueError(f"{name}: missing or unexpected imported files: {set(files) ^ actual}")
    if list(target.rglob(".git")):
        raise ValueError(f"{name}: nested .git metadata found")
    for file in files:
        before = git(source, "rev-parse", f"HEAD:{file}")
        after = subprocess.check_output(["git", "hash-object", str(target / file)]).decode().strip()
        if before != after:
            raise ValueError(f"{name}: imported content differs: {file}")
    print(f"{name}: {len(files)} files match {expected}, no nested Git metadata")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style-source", type=Path, required=True)
    parser.add_argument("--persona-source", type=Path, required=True)
    args = parser.parse_args()
    verify("style-modulation-head", args.style_source, SOURCES["style-modulation-head"])
    verify("persona_vectors", args.persona_source, SOURCES["persona_vectors"])
