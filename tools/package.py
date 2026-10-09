"""Build a clean GitHub source archive with a per-file SHA-256 manifest."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"retapp", "vendor", "configs", "tests", "tools", "docs", ".github"}
FILES = {"README.md", "LICENSE", "THIRD_PARTY.md", "requirements.txt", "environment.yml",
         "pytest.ini", ".gitignore", "run.py", "recover.py", "relabel.py", "validate.py", "train_teacher.py"}
IGNORED = {"__pycache__", ".pytest_cache", ".git"}


def source_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in IGNORED for part in relative.parts) or path.suffix in (".pyc", ".pyo"):
            continue
        if relative.parts[0] in FOLDERS or relative.as_posix() in FILES:
            yield path, relative.as_posix()


def package(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path, name in source_files():
            contents = path.read_bytes()
            manifest[name] = hashlib.sha256(contents).hexdigest()
            archive.writestr("RETA-plusplus/"+name, contents)
        archive.writestr("RETA-plusplus/MANIFEST.json", json.dumps(manifest, indent=2))
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC check failed")
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read("RETA-plusplus/"+name)).hexdigest() != expected:
                raise RuntimeError(f"Archive content mismatch: {name}")
    return {"path": str(destination), "files": len(manifest)+1,
            "bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.output), indent=2))


if __name__ == "__main__":
    main()
