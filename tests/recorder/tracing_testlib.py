"""Helpers shared by recorder unit tests."""

import gzip
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY312 = Path(os.environ["CAPX_PYTHON312"]) if os.environ.get("CAPX_PYTHON312") else None

def clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAPX_TRACE")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env

def file_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.name.endswith(".gz"):
        try:
            return gzip.decompress(raw)
        except OSError:
            return raw
    return raw

def files_containing(root: Path, needle: bytes) -> list[Path]:
    return [p for p in sorted(Path(root).rglob("*")) if p.is_file() and needle in file_bytes(p)]
