"""Validate historical source hashes across the shared-module relocation.

The registry accepts only the exact pre-move and post-move bytes. Subsequent
edits still invalidate cached results. Retired package initializers and the
unused V1 checkpoint verifier are recorded explicitly, not matched by prefix.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).with_suffix('.json')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_matches(path, expected):
    """Check a current or historical source path without needing old folders."""
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        relative = None
    records = json.loads(REGISTRY.read_text())
    for old, record in records.items():
        if relative is None or relative not in (old, record['path']) or expected != record['before']:
            continue
        if record['path'] is None:
            return True  # Explicitly retired source; not part of any current model.
        target = ROOT / record['path']
        return target.is_file() and _sha256(target) == record['after']
    return path.is_file() and _sha256(path) == expected


def cache_source_identity(path):
    """Keep deterministic data-cache keys stable only for unchanged relocations."""
    path = Path(path)
    actual = _sha256(path)
    relative = path.relative_to(ROOT).as_posix()
    for old, record in json.loads(REGISTRY.read_text()).items():
        if record['path'] == relative and actual == record['after']:
            return str(ROOT / old), record['before']
    return str(path), actual
