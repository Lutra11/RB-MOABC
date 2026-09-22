"""Crash-safe artifact persistence for Risk Protocol R2 experiments."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), default=_json_default,
                      allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=_json_default, allow_nan=False),
                         encoding='utf-8')
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    fields = sorted({key for row in rows for key in row}) if rows else []
    with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(canonical_json(value) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


class ArtifactStore:
    """Owns one immutable experiment block and its resumable checkpoints."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / 'manifest.json'

    def freeze_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        frozen = dict(manifest)
        frozen['manifest_hash'] = digest(manifest)
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text(encoding='utf-8'))
            if old != frozen:
                raise ValueError('Output manifest differs; use a new directory')
            return old
        atomic_json(self.manifest_path, frozen)
        return frozen

    def progress(self, message: str) -> None:
        line = f'[{datetime.now().isoformat(timespec="seconds")}] {message}'
        print(line, flush=True)
        path = self.root / 'progress.log'
        with path.open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def event(self, event: str, **payload: Any) -> None:
        append_jsonl(self.root / 'events.jsonl', {
            'time': datetime.now().isoformat(timespec='seconds'),
            'event': event,
            **payload,
        })

    def write_run_arrays(self, run_id: str, **arrays: Any) -> Path:
        path = self.root / f'{run_id}.npz'
        atomic_npz(path, **arrays)
        return path

    def commit_run(self, run_id: str, record: dict[str, Any],
                   manifest: dict[str, Any]) -> Path:
        array_path = self.root / f'{run_id}.npz'
        if not array_path.exists():
            raise ValueError(f'Missing run arrays: {array_path}')
        final = dict(record)
        final.update(run_id=run_id, artifact=array_path.name,
                     artifact_hash=sha256_file(array_path),
                     manifest_hash=manifest['manifest_hash'])
        marker = self.root / f'{run_id}.json'
        atomic_json(marker, final)
        return marker

    def is_complete(self, run_id: str, manifest: dict[str, Any]) -> bool:
        marker = self.root / f'{run_id}.json'
        if not marker.exists():
            return False
        try:
            record = json.loads(marker.read_text(encoding='utf-8'))
            artifact = self.root / record['artifact']
            return (record.get('manifest_hash') == manifest.get('manifest_hash')
                    and artifact.exists()
                    and sha256_file(artifact) == record.get('artifact_hash'))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def commit_evaluation_checkpoint(self, run_id: str, evaluation: int,
                                     record: dict[str, Any], **arrays: Any) -> Path:
        folder = self.root / 'checkpoints' / run_id
        stem = f'eval_{evaluation:07d}'
        array_path = folder / f'{stem}.npz'
        atomic_npz(array_path, **arrays)
        marker = dict(record, evaluation=evaluation,
                      artifact=array_path.name,
                      artifact_hash=sha256_file(array_path))
        marker_path = folder / f'{stem}.json'
        atomic_json(marker_path, marker)
        return marker_path

    def commit_origin(self, run_id: str, origin: int,
                      record: dict[str, Any], **arrays: Any) -> Path:
        folder = self.root / 'checkpoints' / run_id
        stem = f'origin_{origin:04d}'
        array_path = folder / f'{stem}.npz'
        atomic_npz(array_path, **arrays)
        marker = dict(record, origin=origin, artifact=array_path.name,
                      artifact_hash=sha256_file(array_path))
        marker_path = folder / f'{stem}.json'
        atomic_json(marker_path, marker)
        return marker_path

    def latest_valid_origin(self, run_id: str) -> dict[str, Any] | None:
        folder = self.root / 'checkpoints' / run_id
        for marker_path in sorted(folder.glob('origin_*.json'), reverse=True):
            try:
                record = json.loads(marker_path.read_text(encoding='utf-8'))
                artifact = folder / record['artifact']
                if artifact.exists() and sha256_file(artifact) == record['artifact_hash']:
                    return record
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return None

