"""Hashing helpers used throughout the GeoAnomalyMapper pipelines."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Union

BufferLike = Union[bytes, bytearray, memoryview]


def sha256_path(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    """Return the SHA256 hash of the file at ``path``.

    The hash is returned in the ``sha256:<digest>`` multihash format used by
    STAC assets and MLflow logging.
    """
    digest = hashlib.sha256()
    with open(Path(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def sha256_bytes(data: BufferLike) -> str:
    """Return the SHA256 hash of a bytes-like object."""
    digest = hashlib.sha256()
    digest.update(memoryview(data))
    return f"sha256:{digest.hexdigest()}"


def stable_hash_dict(data: Mapping) -> str:
    """Generate a deterministic hash for a mapping.

    The mapping is converted to JSON using sorted keys to ensure stability.
    """
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


__all__ = ["sha256_path", "sha256_bytes", "stable_hash_dict"]
