"""Shared strict-JSON helpers for validation and report artifacts.

Python's standard :mod:`json` writer allows non-standard ``NaN`` and
``Infinity`` tokens by default. Validation artifacts need to be portable to
strict JSON parsers, so this module recursively converts numpy/path/scalar
objects to JSON-native values and replaces non-finite numbers with ``None``
before writing with ``allow_nan=False``.
"""

from __future__ import annotations

import importlib
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, IO

try:  # Optional: keep this helper importable without numpy in minimal contexts.
    _np: Any = importlib.import_module("numpy")
except Exception:  # pragma: no cover - exercised only in numpy-free environments
    _np = None


def to_strict_jsonable(value: Any) -> Any:
    """Return ``value`` converted to strict JSON-compatible Python objects.

    Non-finite numeric values (``NaN``, ``Infinity``, ``-Infinity``) become
    ``None`` so unavailable numeric measurements serialize as JSON ``null``.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if _np is not None:
        ndarray_type = getattr(_np, "ndarray", None)
        generic_type = getattr(_np, "generic", None)
        if ndarray_type is not None and isinstance(value, ndarray_type):
            return to_strict_jsonable(value.tolist())
        if generic_type is not None and isinstance(value, generic_type):
            return to_strict_jsonable(value.item())
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): to_strict_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_strict_jsonable(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_strict_jsonable(dataclasses.asdict(value))
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def dump_strict_json(data: Any, fp: IO[str], **kwargs: Any) -> None:
    """Write strict JSON to an open text file object.

    Caller-supplied keyword arguments are passed through to :func:`json.dump`,
    but ``allow_nan`` is always forced to ``False``.
    """
    kwargs.pop("allow_nan", None)
    json.dump(to_strict_jsonable(data), fp, allow_nan=False, **kwargs)


def dumps_strict_json(data: Any, **kwargs: Any) -> str:
    """Return a strict JSON string for ``data`` with non-finite values nulled."""
    kwargs.pop("allow_nan", None)
    return json.dumps(to_strict_jsonable(data), allow_nan=False, **kwargs)


def write_strict_json(path: Path | str, data: Any, **kwargs: Any) -> None:
    """Write strict JSON to ``path``, creating parent directories if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        dump_strict_json(data, f, **kwargs)
        f.write("\n")
