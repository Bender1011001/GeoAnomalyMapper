"""Feature schema management."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..utils.hashing import stable_hash_dict


@dataclass(frozen=True)
class FeatureSchema:
    bands: List[str]

    @property
    def hash(self) -> str:
        return stable_hash_dict({"bands": self.bands})

    def validate(self, bands: Iterable[str]) -> None:
        candidate = list(bands)
        if candidate != self.bands:
            raise ValueError("Feature bands do not match schema")

    def to_json(self) -> str:
        return json.dumps({"bands": self.bands, "hash": self.hash}, indent=2)

    @classmethod
    def from_file(cls, path: Path) -> "FeatureSchema":
        data = json.loads(Path(path).read_text())
        return cls(bands=list(data["bands"]))

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json())


def ensure_schema(path: Path, bands: Iterable[str]) -> FeatureSchema:
    path = Path(path)
    if path.exists():
        schema = FeatureSchema.from_file(path)
        schema.validate(bands)
        return schema
    schema = FeatureSchema(list(bands))
    schema.save(path)
    return schema


__all__ = ["FeatureSchema", "ensure_schema"]
