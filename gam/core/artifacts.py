from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Root directory for persisted state artifacts
_STATE_ROOT = Path("data") / "outputs" / "state"


def scene_json_path(analysis_id: str) -> Path:
    """
    Compute the filesystem path for the scene.json associated with an analysis.

    Parameters
    ----------
    analysis_id : str
        Identifier used to namespace the persisted scene configuration.

    Returns
    -------
    pathlib.Path
        Path of the form: data/outputs/state/{analysis_id}/scene.json
    """
    return _STATE_ROOT / analysis_id / "scene.json"


def save_scene_config(analysis_id: str, scene_json: str) -> Path:
    """
    Persist a scene configuration to disk.

    The provided scene_json is written to data/outputs/state/{analysis_id}/scene.json.
    If the content is valid JSON, it will be pretty-printed before saving.

    Parameters
    ----------
    analysis_id : str
        Identifier used to namespace the persisted scene configuration.
    scene_json : str
        A JSON string representing the scene (as exported by the GlobeViewer).

    Returns
    -------
    pathlib.Path
        The full path to the saved scene.json file.
    """
    path = scene_json_path(analysis_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = scene_json
    try:
        parsed = json.loads(scene_json)
        content = json.dumps(parsed, indent=2)
    except Exception as exc:  # json.JSONDecodeError or other parsing issues
        logger.debug(
            "save_scene_config: provided scene_json is not valid JSON; writing raw text. "
            "analysis_id=%s error=%s",
            analysis_id,
            exc,
        )

    # Ensure trailing newline for POSIX-friendly text files
    if not content.endswith("\n"):
        content = content + "\n"

    path.write_text(content, encoding="utf-8")
    logger.info("Saved scene configuration to %s", path)
    return path


def load_scene_config(analysis_id: str) -> str:
    """
    Load a persisted scene configuration for the given analysis_id.

    Parameters
    ----------
    analysis_id : str
        Identifier used to locate the persisted scene configuration.

    Returns
    -------
    str
        The scene configuration as a UTF-8 decoded JSON string.

    Raises
    ------
    FileNotFoundError
        If the scene.json file does not exist for the provided analysis_id.
    """
    path = scene_json_path(analysis_id)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path.read_text(encoding="utf-8")