"""
Save and load a setup dictionary as JSON on disk.

Used by the app to keep the student's baseline setup between page refreshes
and app restarts. The file lives in ``user_data/`` at the repository root
(git-ignored) and looks like::

    {
      "format_version": 1,
      "name": "Stiff front ARB",
      "saved_at": "2026-11-11T11:13:05",
      "setup": { ...setup dictionary from utils.agent_parser.setup_to_dict... }
    }

``name`` is optional (files saved before names existed have none).

Loading validates the file against a *template* setup (normally the reference
setup) so that a file written by an older version of the app, or edited by
hand, is rejected instead of crashing the simulation.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "BASELINE_PATH_ENV",
    "DEFAULT_BASELINE_PATH",
    "FORMAT_VERSION",
    "SavedSetup",
    "SetupStoreError",
    "baseline_path",
    "delete_setup",
    "load_setup",
    "save_setup",
]

FORMAT_VERSION = 1

#: Where the app keeps the persisted baseline.
DEFAULT_BASELINE_PATH = Path(__file__).resolve().parent.parent / "user_data" / "baseline.json"

#: Environment variable that overrides the baseline location (the app tests use it
#: so they never touch the student's real baseline).
BASELINE_PATH_ENV = "VDL_BASELINE_PATH"


def baseline_path() -> Path:
    """File used when no explicit ``path`` is given: ``$VDL_BASELINE_PATH`` or the default."""
    override = os.environ.get(BASELINE_PATH_ENV)
    return Path(override) if override else DEFAULT_BASELINE_PATH


class SetupStoreError(ValueError):
    """The saved file exists but is unreadable or doesn't match the expected setup layout."""


#: Longest setup name kept; longer names are truncated.
MAX_NAME_LENGTH = 60


@dataclass(frozen=True)
class SavedSetup:
    setup: dict[str, Any]
    saved_at: datetime
    name: str | None = None


def save_setup(
    setup: dict[str, Any], path: Path | None = None, name: str | None = None
) -> SavedSetup:
    """Write ``setup`` (and an optional display ``name``) to ``path``, replacing any previous file.

    The file is written to a temporary name first and then renamed, so a
    crash mid-write never leaves a half-written baseline behind. Blank names
    are stored as ``None``. ``path`` defaults to :func:`baseline_path`.

    Raises
    ------
    OSError
        If the directory or file can't be written (e.g. read-only disk).
    """
    name = " ".join((name or "").split())[:MAX_NAME_LENGTH] or None
    saved_at = datetime.now().replace(microsecond=0)
    payload = {"format_version": FORMAT_VERSION, "name": name, "saved_at": saved_at.isoformat(), "setup": setup}
    path = Path(path) if path is not None else baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return SavedSetup(setup=setup, saved_at=saved_at, name=name)


def load_setup(template: dict[str, Any], path: Path | None = None) -> SavedSetup | None:
    """Read a setup saved by :func:`save_setup`.

    Parameters
    ----------
    template : dict
        A valid setup dictionary. The saved setup must contain exactly the
        same keys (at every level) with finite numeric values.
    path : Path, optional
        File to read (default: :func:`baseline_path`).

    Returns
    -------
    SavedSetup or None
        ``None`` if no file exists.

    Raises
    ------
    SetupStoreError
        If the file is corrupt, from an unknown format version, or doesn't
        match ``template``.
    """
    path = Path(path) if path is not None else baseline_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SetupStoreError(f"could not read {path.name}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise SetupStoreError(f"{path.name} is not a format-version-{FORMAT_VERSION} setup file")

    setup = payload.get("setup")
    problems = _layout_problems(setup, template)
    if problems:
        raise SetupStoreError(f"{path.name} doesn't match the current setup layout ({problems[0]})")
    try:
        saved_at = datetime.fromisoformat(payload["saved_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SetupStoreError(f"{path.name} has an invalid saved_at timestamp") from exc
    name = payload.get("name")
    if name is not None and not isinstance(name, str):
        raise SetupStoreError(f"{path.name} has an invalid name")
    return SavedSetup(setup=setup, saved_at=saved_at, name=name or None)


def delete_setup(path: Path | None = None) -> None:
    """Remove the saved file if it exists."""
    (Path(path) if path is not None else baseline_path()).unlink(missing_ok=True)


def _layout_problems(value: Any, template: Any, where: str = "setup") -> list[str]:
    if isinstance(template, dict):
        if not isinstance(value, dict):
            return [f"'{where}' should be an object"]
        problems = []
        missing = template.keys() - value.keys()
        extra = value.keys() - template.keys()
        if missing:
            problems.append(f"missing '{where}.{sorted(missing)[0]}'")
        if extra:
            problems.append(f"unexpected '{where}.{sorted(extra)[0]}'")
        for key in template.keys() & value.keys():
            problems.extend(_layout_problems(value[key], template[key], f"{where}.{key}"))
        return problems
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return [f"'{where}' should be a finite number"]
    return []
