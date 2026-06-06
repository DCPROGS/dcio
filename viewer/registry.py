"""Format registry — maps file extensions to viewer classes.

To register a new format
------------------------
1. Create ``viewer/formats/myformat_view.py`` with a class that
   subclasses :class:`~viewer.formats.base.BaseViewer`.
2. Import it here and add it to ``_VIEWERS``.

That's it — the app picks it up automatically.
"""

from __future__ import annotations

from typing import Type

from .formats.base import BaseViewer
from .formats.scan_ini_view import ScanIniViewer
from .formats.scn_view import ScnViewer

# ── Register viewers here ─────────────────────────────────────────────────
_VIEWERS: list[Type[BaseViewer]] = [
    ScanIniViewer,
    ScnViewer,
    # Add future viewers here, e.g.:
    # SsdViewer,
    # ConsAmViewer,
]
# ─────────────────────────────────────────────────────────────────────────

# Build extension → viewer map (lower-case, no dot)
_EXT_MAP: dict[str, Type[BaseViewer]] = {}
for _v in _VIEWERS:
    for _ext in _v.extensions:
        _EXT_MAP[_ext.lower()] = _v


def viewer_for_extension(ext: str) -> Type[BaseViewer] | None:
    """Return the viewer class for *ext* (e.g. ``'scn'``), or ``None``."""
    return _EXT_MAP.get(ext.lower().lstrip("."))


def all_extensions() -> list[str]:
    """Return all registered file extensions (without dot)."""
    return list(_EXT_MAP.keys())


def all_viewers() -> list[Type[BaseViewer]]:
    """Return all registered viewer classes."""
    return list(_VIEWERS)
