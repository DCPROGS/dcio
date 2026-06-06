"""Base class for all file-format viewers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseViewer(ABC):
    """Contract every format viewer must satisfy.

    To add a new format:
    1. Subclass ``BaseViewer`` in ``viewer/formats/yourformat.py``.
    2. Fill in ``name``, ``extensions``, ``description``.
    3. Implement ``read()`` and ``render()``.
    4. Import and register the class in ``viewer/registry.py``.
    """

    #: Human-readable format name shown in the UI.
    name: str = ""
    #: Tuple of lower-case file extensions this viewer handles (no dot).
    extensions: tuple[str, ...] = ()
    #: One-line description shown as a tooltip / subtitle.
    description: str = ""

    @classmethod
    @abstractmethod
    def read(cls, data: bytes, filename: str) -> Any:
        """Parse *data* (raw file bytes) and return a record object.

        Parameters
        ----------
        data:
            Raw bytes from the uploaded file.
        filename:
            Original filename (used for error messages and metadata).

        Returns
        -------
        Any
            Format-specific record dataclass.

        Raises
        ------
        Exception
            Any exception is caught by the app and shown as an error.
        """

    @classmethod
    @abstractmethod
    def render(cls, record: Any) -> None:
        """Render *record* using Streamlit calls.

        Must call only ``st.*`` functions — no return value.
        """
