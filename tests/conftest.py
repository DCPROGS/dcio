"""Shared pytest fixtures for dcio tests."""

from pathlib import Path

import numpy as np
import pytest

from dcio.formats.scn import write

EXAMPLES_SCN = Path(__file__).parent.parent / "examples" / "scn"


@pytest.fixture()
def minimal_scn(tmp_path: Path) -> Path:
    """Write a minimal simulated SCN file and return its path.

    Record layout (all in physical units):
        index 0 – shut  0.010 s   0 pA    flag 0
        index 1 – open  0.002 s  -48 pA   flag 0
        index 2 – shut  0.020 s   0 pA    flag 0
        index 3 – open  0.005 s  -48 pA   flag 0
        index 4 – shut  0.030 s   0 pA    flag 8  (unusable)
    """
    intervals = np.array([0.010, 0.002, 0.020, 0.005, 0.030])
    amplitudes = np.array([0, -48, 0, -48, 0], dtype=np.float64)
    flags = np.array([0, 0, 0, 0, 8], dtype=np.int8)
    path = tmp_path / "minimal.scn"
    write(path, intervals, amplitudes, flags, title="pytest minimal record")
    return path


@pytest.fixture()
def legacy_test_scn() -> Path:
    """Return path to the simulated SCN example file; skip if not found."""
    p = EXAMPLES_SCN / "glyr_simulated.scn"
    if not p.exists():
        pytest.skip("glyr_simulated.scn not found.")
    return p


@pytest.fixture()
def legacy_experimental_scn() -> Path:
    """Return path to the version-104 experimental SCN example; skip if not found."""
    p = EXAMPLES_SCN / "glyr_experimental.scn"
    if not p.exists():
        pytest.skip("glyr_experimental.scn not found.")
    return p


@pytest.fixture()
def examples_dir() -> Path:
    """Directory holding the example SCN records."""
    return EXAMPLES_SCN
