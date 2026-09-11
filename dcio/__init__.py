"""
dcio
======
I/O library for electrophysiology file formats used by the DCProgs suite.

Supported formats
-----------------
* **SCN** – SCAN idealised single-channel records (read + write).

Quick start
-----------
>>> from dcio.formats.scn import read, write
>>> rec = read("myfile.scn")
>>> print(rec)
"""

# Taken from the installed package metadata rather than written here, so it
# cannot drift from pyproject.toml. It already had: 0.1.1 was published with
# pyproject saying 0.1.1 and this saying 0.1.0, so `pip install "dcio>=0.1.1"`
# succeeded and `dcio.__version__` then reported 0.1.0.
try:
    from importlib.metadata import PackageNotFoundError, version as _version
    try:
        __version__ = _version("dcio")
    except PackageNotFoundError:        # a source tree that was never installed
        __version__ = "0.0.0+unknown"
    del _version, PackageNotFoundError
except ImportError:                     # pragma: no cover
    __version__ = "0.0.0+unknown"
__author__ = "Remis Lape"
