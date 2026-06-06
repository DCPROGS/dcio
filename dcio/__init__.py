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

__version__ = "0.1.0"
__author__ = "Remis Lape"
