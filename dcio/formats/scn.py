"""
dcio.formats.scn
==================
Read and write SCAN binary files (.scn).

SCAN is a single-channel analysis program from the DCProgs suite
(University College London, Colquhoun lab).  An SCN file stores an
*idealised* single-channel record: a sequence of open/shut intervals
produced by fitting a raw patch-clamp trace.

Binary layout
-------------
The file is split into a fixed-size **header** followed by a contiguous
**data block**.  Two header layouts exist:

* **Short header** – versions ``-103`` (old simulated) and ``103``
  (simulated).  Minimal metadata; 153 bytes; ``ioffset = 154``.
* **Full header** – version ``104`` (experimental recordings).  Rich
  metadata including recording conditions, filter settings, and display
  parameters.

Data block (immediately after the header)::

    float32[nint]   – interval durations in **milliseconds**
    int16[nint]     – amplitudes in ADC intermediate units
    int8[nint]      – property flags (see :data:`FLAG_*` constants)

This module always returns intervals in **seconds** and amplitudes in
**picoamperes** so callers work entirely in SI units.

Flag bits
---------
``iprops`` is an 8-bit field; bits are OR-ed together:

* ``0`` – all OK
* ``1`` – amplitude dubious
* ``2`` – amplitude fixed
* ``4`` – amplitude of opening constrained
* ``8`` – duration unusable
* ``32`` – first interval of a sweep (CJUMP data only)
* ``64`` – last interval of a sweep (CJUMP data only)

An interval is considered unusable when ``flags & FLAG_UNUSABLE != 0``.
Bits 5 (32) and 6 (64) are only set in concentration-jump (CJUMP)
recordings; they mark sweep boundaries and should be masked off before
analysis if treating each sweep independently.

Public API
----------
* :func:`read` – load an SCN file and return an :class:`SCNRecord`.
* :func:`write` – write intervals/amplitudes/flags as a version ``-103``
  SCN file (simulated format, compatible with SCAN and dcpyps).
* :class:`SCNRecord` – dataclass returned by :func:`read`.
* :class:`SCNHeader` – dataclass holding parsed header fields.

Examples
--------
Round-trip write → read::

    import numpy as np
    from dcio.formats.scn import read, write

    intervals_s  = np.array([0.010, 0.001, 0.020])   # seconds
    amplitudes   = np.array([  0,    -50,     0  ])   # pA  (0 = shut)
    flags        = np.zeros(3, dtype=np.int8)

    write("test.scn", intervals_s, amplitudes, flags)
    rec = read("test.scn")

    print(rec)
    print(rec.intervals)   # seconds
    print(rec.amplitudes)  # pA
"""

from __future__ import annotations

import struct
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Flag constants
# ---------------------------------------------------------------------------

FLAG_OK = 0
FLAG_AMPLITUDE_DUBIOUS = 1
FLAG_AMPLITUDE_FIXED = 2
FLAG_AMPLITUDE_CONSTRAINED = 4
FLAG_UNUSABLE = 8
#: First interval of a concentration-jump sweep (CJUMP data only).
FLAG_SWEEP_FIRST = 32
#: Last interval of a concentration-jump sweep (CJUMP data only).
FLAG_SWEEP_LAST = 64

# ---------------------------------------------------------------------------
# Header dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SCNHeader:
    """Parsed header of a SCAN binary file.

    Attributes
    ----------
    version : int
        SCAN version tag embedded in the file:
        ``-103`` = old/simulated, ``103`` = simulated, ``104`` = experimental.
    record_type : str
        ``'simulated'`` or ``'experimental'``.
    n_intervals : int
        Number of intervals stored in the data block.
    data_offset : int
        Byte offset (1-based, Fortran convention) where the data block starts.
    calfac2 : float
        Calibration factor converting ADC intermediate units to picoamperes
        (pA per ADC unit).
    title : str
        Free-text title string (up to 70 characters).
    date : str
        Date string embedded in the file header (``DD-Mon-YYYY``).
    avamp : float
        Average amplitude in ADC units recorded at analysis time.
    rms : float
        RMS baseline noise in ADC units.
    ffilt : float
        Low-pass filter cut-off frequency in Hz (``-1`` if not set).
    Emem : float
        Membrane potential in mV at the time of recording.
    ipatch : int
        Patch number.
    tapeID : str
        Tape / file identifier string.
    treso : float
        Temporal resolution imposed during analysis (seconds).  Short-header
        files only; ``0.0`` for experimental files.
    tresg : float
        Temporal resolution for group analysis (seconds).  Short-header files
        only; ``0.0`` for experimental files.
    srate : float
        Sampling rate in Hz (experimental files only; ``0.0`` otherwise).
    ffilt_khz : float
        Low-pass filter expressed in kHz (experimental files only).
    extra : dict
        Remaining header fields from full (version 104) headers.
    """

    version: int
    record_type: str
    n_intervals: int
    data_offset: int
    calfac2: float
    title: str
    date: str
    avamp: float = 0.0
    rms: float = 0.0
    ffilt: float = -1.0
    Emem: float = 0.0
    ipatch: int = 0
    tapeID: str = ""
    treso: float = 0.0
    tresg: float = 0.0
    srate: float = 0.0
    ffilt_khz: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class SCNRecord:
    """An idealised single-channel record loaded from an SCN file.

    Attributes
    ----------
    header : SCNHeader
        Parsed file header (metadata).
    intervals : numpy.ndarray, float64
        Interval durations in **seconds**.  Even indices are open intervals,
        odd indices are shut intervals (or vice versa, depending on the first
        interval).
    amplitudes : numpy.ndarray, float64
        Interval amplitudes in **picoamperes**.  Shut intervals have amplitude
        ``0.0``; open intervals carry the mean single-channel current.
    flags : numpy.ndarray, int8
        Per-interval property flags.  An interval is unusable when
        ``flags[i] & FLAG_UNUSABLE != 0``.  See module-level ``FLAG_*``
        constants.
    path : Path or None
        Path of the source file, if loaded from disk.
    """

    header: SCNHeader
    intervals: np.ndarray
    amplitudes: np.ndarray
    flags: np.ndarray
    path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_intervals(self) -> int:
        """Total number of intervals (including unusable ones)."""
        return len(self.intervals)

    @property
    def usable_mask(self) -> np.ndarray:
        """Boolean mask – ``True`` for intervals where flags < FLAG_UNUSABLE."""
        return (self.flags & FLAG_UNUSABLE) == 0

    @property
    def open_mask(self) -> np.ndarray:
        """Boolean mask – ``True`` for open (non-zero amplitude) intervals."""
        return self.amplitudes != 0.0

    @property
    def shut_mask(self) -> np.ndarray:
        """Boolean mask – ``True`` for shut (zero amplitude) intervals."""
        return self.amplitudes == 0.0

    def __repr__(self) -> str:  # noqa: D105
        src = str(self.path) if self.path else "<in-memory>"
        n_usable = int(self.usable_mask.sum())
        n_open = int((self.open_mask & self.usable_mask).sum())
        n_shut = int((self.shut_mask & self.usable_mask).sum())
        mean_open = (
            float(np.mean(self.intervals[self.open_mask & self.usable_mask])) * 1e3
            if n_open
            else float("nan")
        )
        mean_shut = (
            float(np.mean(self.intervals[self.shut_mask & self.usable_mask])) * 1e3
            if n_shut
            else float("nan")
        )
        return (
            f"SCNRecord({src!r})\n"
            f"  version      : {self.header.version} ({self.header.record_type})\n"
            f"  total        : {self.n_intervals} intervals\n"
            f"  usable       : {n_usable}\n"
            f"  open / shut  : {n_open} / {n_shut}\n"
            f"  mean open    : {mean_open:.4f} ms\n"
            f"  mean shut    : {mean_shut:.4f} ms\n"
            f"  calfac2      : {self.header.calfac2:.6g} pA/ADC\n"
            f"  filter       : {self.header.ffilt:.0f} Hz"
        )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def read(path: str | Path) -> SCNRecord:
    """Read a SCAN binary file and return an :class:`SCNRecord`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.scn`` file.

    Returns
    -------
    SCNRecord
        Header metadata plus intervals (seconds), amplitudes (pA), and flags.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file header contains an unrecognised version number.

    Examples
    --------
    >>> rec = read("myrecording.scn")
    >>> rec.intervals          # seconds
    >>> rec.amplitudes         # pA
    >>> rec.flags              # int8 flag array
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    header = _read_header(path)
    intervals_s, amplitudes_pA, flags = _read_data(path, header)

    return SCNRecord(
        header=header,
        intervals=intervals_s,
        amplitudes=amplitudes_pA,
        flags=flags,
        path=path,
    )


def write(
    path: str | Path,
    intervals: np.ndarray,
    amplitudes: np.ndarray,
    flags: np.ndarray,
    *,
    calfac: float = 1.0,
    ffilt: float = -1.0,
    rms: float = 0.0,
    treso: float = 0.0,
    tresg: float = 0.0,
    Emem: float = 0.0,
    title: str = "simulated",
) -> Path:
    """Write an SCN binary file in version ``-103`` (simulated) format.

    Parameters
    ----------
    path : str or Path
        Destination file path.  The ``.scn`` extension is conventional but
        not enforced.
    intervals : array-like, float
        Interval durations in **seconds**.
    amplitudes : array-like, int or float
        Interval amplitudes in **picoamperes**.  Shut intervals must be
        ``0``; open intervals carry the mean current.  Values are stored
        as ``int16`` so will be rounded to the nearest integer.
    flags : array-like, int
        Per-interval property flags (``int8``).  Pass
        ``numpy.zeros(n, dtype=numpy.int8)`` for a clean record.
    calfac : float, optional
        Calibration factor (pA per ADC unit).  For simulated data the
        default ``1.0`` means amplitudes are stored directly as pA.
    ffilt : float, optional
        Low-pass filter cut-off in Hz.  Use ``-1.0`` (default) when not
        applicable.
    rms : float, optional
        RMS baseline noise in ADC units (default ``0.0``).
    treso : float, optional
        Imposed temporal resolution in seconds (stored in header for
        reference; default ``0.0``).
    tresg : float, optional
        Group temporal resolution in seconds (default ``0.0``).
    Emem : float, optional
        Membrane potential in mV (default ``0.0``).
    title : str, optional
        Free-text description stored in the header (max 70 characters).

    Returns
    -------
    Path
        Resolved path of the written file.

    Raises
    ------
    ValueError
        If *intervals*, *amplitudes*, and *flags* do not have the same
        length, or if any amplitude value exceeds the int16 range
        (``±32767``).

    Notes
    -----
    Intervals are converted from seconds to **milliseconds** before being
    packed as ``float32``, which is the native storage unit of the SCAN
    format.

    Examples
    --------
    >>> import numpy as np
    >>> from dcio.formats.scn import write, read
    >>> intervals_s = np.array([0.010, 0.002, 0.015])
    >>> amplitudes  = np.array([    0,   -48,     0])
    >>> flags       = np.zeros(3, dtype=np.int8)
    >>> write("out.scn", intervals_s, amplitudes, flags)
    PosixPath('out.scn')
    """
    path = Path(path)
    intervals = np.asarray(intervals, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    flags = np.asarray(flags, dtype=np.int8)

    if not (len(intervals) == len(amplitudes) == len(flags)):
        raise ValueError(
            f"intervals, amplitudes, and flags must have the same length; "
            f"got {len(intervals)}, {len(amplitudes)}, {len(flags)}"
        )

    amp_int16 = np.round(amplitudes).astype(np.int64)
    if amp_int16.size and (amp_int16.max() > 32767 or amp_int16.min() < -32768):
        raise ValueError(
            "Amplitude values must fit in int16 (±32767 pA when calfac=1.0)."
        )

    nint = len(intervals)
    avamp = float(np.mean(amplitudes[amplitudes != 0])) if np.any(amplitudes != 0) else 0.0

    # Header constants for version -103
    iscanver = -103
    ioffset = 154  # 1-based byte offset to data block (153-byte header)

    title_bytes = f"{title:<70}"[:70].encode("ascii")
    t = time.asctime()
    date_bytes = (t[8:10] + "-" + t[4:7] + "-" + t[20:24]).encode("ascii")  # DD-Mon-YYYY
    tapeid_bytes = f"{title:<24}"[:24].encode("ascii")

    intervals_ms = (intervals * 1e3).astype(np.float32)   # seconds → milliseconds
    amp_stored = amp_int16.astype(np.int16)

    with open(path, "wb") as fout:
        # --- header ---
        fout.write(struct.pack("<iii", iscanver, ioffset, nint))   # 12 bytes
        fout.write(title_bytes)                                     # 70 bytes
        fout.write(date_bytes)                                      # 11 bytes
        fout.write(tapeid_bytes)                                    # 24 bytes
        fout.write(struct.pack("<i", 0))                            # ipatch  4 bytes
        fout.write(struct.pack("<f", Emem))                        # Emem    4 bytes
        fout.write(struct.pack("<i", 0))                            # unknown 4 bytes
        fout.write(struct.pack("<f", avamp))                       # avamp   4 bytes
        fout.write(struct.pack("<f", rms))                         # rms     4 bytes
        fout.write(struct.pack("<f", ffilt))                       # ffilt   4 bytes
        fout.write(struct.pack("<f", calfac))                      # calfac2 4 bytes
        fout.write(struct.pack("<f", treso))                       # treso   4 bytes
        fout.write(struct.pack("<f", tresg))                       # tresg   4 bytes
        # total header = 12 + 70 + 11 + 24 + 4*9 = 153 bytes → ioffset = 154 ✓

        # --- data block ---
        fout.write(intervals_ms.tobytes())                         # float32 × nint
        fout.write(amp_stored.tobytes())                           # int16   × nint
        fout.write(flags.tobytes())                                # int8    × nint

    return path.resolve()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decode(raw: bytes) -> str:
    """Decode a fixed-width ASCII field, stripping null bytes and whitespace."""
    return raw.replace(b"\x00", b"").decode("ascii", errors="replace").strip()


def _read_header(path: Path) -> SCNHeader:
    """Parse the binary header of an SCN file."""
    _f = array("f")
    _i = array("i")

    with open(path, "rb") as fh:
        _i.fromfile(fh, 1); version   = _i.pop()
        _i.fromfile(fh, 1); ioffset   = _i.pop()
        _i.fromfile(fh, 1); nint      = _i.pop()

        title = _decode(fh.read(70))
        date  = _decode(fh.read(11))

        if version not in (-103, 103, 104):
            raise ValueError(
                f"Unrecognised SCAN version {version!r} in {path}. "
                "Expected -103, 103, or 104."
            )

        # ------------------------------------------------------------------
        # Short header  (versions -103 and 103)
        # ------------------------------------------------------------------
        if version in (-103, 103):
            tapeID = _decode(fh.read(24))
            _i.fromfile(fh, 1); ipatch   = _i.pop()
            _f.fromfile(fh, 1); Emem     = _f.pop()
            _i.fromfile(fh, 1)           ; _i.pop()   # unknown1
            _f.fromfile(fh, 1); avamp    = _f.pop()
            _f.fromfile(fh, 1); rms      = _f.pop()
            _f.fromfile(fh, 1); ffilt    = _f.pop()
            _f.fromfile(fh, 1); calfac2  = _f.pop()
            _f.fromfile(fh, 1); treso    = _f.pop()
            _f.fromfile(fh, 1); tresg    = _f.pop()

            return SCNHeader(
                version=version,
                record_type="simulated",
                n_intervals=nint,
                data_offset=ioffset,
                calfac2=calfac2,
                title=title,
                date=date,
                avamp=avamp,
                rms=rms,
                ffilt=ffilt,
                Emem=Emem,
                ipatch=ipatch,
                tapeID=tapeID,
                treso=treso,
                tresg=tresg,
            )

        # ------------------------------------------------------------------
        # Full header (version 104 – experimental recordings)
        # ------------------------------------------------------------------
        extra: dict = {}

        extra["defname"] = _decode(fh.read(6))
        tapeID = _decode(fh.read(24))
        _i.fromfile(fh, 1); ipatch        = _i.pop()
        _i.fromfile(fh, 1); extra["npatch"] = _i.pop()
        _f.fromfile(fh, 1); Emem          = _f.pop()
        _f.fromfile(fh, 1); extra["temper"] = _f.pop()
        extra["adcfil"] = _decode(fh.read(30))
        extra["qfile1"] = _decode(fh.read(35))

        for key in ("cjump", "nfits", "ntmax", "nfmax", "nbuf", "novlap"):
            _i.fromfile(fh, 1); extra[key] = _i.pop()

        _f.fromfile(fh, 1); srate        = _f.pop()
        _f.fromfile(fh, 1); extra["finter"] = _f.pop()
        _f.fromfile(fh, 1); extra["tsect"]  = _f.pop()

        for key in ("ioff", "ndat", "nsec", "nrlast"):
            _i.fromfile(fh, 1); extra[key] = _i.pop()

        _f.fromfile(fh, 1); extra["avtot"] = _f.pop()
        _i.fromfile(fh, 1); extra["navamp"] = _i.pop()
        _f.fromfile(fh, 1); avamp          = _f.pop()
        _f.fromfile(fh, 1); rms            = _f.pop()

        for key in ("nwrit", "nwsav", "newpar", "opendown", "invert",
                    "usepots", "disp"):
            _i.fromfile(fh, 1); extra[key] = _i.pop()

        _f.fromfile(fh, 1); extra["smult"]  = _f.pop()
        _i.fromfile(fh, 1); extra["scrit"]  = _i.pop()
        _i.fromfile(fh, 1); extra["vary"]   = _i.pop()
        _i.fromfile(fh, 1); extra["ntrig"]  = _i.pop()
        _i.fromfile(fh, 1); extra["navtest"] = _i.pop()
        _f.fromfile(fh, 1); extra["dgain"]  = _f.pop()
        _i.fromfile(fh, 1); extra["iboff"]  = _i.pop()
        _f.fromfile(fh, 1); extra["expfac"] = _f.pop()
        _f.fromfile(fh, 1); extra["bdisp"]  = _f.pop()
        _i.fromfile(fh, 1); extra["ibflag"] = _i.pop()
        _i.fromfile(fh, 1); extra["iautosub"] = _i.pop()
        _f.fromfile(fh, 1); extra["xtrig"]  = _f.pop()

        extra["ndev"]    = _decode(fh.read(2))
        extra["cdate"]   = _decode(fh.read(11))
        extra["adctime"] = _decode(fh.read(8))

        _i.fromfile(fh, 1); extra["nsetup"]  = _i.pop()
        extra["filtfile"] = _decode(fh.read(20))

        _f.fromfile(fh, 1); ffilt          = _f.pop()
        _i.fromfile(fh, 1); extra["npfilt"] = _i.pop()
        _f.fromfile(fh, 1); extra["sfac1"]  = _f.pop()
        _f.fromfile(fh, 1); extra["sfac2"]  = _f.pop()
        _f.fromfile(fh, 1); extra["sfac3"]  = _f.pop()
        _i.fromfile(fh, 1); extra["nscale"] = _i.pop()
        _f.fromfile(fh, 1); extra["calfac"] = _f.pop()
        _f.fromfile(fh, 1); extra["calfac1"] = _f.pop()
        _f.fromfile(fh, 1); calfac2         = _f.pop()

        return SCNHeader(
            version=version,
            record_type="experimental",
            n_intervals=nint,
            data_offset=ioffset,
            calfac2=calfac2,
            title=title,
            date=date,
            avamp=avamp,
            rms=rms,
            ffilt=ffilt,
            Emem=Emem,
            ipatch=ipatch,
            tapeID=tapeID,
            srate=srate,
            ffilt_khz=ffilt / 1000.0,
            extra=extra,
        )


def _read_data(
    path: Path, header: SCNHeader
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the data block and return (intervals_s, amplitudes_pA, flags)."""
    tint   = array("f")   # float32 – milliseconds
    iampl  = array("h")   # int16   – ADC intermediate units
    iprops = array("b")   # int8    – property flags

    with open(path, "rb") as fh:
        fh.seek(header.data_offset - 1)   # ioffset is 1-based (Fortran)
        tint.fromfile(fh, header.n_intervals)
        iampl.fromfile(fh, header.n_intervals)
        iprops.fromfile(fh, header.n_intervals)

    # Experimental files: ensure the record ends with a shut interval.
    # SCAN appends a sentinel: the last interval must have amplitude 0.
    if header.version > 0:
        while iampl and iampl[-1] != 0:
            tint.pop(); iampl.pop(); iprops.pop()
        if iampl:
            iprops[-1] = FLAG_UNUSABLE   # mark the sentinel unusable

    intervals_s     = np.array(tint,  dtype=np.float64) * 1e-3  # ms → s
    amplitudes_pA   = np.array(iampl, dtype=np.float64) * header.calfac2
    flags           = np.array(iprops, dtype=np.int8)

    return intervals_s, amplitudes_pA, flags
