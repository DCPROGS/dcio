"""
dcio.formats.ssd
================
Read and write CONSAM SSD files (.ssd / .dat).

CONSAM is the continuous single-channel analysis program from the DCProgs
suite (University College London, Colquhoun lab).  An SSD file stores a
**raw continuous** patch-clamp trace: a sequence of ADC samples acquired
at a fixed sampling rate.

Binary layout
-------------
The file has a fixed 512-byte header, zero-padded after the metadata
fields, followed by a contiguous block of 16-bit ADC samples::

    ┌──────────────────────────┐
    │   Header  (512 bytes)    │
    ├──────────────────────────┤
    │   int16[n_samples]       │  2 × n_samples bytes
    └──────────────────────────┘

All multi-byte numerics are **little-endian** (the CONSAM program ran on
DOS/Windows x86 systems).

Amplitude scaling
-----------------
Samples are stored as ``int16`` ADC units.  The conversion to physical
units uses the header calibration factor::

    signal_pA = int16_sample × calfac
    calfac     = 1.0 / (gain × 6553.6)

where ``gain`` is the transducer gain in V/pA (set at the amplifier) and
``6553.6 ≈ 2¹⁶ / 10`` maps the ±5 V ADC input range to ``int16``.

This module always returns the signal in **picoamperes** so callers work
entirely in physical units.

Sampling interval
-----------------
``idt`` is the sampling interval stored in the header as an ``int16``
value in **microseconds**.  The sampling rate in Hz is ``1e6 / idt``.

Public API
----------
* :func:`read`  – load an SSD file and return an :class:`SSDRecord`.
* :func:`write` – write a signal array as a version 1002 SSD file.
* :class:`SSDRecord` – dataclass returned by :func:`read`.
* :class:`SSDHeader` – dataclass holding parsed header fields.

Examples
--------
Round-trip write → read::

    import numpy as np
    from dcio.formats.ssd import read, write

    # 1 second of synthetic noise at 50 kHz
    rng     = np.random.default_rng(0)
    signal  = rng.normal(0, 2.0, 50_000)   # pA
    dt_us   = 20                            # 50 kHz

    write("test.ssd", signal, dt_us, gain=0.05, filt=10_000.0)
    rec = read("test.ssd")

    print(rec)
    print(rec.dt)        # 2e-05  (seconds)
    print(rec.signal)    # array in pA
"""

from __future__ import annotations

import datetime
import struct
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Header dataclass
# ---------------------------------------------------------------------------

_HEADER_SIZE = 512   # data always starts at byte 512 (ioff)


@dataclass
class SSDHeader:
    """Parsed header of a CONSAM SSD file.

    Attributes
    ----------
    version : int
        Format version: ``1002`` (current) or ``1001`` (old).
    n_samples : int
        Number of samples in the data block (``ilen`` field).
    dt_us : int
        Sampling interval in **microseconds** (``idt`` field).
    calfac : float
        Calibration factor converting ADC ``int16`` units to picoamperes:
        ``signal_pA = sample × calfac``.
    srate : float
        Sample rate in Hz (derived: ``1e6 / dt_us``).
    filt : float
        Low-pass filter cut-off frequency in Hz at the time of recording.
    filt1 : float
        Secondary filter frequency in Hz (often ``0.0``).
    calfac1 : float
        Secondary calibration factor (often ``0.0``).
    title : str
        Free-text description (up to 70 characters).
    date : str
        Acquisition date string (``DD-Mon-YYYY``).
    time : str
        Acquisition time string (``HH-MM-SS``).
    expdate : str
        Experiment date string (``DD-Mon-YYYY``).
    defname : str
        Default file-name prefix (up to 6 characters).
    tapeID : str
        Tape / file identifier (up to 24 characters).
    ipatch : int
        Patch number.
    npatch : int
        Total number of patches in the series.
    Emem : float
        Membrane potential in mV.
    temp : float
        Bath temperature in °C.
    inc : int
        Increment between stored samples (``1`` = every sample).
    id1 : int
        Channel identifier 1 (application-specific).
    id2 : int
        Channel identifier 2 (application-specific).
    cs : str
        Channel string code (e.g. ``"H"`` = holding current).
    data_offset : int
        Byte offset to the start of the data block (always ``512``).
    """

    version: int
    n_samples: int
    dt_us: int
    calfac: float
    srate: float
    filt: float
    title: str
    date: str
    time: str
    filt1: float = 0.0
    calfac1: float = 0.0
    expdate: str = ""
    defname: str = ""
    tapeID: str = ""
    ipatch: int = 0
    npatch: int = 1
    Emem: float = 0.0
    temp: float = 0.0
    inc: int = 1
    id1: int = 0
    id2: int = 0
    cs: str = "H"
    data_offset: int = _HEADER_SIZE


# ---------------------------------------------------------------------------
# Record dataclass
# ---------------------------------------------------------------------------


@dataclass
class SSDRecord:
    """A continuous raw trace loaded from a CONSAM SSD file.

    Attributes
    ----------
    header : SSDHeader
        Parsed file header (metadata).
    signal : numpy.ndarray, float64
        Trace data in **picoamperes**, shape ``(n_samples,)``.
    dt : float
        Sampling interval in **seconds** (convenience; equal to
        ``header.dt_us × 1e-6``).
    path : Path or None
        Path of the source file if loaded from disk.
    """

    header: SSDHeader
    signal: np.ndarray
    dt: float
    path: Optional[Path] = None

    @property
    def n_samples(self) -> int:
        """Number of samples in the trace."""
        return len(self.signal)

    @property
    def duration(self) -> float:
        """Total recording duration in seconds."""
        return self.n_samples * self.dt

    @property
    def time_axis(self) -> np.ndarray:
        """Time axis in seconds, shape ``(n_samples,)``."""
        return np.arange(self.n_samples) * self.dt

    def __repr__(self) -> str:  # noqa: D105
        src = str(self.path) if self.path else "<in-memory>"
        return (
            f"SSDRecord({src!r})\n"
            f"  version    : {self.header.version}\n"
            f"  n_samples  : {self.n_samples:,}\n"
            f"  duration   : {self.duration * 1e3:.2f} ms\n"
            f"  dt         : {self.dt * 1e6:.2f} µs  ({self.header.srate:.0f} Hz)\n"
            f"  filter     : {self.header.filt:.0f} Hz\n"
            f"  signal     : {self.signal.min():.2f} … {self.signal.max():.2f} pA"
            f"  (mean {self.signal.mean():.2f} pA)\n"
            f"  Emem       : {self.header.Emem:.1f} mV"
        )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def read(path: str | Path) -> SSDRecord:
    """Read a CONSAM SSD file and return an :class:`SSDRecord`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.ssd`` or ``.dat`` file.

    Returns
    -------
    SSDRecord
        Parsed header plus signal in picoamperes.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the version tag is not ``1001`` or ``1002``.

    Examples
    --------
    >>> rec = read("trace.ssd")
    >>> rec.signal          # picoamperes, shape (n_samples,)
    >>> rec.dt              # seconds
    >>> rec.header.srate    # Hz
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    header = _read_header(path)
    signal = _read_data(path, header)

    return SSDRecord(
        header=header,
        signal=signal,
        dt=header.dt_us * 1e-6,
        path=path,
    )


def write(
    path: str | Path,
    signal: np.ndarray,
    dt_us: int,
    *,
    gain: float = 1.0,
    filt: float = 1000.0,
    Emem: float = 0.0,
    temp: float = 0.0,
    title: str = "",
) -> Path:
    """Write a signal array as a CONSAM SSD file (version 1002).

    Parameters
    ----------
    path : str or Path
        Destination file path.  The ``.ssd`` extension is conventional
        but not enforced.
    signal : array-like, float
        Trace data in **picoamperes**, shape ``(n_samples,)``.
    dt_us : int
        Sampling interval in **microseconds**.  Must be in the range
        ``1 … 32767`` (the ``int16`` limit).  Common values: ``10``
        (100 kHz), ``20`` (50 kHz), ``50`` (20 kHz).
    gain : float, optional
        Transducer gain in V/pA (default ``1.0``).  Determines the ADC
        scaling: ``adc = signal_pA × gain × 6553.6``.
        Typical amplifier outputs: 50 mV/pA → ``gain = 0.05``;
        100 mV/pA → ``gain = 0.1``.
    filt : float, optional
        Low-pass filter cut-off in Hz stored in the header for
        reference (default ``1000.0``).
    Emem : float, optional
        Membrane potential in mV (default ``0.0``).
    temp : float, optional
        Bath temperature in °C (default ``0.0``).
    title : str, optional
        Free-text title stored in the header (max 70 characters).

    Returns
    -------
    Path
        Resolved path of the written file.

    Raises
    ------
    ValueError
        If *signal* is empty, or if *dt_us* is outside ``[1, 32767]``,
        or if any scaled sample overflows ``int16`` range.

    Notes
    -----
    Samples are quantised to ``int16`` using the formula::

        int16_sample = round(signal_pA × gain × 6553.6)

    The complementary calibration factor stored in the header is::

        calfac = 1.0 / (gain × 6553.6)

    so that ``read()`` recovers picoamperes transparently.

    Examples
    --------
    >>> import numpy as np
    >>> from dcio.formats.ssd import write, read
    >>> signal = np.zeros(1000)
    >>> write("out.ssd", signal, dt_us=20, gain=0.05)
    PosixPath('out.ssd')
    """
    path = Path(path)
    signal = np.asarray(signal, dtype=np.float64)

    if signal.size == 0:
        raise ValueError("signal must not be empty.")
    if not (1 <= dt_us <= 32767):
        raise ValueError(
            f"dt_us must be in [1, 32767] µs; got {dt_us}."
        )

    scaled = signal * gain * 6553.6
    if scaled.max() > 32767 or scaled.min() < -32768:
        raise ValueError(
            "Scaled signal overflows int16. Reduce gain or signal amplitude.\n"
            f"  signal range : {signal.min():.3g} … {signal.max():.3g} pA\n"
            f"  scaled range : {scaled.min():.3g} … {scaled.max():.3g}"
        )

    adc = np.round(scaled).astype(np.int16)
    calfac = 1.0 / (gain * 6553.6)
    srate = 1e6 / dt_us
    now = datetime.datetime.now()

    _title    = f"{title:<70}"[:70]
    _date     = now.strftime("%d-%b-%Y")   # 11 chars: DD-Mon-YYYY
    _time     = now.strftime("%H-%M-%S")   # 8 chars:  HH-MM-SS
    _expdate  = _date
    _defname  = f"{'dcio':<6}"[:6]
    _tapeid   = f"{'dcio':<24}"[:24]
    _cs       = f"{'H':<3}"[:3]

    with open(path, "wb") as fout:
        fout.write(struct.pack("<h", 1002))                    # version     2 B
        fout.write(_title.encode("ascii"))                     # title      70 B
        fout.write(_date.encode("ascii"))                      # date       11 B
        fout.write(_time.encode("ascii"))                      # time        8 B
        fout.write(struct.pack("<h", int(dt_us)))              # idt         2 B
        fout.write(struct.pack("<i", _HEADER_SIZE))            # ioff        4 B
        fout.write(struct.pack("<i", len(adc)))                # ilen        4 B
        fout.write(struct.pack("<h", 1))                       # inc         2 B
        fout.write(struct.pack("<hh", 0, 0))                   # id1, id2    4 B
        fout.write(_cs.encode("ascii"))                        # cs          3 B
        fout.write(struct.pack("<f", calfac))                  # calfac      4 B
        fout.write(struct.pack("<f", srate))                   # srate       4 B
        fout.write(struct.pack("<f", filt))                    # filt        4 B
        fout.write(struct.pack("<ff", 0.0, 0.0))               # filt1, cf1  8 B
        fout.write(_expdate.encode("ascii"))                   # expdate    11 B
        fout.write(_defname.encode("ascii"))                   # defname     6 B
        fout.write(_tapeid.encode("ascii"))                    # tapeID     24 B
        fout.write(struct.pack("<ii", 0, 1))                   # ipatch/npa  8 B
        fout.write(struct.pack("<ff", Emem, temp))             # Emem, temp  8 B
        # zero-pad to byte 512
        fout.seek(_HEADER_SIZE)
        adc.tofile(fout)

    return path.resolve()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decode(raw: bytes) -> str:
    """Decode a fixed-width ASCII field, stripping nulls and dots padding."""
    return raw.replace(b"\x00", b"").decode("ascii", errors="replace").strip(".")


def _read_header(path: Path) -> SSDHeader:
    """Parse the 512-byte CONSAM header."""
    _f = array("f")
    _i = array("i")
    _h = array("h")

    with open(path, "rb") as fh:
        _h.fromfile(fh, 1); version = _h.pop()
        if version not in (1001, 1002):
            raise ValueError(
                f"Unrecognised SSD version {version!r} in {path}. "
                "Expected 1001 or 1002."
            )

        title   = _decode(fh.read(70))
        date    = _decode(fh.read(11))
        time_   = _decode(fh.read(8))

        _h.fromfile(fh, 1); dt_us = int(_h.pop())
        _i.fromfile(fh, 1); ioff  = _i.pop()
        _i.fromfile(fh, 1); ilen  = _i.pop()
        _h.fromfile(fh, 1); inc   = _h.pop()
        _h.fromfile(fh, 1); id1   = _h.pop()
        _h.fromfile(fh, 1); id2   = _h.pop()
        cs      = _decode(fh.read(3))

        _f.fromfile(fh, 1); calfac  = _f.pop()
        _f.fromfile(fh, 1); srate   = _f.pop()
        _f.fromfile(fh, 1); filt    = _f.pop()
        _f.fromfile(fh, 1); filt1   = _f.pop()
        _f.fromfile(fh, 1); calfac1 = _f.pop()

        expdate = _decode(fh.read(11))
        defname = _decode(fh.read(6))
        tapeID  = _decode(fh.read(24))

        _i.fromfile(fh, 1); ipatch = _i.pop()
        _i.fromfile(fh, 1); npatch = _i.pop()
        _f.fromfile(fh, 1); Emem   = _f.pop()
        _f.fromfile(fh, 1); temp   = _f.pop()

    return SSDHeader(
        version=version,
        n_samples=ilen,
        dt_us=dt_us,
        calfac=calfac,
        srate=srate,
        filt=filt,
        title=title,
        date=date,
        time=time_,
        filt1=filt1,
        calfac1=calfac1,
        expdate=expdate,
        defname=defname,
        tapeID=tapeID,
        ipatch=ipatch,
        npatch=npatch,
        Emem=Emem,
        temp=temp,
        inc=inc,
        id1=id1,
        id2=id2,
        cs=cs,
        data_offset=ioff,
    )


def _read_data(path: Path, header: SSDHeader) -> np.ndarray:
    """Read ADC samples and return signal in picoamperes."""
    with open(path, "rb") as fh:
        fh.seek(header.data_offset)
        signal = np.fromfile(fh, dtype="<i2").astype(np.float64) * header.calfac
    return signal
