"""
dcio.formats.scan_ini
======================
Read SCAN.INI binary settings files.

SCAN.INI is written by the SCAN single-channel analysis program
(DCProgs, UCL Colquhoun lab) to persist analysis parameters between
sessions.  It is a *direct-access unformatted* Fortran binary record
(no record markers) stored on a 256-byte or 512-byte boundary.

The 256-byte format is the original; the 512-byte format was introduced
later and adds three extra fields (``scritvar``, ``smultmin``,
``stpfac``).  When SCAN encounters a 256-byte file it converts it to
512 bytes automatically; this reader handles both.

Binary layout
-------------
All multi-byte integers and floats are **little-endian** (x86 / Lahey
Fortran on DOS/Windows).  Fortran ``LOGICAL*4`` fields are stored as a
4-byte integer: ``0`` = ``.FALSE.``, non-zero = ``.TRUE.``.
Fortran ``CHARACTER*N`` fields are stored as *N* raw bytes, typically
space-padded with no NUL terminator (though NUL bytes may appear in
practice when a field is shorter than its declared width).

Field layout (tight-packed, no alignment padding)::

    offset   size  Fortran type    field
    ------   ----  ------------    -----
         0      1  CHARACTER*1     savin
         1      4  INTEGER*4       nbuf
         5      4  INTEGER*4       novlap
         9      4  LOGICAL*4       opendown
        13      4  LOGICAL*4       invert
        17      4  REAL*4          smult
        21      4  INTEGER*4       ntrig
        25      4  REAL*4          dgain
        29      4  INTEGER*4       iboff
        33      4  INTEGER*4       nwrit
        37      4  INTEGER*4       iexp
        41      4  REAL*4          bdisp
        45      4  INTEGER*4       isub
        49      4  REAL*4          xtrig
        53      4  LOGICAL*4       usepots
        57      2  CHARACTER*2     ndevdat
        59      4  LOGICAL*4       disp
        63      4  INTEGER*4       nsetup
        67     20  CHARACTER*20    filtfile
        87      4  INTEGER*4       navtest
        91      4  INTEGER*4       iautosub
        95      4  LOGICAL*4       disptran
        99      4  LOGICAL*4       dispderiv
       103      4  LOGICAL*4       dispguess
       107      4  REAL*4          tsfac
       111      4  REAL*4          tlfac
       115      4  REAL*4          tcfac
       119      4  REAL*4          facjump
       123      4  REAL*4          ampfac
       127      4  REAL*4          errfac
       131      4  REAL*4          derivfac
       135      4  REAL*4          confac
       139     30  CHARACTER*30    adcfil
       169      4  INTEGER*4       nampmark
       173     40  INTEGER*4[10]   iamark
       213      4  REAL*4          tmin
       217      4  LOGICAL*4       cjump
       221      4  INTEGER*4       izoom
       225      4  REAL*4          fcz
       229      4  REAL*4          ampz
       233      4  INTEGER*4       itsimp
       237      4  INTEGER*4       minmeth
       241      4  INTEGER*4       nbasemin
       245      4  INTEGER*4       iscrit
       --- 512-byte format only ---
       249      4  LOGICAL*4       scritvar
       253      4  REAL*4          smultmin
       257      4  REAL*4          stpfac

The layout was reverse-engineered from the Fortran source files
``inscan.for`` and ``scan.for`` (DCProgs, UCL) and validated against a
real SCAN.INI binary by matching the known string anchor
``adcfil`` at offset 139.

Public API
----------
* :func:`read` – parse a SCAN.INI file and return a
  :class:`ScanIniRecord`.
* :class:`ScanIniRecord` – dataclass with all stored fields.

Examples
--------
::

    from dcio.formats.scan_ini import read

    ini = read("examples/ini/SCAN.INI")
    print(ini.adcfil)          # path to the ADC data file
    print(ini.opendown)        # True  → openings are downward deflections
    print(ini.tmin)            # minimum interval in µs (e.g. 15.0)
    print(ini.iscrit)          # 1=fraction-of-amplitude, 2=multiple-of-RMS
    print(ini.smult)           # threshold multiplier
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Expected file sizes (bytes).
INI_SIZE_OLD = 256
INI_SIZE_NEW = 512

#: ``iscrit`` values.
ISCRIT_FRACTION_AMP = 1  # threshold = smult * mean_amplitude
ISCRIT_MULTIPLE_RMS = 2  # threshold = smult * RMS_noise

# ---------------------------------------------------------------------------
# Internal struct format – derived from inscan.for layout
# ---------------------------------------------------------------------------

# Each entry: (field_name, offset, fmt)
# fmt is a struct.pack format character or a tuple (N, 'N_chars') for strings.
# Offsets are byte offsets into the 512-byte record (0-based).
#
# NB: We parse each field individually because the layout is not uniform
# (mixed chars, int32, float32, logical4, char arrays).

_CHAR_FIELDS = {
    # name: (offset, n_bytes)
    "savin":    (0,   1),
    "ndevdat":  (57,  2),
    "filtfile": (67,  20),
    "adcfil":   (139, 30),
}

_INT32_FIELDS = {
    # name: offset
    "nbuf":     1,
    "novlap":   5,
    "ntrig":    21,
    "iboff":    29,
    "nwrit":    33,
    "iexp":     37,
    "isub":     45,
    "nsetup":   63,
    "navtest":  87,
    "iautosub": 91,
    "nampmark": 169,
    "izoom":    221,
    "itsimp":   233,
    "minmeth":  237,
    "nbasemin": 241,
    "iscrit":   245,
}

_FLOAT32_FIELDS = {
    # name: offset
    "smult":    17,
    "dgain":    25,
    "bdisp":    41,
    "xtrig":    49,
    "tsfac":    107,
    "tlfac":    111,
    "tcfac":    115,
    "facjump":  119,
    "ampfac":   123,
    "errfac":   127,
    "derivfac": 131,
    "confac":   135,
    "tmin":     213,
    "fcz":      225,
    "ampz":     229,
}

_LOGICAL4_FIELDS = {
    # name: offset  (Fortran LOGICAL*4: 0 = False, non-zero = True)
    "opendown":  9,
    "invert":    13,
    "usepots":   53,
    "disp":      59,
    "disptran":  95,
    "dispderiv": 99,
    "dispguess": 103,
    "cjump":     217,
}

# iamark: 10 × INTEGER*4 starting at offset 173
_IAMARK_OFFSET = 173
_IAMARK_COUNT = 10

# Extra fields present only in the 512-byte (new) format
_FLOAT32_FIELDS_512 = {
    "smultmin": 253,
    "stpfac":   257,
}
_LOGICAL4_FIELDS_512 = {
    "scritvar": 249,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_char(data: bytes, offset: int, n: int) -> str:
    """Return a stripped ASCII string from *n* bytes starting at *offset*."""
    raw = data[offset: offset + n]
    return raw.rstrip(b"\x00 ").decode("ascii", errors="replace")


def _read_i32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _read_f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def _read_l4(data: bytes, offset: int) -> bool:
    """Fortran LOGICAL*4: non-zero integer → True."""
    val = struct.unpack_from("<i", data, offset)[0]
    return val != 0


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScanIniRecord:
    """Parsed contents of a SCAN.INI settings file.

    Parameters marked **[key]** are the most relevant for re-analysis or
    for understanding how a recording was idealised.

    Attributes
    ----------
    adcfil : str
        **[key]** Path to the ADC data file (CONSAM or CJUMP) as stored
        when SCAN.INI was last saved.  Up to 30 characters.
    opendown : bool
        **[key]** ``True`` if channel openings appear as *downward*
        deflections in the raw trace.  Must be ``True`` for the SCAN
        algorithm to work correctly in normal operation.
    smult : float
        **[key]** Threshold multiplier.  Interpretation depends on
        :attr:`iscrit`:

        * ``iscrit == 1`` → threshold = ``smult × mean_amplitude``
          (fraction; typical range 0.10–0.20)
        * ``iscrit == 2`` → threshold = ``smult × RMS_noise``
          (multiple; typical range 3–6)
    iscrit : int
        **[key]** Threshold criterion: ``1`` = fraction of mean amplitude
        (default), ``2`` = multiple of RMS noise.  Values outside 1–2
        are corrected to 1 by SCAN on load.
    tmin : float
        **[key]** Minimum interval length (µs) below which a re-fit
        without short events is offered.  Default 15 µs.
    cjump : bool
        **[key]** ``True`` if data come from a concentration-jump
        (CJUMP) experiment rather than a continuous CONSAM file.
    ntrig : int
        Number of consecutive points above the threshold required to
        confirm a transition.  Default 2.
    nbuf : int
        Number of data points held in memory per section.  Default
        131 072.
    novlap : int
        Overlap between adjacent data sections (points).  Default 2048.
    invert : bool
        ``True`` if the trace was inverted for display (not the same as
        :attr:`opendown`).
    dgain : float
        Display gain factor.
    iboff : int
        Baseline offset for display (ADC units).
    nwrit : int
        Write interval to disc every *nwrit* transitions.
    iexp : int
        Internal expansion factor (legacy field).
    bdisp : float
        Baseline position as a fraction of the Y axis (e.g. 0.75).
    isub : int
        Sub-level auto-fit mode (0, 1, or 2).
    xtrig : float
        Trigger position as a fraction of the X axis.
    usepots : bool
        ``True`` if potentiometers were used for display control
        (legacy hardware option; always ``False`` in practice).
    ndevdat : str
        Default device/disk identifier (e.g. ``'C:'``).
    disp : bool
        ``True`` if display-only mode (no step-response fitting).
    nsetup : int
        Filter setup index; negative = verify against stored value.
    filtfile : str
        Name of the filter definition file (up to 20 characters).
    navtest : int
        Number of points averaged before accepting a baseline level.
    iautosub : int
        Auto-sublevel fitting mode (1 or 2).
    disptran : bool
        ``True`` → show guessed transition positions on screen.
    dispderiv : bool
        ``True`` → show first derivative on screen.
    dispguess : bool
        ``True`` → show guesses before fitting.
    tsfac : float
        Multiple of risetime defining a 'short' interval.
    tlfac : float
        Multiple of risetime defining a 'long' interval.
    tcfac : float
        Multiple of risetime defining a 'close' transition.
    facjump : float
        Fraction of filter length to jump after a transition.
    ampfac : float
        Fraction of full amplitude below which amplitude differences
        are treated as zero.
    errfac : float
        Convergence criterion for the Simplex minimiser.
    derivfac : float
        Multiplier of derivative SD for detecting inflections.
    confac : float
        Simplex contraction factor.
    nampmark : int
        Number of amplitude markers to display (0 = none).
    iamark : list of int
        ADC-unit values of amplitude markers (10 slots; unused slots
        are 0).
    izoom : int
        Default zoom factor (must be a power of 2).
    fcz : float
        Filter cut-off frequency when zoomed (kHz).
    ampz : float
        'Full amplitude' for event detection when zoomed (pA).
    itsimp : int
        Number of Simplex iterations before switching to DFPMIN.
    minmeth : int
        Minimisation method: 1 = Simplex only, 2 = Simplex then DFPMIN,
        3 = DFPMIN only.
    nbasemin : int
        Minimum number of baseline points required for amplitude
        estimation.
    scritvar : bool or None
        ``True`` → use reduced threshold for small-amplitude events.
        ``None`` if the file is the old 256-byte format.
    smultmin : float or None
        Minimum threshold multiplier (as multiple of RMS) when
        :attr:`scritvar` is active.  ``None`` for 256-byte files.
    stpfac : float or None
        Simplex initial step-size factor.  ``None`` for 256-byte files.
    savin : str
        ``'Y'`` if parameters were saved on exit, ``'N'`` otherwise.
    path : Path or None
        Source file path (set by :func:`read`).
    ini_size : int
        Actual file size in bytes (256 or 512).
    """

    # --- Key analysis fields ---
    adcfil: str = ""
    opendown: bool = True
    smult: float = 0.14
    iscrit: int = 1
    tmin: float = 15.0
    cjump: bool = False
    ntrig: int = 2

    # --- Buffer / I/O ---
    nbuf: int = 131072
    novlap: int = 2048
    nwrit: int = 100

    # --- Display / gain ---
    invert: bool = False
    dgain: float = 1.0
    iboff: int = 0
    bdisp: float = 0.75
    isub: int = 0
    xtrig: float = 0.2
    usepots: bool = False
    disp: bool = False

    # --- Device / file ---
    ndevdat: str = "C:"
    nsetup: int = 0
    filtfile: str = ""
    savin: str = "N"

    # --- Fitting parameters ---
    iexp: int = 0
    navtest: int = 1
    iautosub: int = 1
    disptran: bool = False
    dispderiv: bool = False
    dispguess: bool = True
    tsfac: float = 2.0
    tlfac: float = 3.0
    tcfac: float = 4.0
    facjump: float = 0.6
    ampfac: float = 0.05
    errfac: float = 0.005
    derivfac: float = 3.0
    confac: float = 0.5

    # --- Amplitude markers ---
    nampmark: int = 0
    iamark: List[int] = field(default_factory=lambda: [0] * 10)

    # --- Zoom ---
    izoom: int = 1
    fcz: float = 1.0
    ampz: float = 2.5

    # --- Minimiser ---
    itsimp: int = 1200
    minmeth: int = 2
    nbasemin: int = 10

    # --- 512-byte only (None for old 256-byte files) ---
    scritvar: bool | None = None
    smultmin: float | None = None
    stpfac: float | None = None

    # --- Metadata ---
    path: Path | None = None
    ini_size: int = INI_SIZE_NEW


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read(path: Union[str, Path]) -> ScanIniRecord:
    """Parse a SCAN.INI binary file.

    Parameters
    ----------
    path:
        Path to the SCAN.INI file.

    Returns
    -------
    ScanIniRecord
        Parsed record with all fields populated.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file size is neither 256 nor 512 bytes (not a valid
        SCAN.INI file).

    Examples
    --------
    ::

        from dcio.formats.scan_ini import read, ISCRIT_FRACTION_AMP

        ini = read("examples/ini/SCAN.INI")
        print(f"Data file : {ini.adcfil}")
        print(f"Open down : {ini.opendown}")
        if ini.iscrit == ISCRIT_FRACTION_AMP:
            print(f"Threshold : {ini.smult * 100:.1f}% of mean amplitude")
        else:
            print(f"Threshold : {ini.smult:.1f} × RMS noise")
    """
    path = Path(path)
    data = path.read_bytes()
    size = len(data)

    if size not in (INI_SIZE_OLD, INI_SIZE_NEW):
        raise ValueError(
            f"{path}: expected {INI_SIZE_OLD} or {INI_SIZE_NEW} bytes, "
            f"got {size}"
        )

    rec = ScanIniRecord(path=path, ini_size=size)

    # Character fields
    for name, (offset, n) in _CHAR_FIELDS.items():
        setattr(rec, name, _read_char(data, offset, n))

    # Integer fields
    for name, offset in _INT32_FIELDS.items():
        setattr(rec, name, _read_i32(data, offset))

    # Float fields
    for name, offset in _FLOAT32_FIELDS.items():
        setattr(rec, name, _read_f32(data, offset))

    # Logical fields
    for name, offset in _LOGICAL4_FIELDS.items():
        setattr(rec, name, _read_l4(data, offset))

    # iamark array: 10 × int32 starting at offset 173
    rec.iamark = [
        _read_i32(data, _IAMARK_OFFSET + i * 4)
        for i in range(_IAMARK_COUNT)
    ]

    # 512-byte-only fields
    if size == INI_SIZE_NEW:
        for name, offset in _LOGICAL4_FIELDS_512.items():
            setattr(rec, name, _read_l4(data, offset))
        for name, offset in _FLOAT32_FIELDS_512.items():
            setattr(rec, name, _read_f32(data, offset))

    return rec
