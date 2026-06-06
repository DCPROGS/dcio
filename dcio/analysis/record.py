"""Single-channel record analysis: time resolution and open/shut period extraction.

Typical workflow::

    from dcio.formats.scn import read
    from dcio.analysis.record import from_scn

    rec   = read("path/to/file.scn")
    scr   = from_scn(rec, tres=40e-6)   # 40 µs dead time
    print(scr.periods.n_open, "open periods")
    print(scr.periods.open_intervals.mean() * 1000, "ms mean open")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from dcio.formats.scn import SCNRecord


_FLAG_UNUSABLE = 8


# ---------------------------------------------------------------------------
# Periods — result of set_periods
# ---------------------------------------------------------------------------


@dataclass
class Periods:
    """Open/shut period lists in alternating order.

    ``intervals[0::2]`` are open periods; ``intervals[1::2]`` are shut periods.
    The sequence starts with an open period and ends with an open period
    (trailing shuts are trimmed).
    """

    intervals: np.ndarray   # seconds
    amplitudes: np.ndarray  # pA  (0.0 for shut periods)
    flags: np.ndarray       # int8

    @property
    def open_intervals(self) -> np.ndarray:
        """Open period durations (s)."""
        return self.intervals[0::2]

    @property
    def open_amplitudes(self) -> np.ndarray:
        """Time-weighted mean amplitude for each open period (pA)."""
        return self.amplitudes[0::2]

    @property
    def open_flags(self) -> np.ndarray:
        """Flags for each open period (bit 3 = unusable)."""
        return self.flags[0::2]

    @property
    def shut_intervals(self) -> np.ndarray:
        """Shut period durations (s)."""
        return self.intervals[1::2]

    @property
    def shut_amplitudes(self) -> np.ndarray:
        """Amplitudes for each shut period (always 0.0 pA)."""
        return self.amplitudes[1::2]

    @property
    def shut_flags(self) -> np.ndarray:
        """Flags for each shut period."""
        return self.flags[1::2]

    @property
    def n_open(self) -> int:
        """Number of open periods."""
        return len(self.open_intervals)

    @property
    def n_shut(self) -> int:
        """Number of shut periods."""
        return len(self.shut_intervals)

    def __repr__(self) -> str:
        if self.n_open == 0:
            return "Periods(empty)"
        return (
            f"Periods(n_open={self.n_open}, n_shut={self.n_shut}, "
            f"mean_open={self.open_intervals.mean() * 1e3:.3f} ms, "
            f"mean_shut={self.shut_intervals.mean() * 1e3:.3f} ms)"
            if self.n_shut > 0
            else f"Periods(n_open={self.n_open})"
        )


# ---------------------------------------------------------------------------
# SingleChannelRecord — top-level result object
# ---------------------------------------------------------------------------


@dataclass
class SingleChannelRecord:
    """Idealised single-channel record with dead-time resolution applied.

    Build via :func:`from_scn` rather than instantiating directly.
    """

    intervals: np.ndarray    # raw, seconds
    amplitudes: np.ndarray   # raw, pA
    flags: np.ndarray        # raw, int8
    record_type: str         # 'simulated' or 'experimental'
    tres: float              # seconds (0 = no resolution applied)
    path: Optional[Path]

    resolved_intervals: np.ndarray   # after impose_resolution, seconds
    resolved_amplitudes: np.ndarray  # pA
    resolved_flags: np.ndarray       # int8

    periods: Periods

    @property
    def n_raw(self) -> int:
        """Number of intervals in the raw record."""
        return len(self.intervals)

    @property
    def n_resolved(self) -> int:
        """Number of intervals after dead-time resolution."""
        return len(self.resolved_intervals)

    @property
    def open_periods(self) -> np.ndarray:
        """Open period durations (s); shortcut for ``periods.open_intervals``."""
        return self.periods.open_intervals

    @property
    def shut_periods(self) -> np.ndarray:
        """Shut period durations (s); shortcut for ``periods.shut_intervals``."""
        return self.periods.shut_intervals

    def __repr__(self) -> str:
        name = self.path.name if self.path else "in-memory"
        lines = [
            f"SingleChannelRecord({name!r})",
            f"  record_type : {self.record_type}",
            f"  tres        : {self.tres * 1e6:.1f} µs",
            f"  n_raw       : {self.n_raw}",
            f"  n_resolved  : {self.n_resolved}",
        ]
        if self.periods.n_open > 0:
            lines.append(
                f"  open periods: {self.periods.n_open}"
                f"  mean {self.open_periods.mean() * 1e3:.3f} ms"
            )
        if self.periods.n_shut > 0:
            lines.append(
                f"  shut periods: {self.periods.n_shut}"
                f"  mean {self.shut_periods.mean() * 1e3:.3f} ms"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# impose_resolution
# ---------------------------------------------------------------------------


def impose_resolution(
    intervals: np.ndarray,
    amplitudes: np.ndarray,
    flags: np.ndarray,
    tres: float,
    *,
    record_type: str = "experimental",
    badopen: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a dead time to an idealised interval list.

    Intervals shorter than *tres* are concatenated into their neighbours.
    Open periods accumulate a time-weighted mean amplitude; shut periods
    accumulate total duration.  The first and last intervals in each
    concatenated group must be resolvable, though they may be flagged bad.

    Parameters
    ----------
    intervals : ndarray
        Interval durations in seconds.
    amplitudes : ndarray
        Amplitudes in pA (0 = shut, non-zero = open).
    flags : ndarray
        Flag bytes; bit 3 set (value ≥ 8) means unusable.
    tres : float
        Dead time in seconds.  Intervals shorter than *tres* are
        concatenated into adjacent intervals.  Use 0 for no filtering.
    record_type : {'experimental', 'simulated'}
        Experimental records compute weighted-mean amplitude across
        sub-conductance levels.  Simulated records use the raw amplitude
        directly (binary 0/1).
    badopen : float
        Any open interval longer than this (seconds) is flagged unusable.
        Set to 0 (default) to disable.

    Returns
    -------
    intervals : ndarray, float64, seconds
    amplitudes : ndarray, float64, pA
    flags : ndarray, int8

    Raises
    ------
    ValueError
        If no resolvable usable interval exists in the record.
    """
    flags = flags.copy().astype(np.int32)
    flags[intervals < 0] |= _FLAG_UNUSABLE

    usable = np.where((intervals > tres) & (flags < _FLAG_UNUSABLE))[0]
    if len(usable) == 0:
        raise ValueError(
            f"No resolvable usable interval (tres={tres * 1e6:.1f} µs)."
        )

    n = int(usable[0])
    sim = record_type == "simulated"
    N = len(intervals)

    rtint: list[float] = []
    rampl: list[float] = []
    rprop: list[int] = []

    ttemp = float(intervals[n])
    otemp = int(flags[n])
    isopen: bool = bool(amplitudes[n] != 0)
    if not isopen:
        atemp = 0.0
    elif sim:
        atemp = float(amplitudes[n])
    else:
        atemp = float(amplitudes[n]) * ttemp
    n += 1

    while n < N:
        t = float(intervals[n])
        a = float(amplitudes[n])
        f = int(flags[n])

        if t < tres:  # unresolvable — concatenate
            # Special case: very last interval is a brief shut that terminates
            # an open run.  Emit the open, then start the short shut sentinel.
            if n == N - 1 and a == 0.0 and isopen:
                rtint.append(ttemp)
                rampl.append(atemp / ttemp)
                rprop.append(otemp)
                isopen = False
                ttemp = t
                atemp = 0.0
                otemp = _FLAG_UNUSABLE
            else:
                ttemp += t
                if f >= _FLAG_UNUSABLE:
                    otemp = f
                if isopen:
                    atemp += a * t
        else:  # resolvable
            if a == 0.0:  # shutting
                if not isopen:  # extend existing shut
                    ttemp += t
                    if f >= _FLAG_UNUSABLE:
                        otemp = f
                else:  # open → shut transition
                    if badopen > 0 and ttemp > badopen:
                        otemp = _FLAG_UNUSABLE
                    rtint.append(ttemp)
                    rampl.append(atemp if sim else atemp / ttemp)
                    rprop.append(otemp)
                    ttemp = t
                    otemp = f
                    atemp = 0.0
                    isopen = False
            else:  # opening
                if not isopen:  # shut → open transition
                    rtint.append(ttemp)
                    rampl.append(0.0)
                    rprop.append(otemp)
                    ttemp = t
                    otemp = f
                    atemp = a if sim else a * t
                    isopen = True
                else:  # extend existing open
                    if sim:
                        ttemp += t
                        if f >= _FLAG_UNUSABLE:
                            otemp = f
                    else:
                        cur_avg = atemp / ttemp
                        if math.fabs(cur_avg - a) <= 1e-5:  # same sub-level
                            ttemp += t
                            atemp += a * t
                            if f >= _FLAG_UNUSABLE:
                                otemp = f
                        else:  # sub-level change → emit, restart
                            if badopen > 0 and ttemp > badopen:
                                otemp = _FLAG_UNUSABLE
                            rtint.append(ttemp)
                            rampl.append(atemp / ttemp)
                            rprop.append(otemp)
                            ttemp = t
                            otemp = f
                            atemp = a * t
        n += 1

    # Emit final accumulated interval.  An unfinished opening gets duration
    # –1 as a sentinel; the last interval is always flagged unusable.
    if isopen:
        rtint.append(-1.0)
        rampl.append(atemp if sim else atemp / ttemp)
    else:
        rtint.append(ttemp)
        rampl.append(0.0)
    rprop.append(_FLAG_UNUSABLE)

    return (
        np.array(rtint, dtype=np.float64),
        np.array(rampl, dtype=np.float64),
        np.array(rprop, dtype=np.int8),
    )


# ---------------------------------------------------------------------------
# set_periods
# ---------------------------------------------------------------------------


def set_periods(
    intervals: np.ndarray,
    amplitudes: np.ndarray,
    flags: np.ndarray,
) -> Periods:
    """Group resolved intervals into alternating open/shut periods.

    Multiple consecutive openings at different sub-conductance levels (as can
    appear in experimental records after :func:`impose_resolution`) are merged
    into one open period with a time-weighted mean amplitude.  A single
    unusable opening (flag bit 3) contaminates the entire open period.

    The output ``Periods.intervals`` array always starts with an open period
    and alternates [open, shut, open, shut, …].  Trailing shuts and the
    final unusable sentinel produced by :func:`impose_resolution` are trimmed.

    Parameters
    ----------
    intervals, amplitudes, flags
        Output of :func:`impose_resolution`.

    Returns
    -------
    Periods
        Empty ``Periods`` (all arrays length 0) if no valid periods remain
        after trimming.
    """
    rtint = list(intervals.astype(float))
    rampl = list(amplitudes.astype(float))
    rprop = list(flags.astype(int))

    # Trim leading shut.
    if rtint and rampl[0] == 0.0:
        rtint.pop(0); rampl.pop(0); rprop.pop(0)
    # Trim trailing sentinel (unfinished opening: t < 0).
    if rtint and rtint[-1] < 0.0:
        rtint.pop(); rampl.pop(); rprop.pop()
    # Trim trailing shuts.
    while rtint and rampl[-1] == 0.0:
        rtint.pop(); rampl.pop(); rprop.pop()

    if not rtint:
        empty = np.array([], dtype=np.float64)
        return Periods(empty, empty, np.array([], dtype=np.int8))

    pint: list[float] = []
    pamp: list[float] = []
    popt: list[int] = []

    # Accumulator for the current open period.
    oint = rtint[0]
    oamp = rampl[0] * rtint[0]  # weighted amplitude-time product
    oopt = rprop[0]
    n = 1

    while n < len(rtint):
        if rampl[n] != 0.0:  # opening
            oint += rtint[n]
            oamp += rampl[n] * rtint[n]
            if rprop[n] >= _FLAG_UNUSABLE:
                oopt = _FLAG_UNUSABLE
            if n == len(rtint) - 1:  # last interval: emit accumulated open
                pamp.append(oamp / oint)
                pint.append(oint)
                popt.append(oopt)
        else:  # shutting
            if oamp == 0.0 and oopt < _FLAG_UNUSABLE:
                # Two consecutive shuts (defensive; shouldn't happen after
                # impose_resolution) — extend preceding shut period.
                if pint:
                    pint[-1] += rtint[n]
            elif oopt >= _FLAG_UNUSABLE:
                # Bad open period: mark preceding shut as unusable and discard
                # the bad open without emitting it.
                if popt:
                    popt[-1] = _FLAG_UNUSABLE
                oint = 0.0; oamp = 0.0; oopt = 0
            else:  # good open period terminated by a shut
                pamp.append(oamp / oint)
                pint.append(oint)
                popt.append(oopt)
                oint = 0.0; oamp = 0.0; oopt = 0
                pamp.append(0.0)
                pint.append(rtint[n])
                popt.append(rprop[n])
        n += 1

    # If the open accumulator still holds data (single open period with no
    # following shut, or record ends on an open), emit it.
    if oint > 0.0 and not pint:
        pamp.append(oamp / oint)
        pint.append(oint)
        popt.append(oopt)

    return Periods(
        intervals=np.array(pint, dtype=np.float64),
        amplitudes=np.array(pamp, dtype=np.float64),
        flags=np.array(popt, dtype=np.int8),
    )


# ---------------------------------------------------------------------------
# from_scn — main entry point
# ---------------------------------------------------------------------------


def from_scn(
    rec: SCNRecord,
    tres: float = 0.0,
    *,
    badopen: float = 0.0,
) -> SingleChannelRecord:
    """Build a :class:`SingleChannelRecord` from an :class:`~dcio.formats.scn.SCNRecord`.

    Parameters
    ----------
    rec : SCNRecord
        Loaded by :func:`dcio.formats.scn.read`.
    tres : float
        Dead time in seconds.  Intervals shorter than *tres* are concatenated
        into adjacent intervals.  0 = no dead-time filtering.
    badopen : float
        Openings longer than this (seconds) are flagged unusable.
        0 = disabled.

    Returns
    -------
    SingleChannelRecord

    Examples
    --------
    >>> from dcio.formats.scn import read
    >>> from dcio.analysis.record import from_scn
    >>> rec = read("examples/scn/041208S6.scn")
    >>> scr = from_scn(rec, tres=40e-6)
    >>> scr.periods.n_open
    ...
    """
    ri, ra, rf = impose_resolution(
        rec.intervals,
        rec.amplitudes,
        rec.flags,
        tres,
        record_type=rec.header.record_type,
        badopen=badopen,
    )
    periods = set_periods(ri, ra, rf)
    return SingleChannelRecord(
        intervals=rec.intervals,
        amplitudes=rec.amplitudes,
        flags=rec.flags,
        record_type=rec.header.record_type,
        tres=tres,
        path=rec.path,
        resolved_intervals=ri,
        resolved_amplitudes=ra,
        resolved_flags=rf,
        periods=periods,
    )
