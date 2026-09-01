"""Burst segmentation of an idealised single-channel record.

A *burst* is a run of openings and short shuttings delimited by shut intervals
longer than a critical time ``tcrit``.  Segmentation sits directly on top of
:mod:`dcio.analysis.record`: it takes the resolved intervals a dead time
produced and cuts them into bursts.

Typical workflow::

    from dcio.formats.scn import read
    from dcio.analysis.record import from_scn
    from dcio.analysis.bursts import extract_bursts

    scr = from_scn(read("file.scn"), tres=25e-6)
    lengths, n_openings = extract_bursts(
        scr.resolved_intervals, scr.resolved_amplitudes,
        tcrit=4e-3, flags=scr.resolved_flags,
    )

or, equivalently, straight from the record::

    from dcio.analysis.bursts import bursts_from_record

    lengths, n_openings = bursts_from_record(scr, tcrit=4e-3)

Convention
----------
The convention is the one EKDIST states in ``Bursts.slice_bursts`` and dcpyps
followed:

1. no gap longer than ``tcrit`` is required before the first burst of a
   record -- the first defined opening is a valid burst start;
2. an unusable interval is a valid end of burst.

Both matter.  Time-course fitting in SCAN leaves the last interval of a record
with no defined length, flagged unusable; it still ends the burst before it.  A
leading interval may likewise be bad, and is discarded, but the first defined
opening after it starts a real burst.  Dropping the runs at both ends -- as
earlier versions of this code did -- loses two bursts from every record, which
at 30 uM in the Burzomato 2004 set is a third of the data.
"""

from __future__ import annotations

import numpy as np

from dcio.formats.scn import FLAG_UNUSABLE

__all__ = [
    "extract_bursts",
    "extract_burst_intervals",
    "bursts_from_record",
]


def _burst_segments(intervals, amplitudes, tcrit, flags=None):
    """Split a record into bursts and return them as (interval, amplitude) pairs.

    Shared by :func:`extract_bursts` and :func:`extract_burst_intervals`, which
    differ only in what they report about one and the same segmentation.

    Parameters
    ----------
    intervals, amplitudes : array_like
        Alternating interval record (ideal or apparent).  Durations in
        seconds, amplitudes in pA with 0 meaning shut.
    tcrit : float
        Critical shut time separating within- from between-burst gaps [s].
        Its magnitude only -- see :func:`bursts_from_record` for the sign
        convention that HJCFIT places on the same number.
    flags : array_like of int, optional
        Per-interval SCN property flags.  An interval is unusable when
        ``flags & FLAG_UNUSABLE`` is set.  Without them no interval is treated
        as unusable, which is right for a simulated record and wrong for an
        experimental one.

    Returns
    -------
    list of list of (float, float)
        One list of ``(interval, amplitude)`` pairs per burst, each starting
        and ending on an opening.
    """
    intervals = np.asarray(intervals, float)
    amplitudes = np.asarray(amplitudes, float)
    if intervals.size == 0:
        return []

    if flags is None:
        unusable = np.zeros(intervals.shape, dtype=bool)
    else:
        unusable = (np.asarray(flags, int) & FLAG_UNUSABLE) != 0

    # A burst ends at a between-burst gap or at an interval of unknown length.
    # An unusable interval has no measured duration, so it is never compared
    # with tcrit.
    # Strictly greater: tcrit is the time such that gaps *longer* than it are
    # between-burst.  EKDIST's Bursts.slice_bursts uses the same test, and a
    # shut interval exactly equal to tcrit is within-burst by that reading.
    separator = ((amplitudes == 0.0) & (intervals > tcrit) & ~unusable) | unusable

    # Trim to the first and last defined opening, so the record begins and
    # ends on one.  What lies outside is a shut interval or an unusable one,
    # and neither belongs to a burst.
    opening = (amplitudes != 0.0) & ~unusable
    if not opening.any():
        return []
    lo = int(np.argmax(opening))
    hi = int(len(opening) - np.argmax(opening[::-1]))

    raw, seg = [], []
    for t, a, sep in zip(intervals[lo:hi], amplitudes[lo:hi], separator[lo:hi]):
        if sep:
            if seg:
                raw.append(seg)
            seg = []
        else:
            seg.append((t, a))
    if seg:
        raw.append(seg)

    bursts = []
    for seg in raw:
        while seg and seg[0][1] == 0.0:             # trim leading shut
            seg = seg[1:]
        while seg and seg[-1][1] == 0.0:            # trim trailing shut
            seg = seg[:-1]
        if seg:
            bursts.append(seg)
    return bursts


def extract_bursts(intervals, amplitudes, tcrit, flags=None):
    """Split a record into bursts at shut intervals longer than ``tcrit``.

    Each burst is trimmed to start and end on an opening.

    Parameters
    ----------
    intervals, amplitudes : array_like
        Alternating interval record (ideal or apparent).
    tcrit : float
        Critical shut time separating within- from between-burst gaps [s].
    flags : array_like of int, optional
        Per-interval SCN property flags; see :func:`_burst_segments`.

    Returns
    -------
    lengths : ndarray
        Burst lengths [s] (first opening start to last opening end).
    n_openings : ndarray of int
        Number of (apparent) openings in each burst.

    See Also
    --------
    extract_burst_intervals : the same bursts, as interval sequences.
    """
    bursts = _burst_segments(intervals, amplitudes, tcrit, flags)
    lengths = [sum(t for t, _ in seg) for seg in bursts]
    nops = [sum(1 for _, a in seg if a != 0.0) for seg in bursts]
    return np.array(lengths), np.array(nops, dtype=int)


def extract_burst_intervals(intervals, amplitudes, tcrit, flags=None):
    """Split a record into bursts and return the intervals of each.

    The segmentation is that of :func:`extract_bursts`, which reduces each
    burst to its length and its number of openings.  Maximum-likelihood
    fitting of missed-events mechanisms needs the interval sequences
    themselves: the HJC likelihood is a product of matrices, one per interval,
    so the order and the individual durations both matter.

    Parameters
    ----------
    intervals, amplitudes : array_like
        Alternating interval record (ideal or apparent).
    tcrit : float
        Critical shut time separating within- from between-burst gaps [s].
    flags : array_like of int, optional
        Per-interval SCN property flags; see :func:`_burst_segments`.

    Returns
    -------
    list of ndarray
        One array of interval durations [s] per burst, alternating open and
        shut and both starting and ending with an opening -- so every array
        has odd length, which is what the missed-events likelihood requires.

    See Also
    --------
    extract_bursts : the same bursts, as lengths and opening counts.
    """
    return [np.array([t for t, _ in seg], dtype=float)
            for seg in _burst_segments(intervals, amplitudes, tcrit, flags)]


def bursts_from_record(record, tcrit, intervals_only=False):
    """Segment a :class:`~dcio.analysis.record.SingleChannelRecord` into bursts.

    Convenience over :func:`extract_bursts` that takes the intervals,
    amplitudes and flags from the record itself, so a caller can get neither
    the flags nor the input series wrong.

    Segmentation runs on the record's **periods**, not on its resolved
    intervals, and that distinction is not cosmetic.
    :func:`~dcio.analysis.record.impose_resolution` emits a fresh open
    interval whenever the fitted amplitude changes, so a record idealised with
    sub-conductance levels contains runs of consecutive open intervals: the
    experimental example shipped with dcio has 1770 such adjacencies in 3232
    resolved intervals.  Segmenting those directly gives bursts that do not
    alternate open/shut, and counts each sub-level as a separate opening.
    :func:`~dcio.analysis.record.set_periods` merges them, which is what makes
    the burst an alternating sequence the missed-events likelihood can consume.

    Burst count and burst lengths are identical either way; the number of
    openings per burst is not.

    Parameters
    ----------
    record : SingleChannelRecord
        A record with a dead time already applied.
    tcrit : float
        Critical shut time [s].  Only its magnitude is used.  The sign carries
        no meaning here, but it does downstream: HJCFIT reads a negative
        ``tcrit`` as a flag selecting equilibrium vectors (Colquhoun & Hawkes
        1982) over CHS vectors (Colquhoun, Hawkes & Srodzinski 1996), and the
        same number is passed to both.  Taking the magnitude here means a
        caller can hand the same value to either without thinking about it.
    intervals_only : bool, default False
        Return the per-burst interval sequences instead of lengths and
        opening counts.

    Returns
    -------
    lengths, n_openings : ndarray, ndarray
        When *intervals_only* is False.
    list of ndarray
        When *intervals_only* is True.
    """
    periods = record.periods
    args = (periods.intervals, periods.amplitudes, abs(tcrit), periods.flags)
    if intervals_only:
        return extract_burst_intervals(*args)
    return extract_bursts(*args)
