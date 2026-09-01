"""Log-binned dwell-time histograms: the arithmetic, not the figure.

Dwell times span decades, so they are histogrammed on a logarithmic time axis:
each bin is a fixed ratio wider than the last, and the ordinate is plotted as
the square root of the count (Sigworth & Sine 1987), which makes an exponential
component a recognisable peak rather than a featureless decay.

This module carries only the numbers -- bin counts, bin edges, and the step
coordinates a staircase plot needs.  Drawing stays with the caller: EKDIST
draws for a person, HJCFIT draws inside a notebook, SCALCS feeds a Qt canvas,
and those are legitimately different figures over the same bins.

    from dcio.analysis.histogram import log_bin_histogram, staircase

    counts, edges, nbdec = log_bin_histogram(open_periods, tres=25e-6)
    x, y = staircase(edges, counts)
    ax.semilogx(x, np.sqrt(y))
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bins_per_decade",
    "log_bin_edges",
    "log_bin_histogram",
    "staircase",
]


def bins_per_decade(n):
    """Bins per decade for a sample of *n* intervals.

    The DCprogs convention: widen the bins for smaller samples so the counts
    in them stay usable.

    Parameters
    ----------
    n : int
        Number of intervals in the sample.

    Returns
    -------
    int
        5 at 300 intervals or fewer, 8 to 1000, 10 to 3000, 12 above.
    """
    if n <= 300:
        return 5
    if n <= 1000:
        return 8
    if n <= 3000:
        return 10
    return 12


def log_bin_edges(intervals, tres, nbdec=None):
    """Geometric bin edges for a dwell-time histogram.

    Bins start at the resolution and each is ``10 ** (1 / nbdec)`` times wider
    than the last, so *nbdec* of them span a decade.  The last edge is at or
    above the longest interval, rounded up to a whole decade.

    That rounding is the part worth stating.  Earlier implementations in this
    stack wrote the decade round-up as ``exp(ceil(log(max)))``, which rounds up
    to the next power of *e* rather than of ten.  Because it can land below the
    longest interval, ``numpy.histogram`` then drops the tail of the
    distribution without saying so -- in randomly drawn exponential samples
    that happens about three times in ten.

    Parameters
    ----------
    intervals : array_like
        Observed dwell times, in seconds.  Must contain a positive value.
    tres : float
        Resolution, in seconds.  The histogram starts here.
    nbdec : int, optional
        Bins per decade.  Chosen from the sample size when omitted; see
        :func:`bins_per_decade`.

    Returns
    -------
    edges : ndarray
        Bin edges, in seconds, ascending from *tres*.
    nbdec : int
        The number of bins per decade actually used.

    Raises
    ------
    ValueError
        If *tres* is not positive, or no interval is positive.
    """
    intervals = np.asarray(intervals, dtype=float)
    if tres <= 0.0:
        raise ValueError(f"tres must be positive, got {tres!r}")

    positive = intervals[intervals > 0.0]
    if positive.size == 0:
        raise ValueError("no positive interval to histogram")

    if nbdec is None:
        nbdec = bins_per_decade(len(intervals))

    ratio = 10.0 ** (1.0 / nbdec)
    tmax = 10.0 ** np.ceil(np.log10(positive.max()))
    nbin = int(np.log(tmax / tres) / np.log(ratio)) + 1
    return tres * ratio ** np.arange(nbin + 1), nbdec


def log_bin_histogram(intervals, tres, nbdec=None):
    """Bin dwell times logarithmically.

    Parameters
    ----------
    intervals : array_like
        Observed dwell times, in seconds.
    tres : float
        Resolution, in seconds.
    nbdec : int, optional
        Bins per decade; see :func:`bins_per_decade`.

    Returns
    -------
    counts : ndarray of int
        Number of intervals in each bin.
    edges : ndarray
        Bin edges, in seconds.
    nbdec : int
        Bins per decade used.

    Notes
    -----
    Intervals shorter than *tres* fall below the first edge and are not
    counted; by construction of the record they should not exist.  No interval
    falls above the last edge -- see :func:`log_bin_edges`.
    """
    intervals = np.asarray(intervals, dtype=float)
    edges, nbdec = log_bin_edges(intervals, tres, nbdec)
    counts, _ = np.histogram(intervals, bins=edges)
    return counts, edges, nbdec


def staircase(edges, counts):
    """Step-plot coordinates for a binned histogram.

    Turns *n* counts and *n + 1* edges into the vertex sequence a line plot
    needs to render as a staircase, closed to zero at both ends.

    Parameters
    ----------
    edges : array_like
        Bin edges, length ``n + 1``.
    counts : array_like
        Bin counts, length ``n``.

    Returns
    -------
    x, y : ndarray
        Vertices, each of length ``2 * (n + 1)``.  Plot them directly; take
        ``sqrt(y)`` for the conventional square-root ordinate.
    """
    edges = np.asarray(edges, dtype=float)
    counts = np.asarray(counts)
    if len(edges) != len(counts) + 1:
        raise ValueError(
            f"expected {len(counts) + 1} edges for {len(counts)} counts, "
            f"got {len(edges)}"
        )
    x = np.repeat(edges, 2)
    y = np.concatenate(([0], np.repeat(counts, 2), [0]))
    return x, y
