"""Burst segmentation against the rule transcribed from the Fortran.

``dcio.analysis.bursts`` is checked here against an independent implementation
of the same definition, transcribed from ``hjclik.for`` in
``DCPROGS/DCFORTRAN`` (``Fort90/HJCFIT/``) -- the likelihood that produced
every published HJCFIT result, Colquhoun, Hatton & Hawkes (2003) among them.

``test_bursts.py`` fixes the conventions by stating them; this fixes them
against a separate reading of the original, which is a different kind of
evidence. The three implementations in the DCPROGS stack disagreed on exactly
these points, and agreeing with each other would not have settled which was
right.

The Fortran does not pre-split a record. ``hjclik.for`` line 1009 walks it and
ends the group when a shut time exceeds ``tcrit``::

    if(tint(in,jset).gt.tcrit(jset)) goto 92   !end present group with prev opening

Three things follow, and each is a separate test below:

* the comparison is **strictly greater**, so a gap exactly equal to ``tcrit``
  is within-burst;
* the group ends *with the previous opening*, so the separating gap belongs to
  no group and every group starts and ends on an opening;
* the test is **not guarded by** ``burst(jset)``. A ``tcrit`` of zero therefore
  ends the group at every shut time. HJCFIT means "do not divide" by setting
  ``tcrit`` enormous -- 3.1536e10 ms, one year (``Hjcfit1-09122003.for`` line
  1568) -- not by setting it to zero. That trap cost real time to find and is
  worth a test of its own.
"""

import numpy as np
import numpy.testing as npt
import pytest

from dcio.analysis.bursts import extract_burst_intervals

TCRIT = 3.5e-3


def fortran_bursts(tints, ampls, tcrit):
    """``hjclik.for`` line 1009, transcribed.

    A shut time greater than ``tcrit`` ends the group with the previous
    opening. Returns a list of interval arrays, each starting and ending on an
    opening.
    """
    t = np.asarray(tints, float)
    a = np.asarray(ampls, float)
    groups, cur = [], []
    for ti, ai in zip(t, a):
        if ai == 0.0 and ti > tcrit:            # strictly greater
            if cur:
                groups.append(cur)
            cur = []
        else:
            cur.append((ti, ai))
    if cur:
        groups.append(cur)

    out = []
    for g in groups:                            # "with prev opening"
        while g and g[0][1] == 0.0:
            g = g[1:]
        while g and g[-1][1] == 0.0:
            g = g[:-1]
        if g:
            out.append(np.array([x[0] for x in g]))
    return out


def alternating(n, seed, tcrit=TCRIT):
    """A record that alternates open and shut, with gaps either side of tcrit."""
    rng = np.random.default_rng(seed)
    t = np.empty(n)
    a = np.zeros(n)
    for i in range(n):
        if i % 2 == 0:                          # opening
            t[i] = rng.exponential(3e-4)
            a[i] = 5.0
        else:                                   # shut: a mixture, so that
            t[i] = (rng.exponential(2e-4) if rng.random() < 0.75
                    else rng.exponential(2e-2))  # some gaps exceed tcrit
    return t, a


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_agrees_with_the_fortran_rule(seed):
    t, a = alternating(4000, seed)
    got = extract_burst_intervals(t, a, TCRIT)
    want = fortran_bursts(t, a, TCRIT)

    assert len(got) == len(want)
    for g, w in zip(got, want):
        npt.assert_allclose(np.asarray(g, float), w, rtol=0, atol=0)
    assert len(want) > 10, "the record should actually contain bursts"


@pytest.mark.parametrize("tcrit", [1e-4, 5e-4, 3.5e-3, 2e-2])
def test_agrees_at_every_tcrit(tcrit):
    t, a = alternating(2000, seed=7)
    got = extract_burst_intervals(t, a, tcrit)
    want = fortran_bursts(t, a, tcrit)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        npt.assert_allclose(np.asarray(g, float), w, rtol=0, atol=0)


def test_the_comparison_is_strictly_greater():
    """A gap exactly equal to tcrit is within-burst, as line 1009 has it."""
    t = np.array([1e-4, TCRIT, 1e-4, TCRIT * 1.000001, 1e-4])
    a = np.array([5.0, 0.0, 5.0, 0.0, 5.0])

    got = extract_burst_intervals(t, a, TCRIT)
    want = fortran_bursts(t, a, TCRIT)
    assert len(got) == len(want) == 2, "only the longer gap should split"
    npt.assert_allclose(np.asarray(got[0], float), want[0])


def test_every_burst_starts_and_ends_on_an_opening():
    """"end present group with prev opening" -- the separating gap is dropped."""
    t, a = alternating(2000, seed=11)
    for g in extract_burst_intervals(t, a, TCRIT):
        g = np.asarray(g, float)
        assert g.size % 2 == 1, "odd length means opening at both ends"


def test_tcrit_of_zero_splits_at_every_shut_time():
    """The trap: line 1009 is not guarded by burst(jset).

    HJCFIT means "do not divide" by setting tcrit to a year, not to zero. A
    caller who passes zero expecting "no division" gets one group per opening,
    silently, and this records that both implementations behave that way.
    """
    t, a = alternating(200, seed=13)
    got = extract_burst_intervals(t, a, 0.0)
    want = fortran_bursts(t, a, 0.0)
    assert len(got) == len(want)
    assert all(np.asarray(g, float).size == 1 for g in got)
    assert len(got) == int((a != 0).sum())


def test_an_enormous_tcrit_is_how_you_ask_for_one_group():
    """3.1536e10 ms, one year: Hjcfit1-09122003.for line 1568."""
    t, a = alternating(200, seed=17)
    one_year = 3.1536e10 * 1e-3                  # ms -> s
    got = extract_burst_intervals(t, a, one_year)
    want = fortran_bursts(t, a, one_year)
    assert len(got) == len(want) == 1
    npt.assert_allclose(np.asarray(got[0], float), want[0])
