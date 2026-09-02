"""Golden values for the record layer.

Every package in the DCPROGS stack reads its records through this code, so a
change here reaches SCALCS, EKDIST and HJCFIT at once. These are the numbers
that pin it: fixed outputs for the shipped example records at several
resolutions and critical times.

Two kinds of assertion, deliberately mixed.

The *golden* ones are exact and will move if any of the arithmetic changes --
resolution imposition, period grouping, the burst convention, the binning.
That is what they are for. When one moves, the question to answer is whether
the change was intended, not whether the number should be updated.

The *invariant* ones hold for any record at any resolution and say what the
layer guarantees rather than what it happened to produce. They are the ones
that keep meaning something if the example files are ever replaced.

Resolutions are chosen so the dead time actually bites. The simulated record's
shortest interval is 125 us, so anything at or below 100 us merges nothing and
would exercise the concatenation logic not at all.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from dcio.analysis import bursts_from_record, from_scn, log_bin_histogram
from dcio.formats.scn import FLAG_UNUSABLE, read


@dataclass(frozen=True)
class Case:
    filename: str
    tres: float
    tcrit: float
    n_resolved: int
    n_open: int
    n_shut: int
    n_bursts: int
    n_openings: int
    mean_burst_ms: float
    mean_popen: float
    open_nbdec: int
    open_bins: int
    shut_nbdec: int
    shut_bins: int

    def __str__(self):
        return f"{self.filename.split('_')[1][:4]}-{self.tres * 1e6:.0f}us-{self.tcrit * 1e3:.0f}ms"


CASES = [
    Case("glyr_experimental.scn", 2.5e-05, 0.004,
         n_resolved=3232, n_open=731, n_shut=730,
         n_bursts=63, n_openings=731,
         mean_burst_ms=98.629089, mean_popen=0.991578,
         open_nbdec=8, open_bins=29, shut_nbdec=8, shut_bins=53),
    Case("glyr_experimental.scn", 0.0001, 0.004,
         n_resolved=2408, n_open=90, n_shut=89,
         n_bursts=62, n_openings=90,
         mean_burst_ms=100.030402, mean_popen=0.997952,
         open_nbdec=5, open_bins=20, shut_nbdec=5, shut_bins=30),
    Case("glyr_experimental.scn", 0.0005, 0.01,
         n_resolved=1812, n_open=67, n_shut=66,
         n_bursts=56, n_openings=67,
         mean_burst_ms=111.190894, mean_popen=0.995910,
         open_nbdec=5, open_bins=17, shut_nbdec=5, shut_bins=27),
    Case("glyr_simulated.scn", 0.0005, 0.004,
         n_resolved=33377, n_open=16688, n_shut=16687,
         n_bursts=9270, n_openings=16688,
         mean_burst_ms=111.686132, mean_popen=0.993102,
         open_nbdec=12, open_bins=40, shut_nbdec=12, shut_bins=52),
    Case("glyr_simulated.scn", 0.001, 0.01,
         n_resolved=23095, n_open=11547, n_shut=11546,
         n_bursts=7021, n_openings=11547,
         mean_burst_ms=149.705263, mean_popen=0.983202,
         open_nbdec=12, open_bins=37, shut_nbdec=12, shut_bins=49),
]


@pytest.fixture(params=CASES, ids=str)
def case(request, examples_dir):
    c = request.param
    path = examples_dir / c.filename
    if not path.is_file():
        pytest.skip(f"{c.filename} not found")
    return c, from_scn(read(path), tres=c.tres)


# --------------------------------------------------------------------------
# Golden: exact outputs
# --------------------------------------------------------------------------

class TestGoldenValues:

    def test_resolved_interval_count(self, case):
        c, rec = case
        assert rec.n_resolved == c.n_resolved

    def test_period_counts(self, case):
        c, rec = case
        assert (rec.periods.n_open, rec.periods.n_shut) == (c.n_open, c.n_shut)

    def test_burst_count_and_openings(self, case):
        c, rec = case
        lengths, nops = bursts_from_record(rec, tcrit=c.tcrit)
        assert len(lengths) == c.n_bursts
        assert int(nops.sum()) == c.n_openings

    def test_mean_burst_length(self, case):
        c, rec = case
        lengths, _ = bursts_from_record(rec, tcrit=c.tcrit)
        assert lengths.mean() * 1e3 == pytest.approx(c.mean_burst_ms, abs=1e-6)

    def test_mean_popen_within_bursts(self, case):
        c, rec = case
        segs = bursts_from_record(rec, tcrit=c.tcrit, intervals_only=True)
        popen = np.mean([s[0::2].sum() / s.sum() for s in segs])
        assert popen == pytest.approx(c.mean_popen, abs=1e-6)

    def test_histogram_binning(self, case):
        c, rec = case
        for kind, X, nbdec, nbins in (
            ("open", rec.periods.open_intervals, c.open_nbdec, c.open_bins),
            ("shut", rec.periods.shut_intervals, c.shut_nbdec, c.shut_bins),
        ):
            counts, _, got = log_bin_histogram(X, c.tres)
            assert got == nbdec, kind
            assert len(counts) == nbins, kind


# --------------------------------------------------------------------------
# Invariant: what the layer guarantees, for any record at any resolution
# --------------------------------------------------------------------------

class TestInvariants:

    def test_resolution_actually_bites(self, case):
        """A golden case that merges nothing tests the concatenation logic not
        at all. The simulated record's shortest interval is 125 us, so this
        would silently pass for every tres below that."""
        c, rec = case
        assert rec.n_resolved < len(rec.intervals)

    def test_resolved_intervals_are_resolvable_unless_flagged(self, case):
        """Every resolved interval is at least one dead time long, unless it is
        flagged unusable.

        Two kinds of exception exist and both are deliberate: a negative
        sentinel for an unfinished final opening, and the short shut that
        impose_resolution emits when the last interval of the record is a brief
        shutting that terminates an open run. Both carry FLAG_UNUSABLE, so the
        invariant is about the usable intervals -- which is what a consumer
        actually relies on."""
        c, rec = case
        usable = (rec.resolved_flags & FLAG_UNUSABLE) == 0
        finite = rec.resolved_intervals[usable & (rec.resolved_intervals > 0)]
        assert (finite >= c.tres - 1e-15).all()

    def test_total_time_is_conserved_after_the_first_resolvable_interval(self, case):
        """Concatenation moves time between intervals; it must not create or
        destroy any.

        The head of the record is the one legitimate loss. impose_resolution
        starts at the first interval that is both resolvable and usable, since
        anything earlier has a start that cannot be located, and the time
        before it is dropped. On the experimental example at 100 us that is
        75.8 us of a 102.7 s record -- small, but not rounding, and an
        invariant that ignored it would be false rather than approximate."""
        c, rec = case
        flags = rec.flags.astype(np.int32).copy()
        flags[rec.intervals < 0] |= FLAG_UNUSABLE
        resolvable = np.where((rec.intervals > c.tres) & (flags < FLAG_UNUSABLE))[0]
        head = rec.intervals[:int(resolvable[0])]
        discarded = head[head > 0].sum()

        raw = rec.intervals[rec.intervals > 0].sum()
        resolved = rec.resolved_intervals[rec.resolved_intervals > 0].sum()
        assert resolved + discarded == pytest.approx(raw, rel=1e-12)

    def test_periods_strictly_alternate(self, case):
        """set_periods merges runs of the same conductance, so the period list
        alternates open/shut with no two of a kind adjacent. Burst
        segmentation depends on this."""
        c, rec = case
        is_open = rec.periods.amplitudes != 0.0
        assert not (is_open[:-1] & is_open[1:]).any()
        assert not (~is_open[:-1] & ~is_open[1:]).any()

    def test_periods_start_and_end_open(self, case):
        c, rec = case
        assert rec.periods.n_open == rec.periods.n_shut + 1

    def test_every_burst_has_an_odd_interval_count(self, case):
        """Each burst starts and ends on an opening. The missed-events
        likelihood is a product of matrices alternating A->F and F->A; an
        even-length burst would end on a shut."""
        c, rec = case
        for seq in bursts_from_record(rec, tcrit=c.tcrit, intervals_only=True):
            assert len(seq) % 2 == 1

    def test_burst_openings_equal_period_openings(self, case):
        """Every open period belongs to exactly one burst, because the record
        begins and ends on an opening and nothing between is discarded."""
        c, rec = case
        _, nops = bursts_from_record(rec, tcrit=c.tcrit)
        assert int(nops.sum()) == rec.periods.n_open

    def test_no_burst_contains_a_gap_longer_than_tcrit(self, case):
        c, rec = case
        for seq in bursts_from_record(rec, tcrit=c.tcrit, intervals_only=True):
            if len(seq) > 1:
                assert (seq[1::2] <= c.tcrit).all()

    def test_histogram_counts_every_interval(self, case):
        c, rec = case
        for X in (rec.periods.open_intervals, rec.periods.shut_intervals):
            counts, edges, _ = log_bin_histogram(X, c.tres)
            assert counts.sum() == len(X)
            assert X.max() <= edges[-1]

    def test_tcrit_sign_is_ignored(self, case):
        c, rec = case
        pos, _ = bursts_from_record(rec, tcrit=c.tcrit)
        neg, _ = bursts_from_record(rec, tcrit=-c.tcrit)
        np.testing.assert_allclose(pos, neg)


class TestGoldenTableItself:
    """The table is only worth having if it covers what it claims to."""

    def test_both_example_records_are_covered(self):
        assert {c.filename for c in CASES} == {
            "glyr_experimental.scn", "glyr_simulated.scn"}

    def test_experimental_record_carries_unusable_intervals(self, examples_dir):
        """The experimental record is the one that exercises flag handling;
        if it ever stops containing unusable intervals, the burst convention
        for them is no longer covered by any golden case."""
        path = examples_dir / "glyr_experimental.scn"
        if not path.is_file():
            pytest.skip("glyr_experimental.scn not found")
        rec = read(path)
        assert int((rec.flags & FLAG_UNUSABLE != 0).sum()) > 0

    def test_cases_span_more_than_one_resolution_per_record(self):
        for name in {c.filename for c in CASES}:
            assert len({c.tres for c in CASES if c.filename == name}) > 1
