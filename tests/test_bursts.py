"""Burst segmentation.

The convention tests are the ones that pinned this behaviour in SCALCS; they
move here with the code.  Each names the reading of the burst definition it
fixes, because the three implementations in the DCPROGS stack disagreed on
exactly these points until they were written down.
"""

import numpy as np
import pytest

from dcio.analysis.bursts import (
    bursts_from_record,
    extract_burst_intervals,
    extract_bursts,
)
from dcio.analysis.record import from_scn
from dcio.formats.scn import FLAG_UNUSABLE, read


class TestExtractBursts:

    def test_handbuilt_three_bursts(self):
        """This record opens and closes with a long shut, so all three runs
        are bounded by separators and all three are complete bursts.

        These ends used to be dropped unconditionally.  Against the Burzomato
        2004 records that lost two bursts from every file -- at 30 uM, two of
        six."""
        #        sep  A.o  A.s   A.o  sep  B.o  B.s   B.o  B.s   B.o  sep  C.o  sep
        t = np.array([1, 0.1, 0.01, 0.1, 1, 0.2, 0.02, 0.3, 0.01, 0.1, 1, 0.5, 1.0])
        a = np.array([0, 5,   0,    5,   0, 5,   0,    5,   0,    5,   0, 5,   0.0])
        lengths, nops = extract_bursts(t, a, tcrit=0.5)
        assert len(lengths) == 3
        assert lengths[0] == pytest.approx(0.21)     # 0.1+0.01+0.1
        assert lengths[1] == pytest.approx(0.63)     # 0.2+0.02+0.3+0.01+0.1
        assert lengths[2] == pytest.approx(0.5)
        assert list(nops) == [2, 3, 1]

    def test_keeps_ends_bounded_by_separators(self):
        """Both runs are bounded by long shuts, so both are complete."""
        t = np.array([1.0, 0.1, 1.0, 0.2, 1.0])
        a = np.array([0.0, 5.0, 0.0, 5.0, 0.0])
        lengths, _ = extract_bursts(t, a, tcrit=0.5)
        assert len(lengths) == 2

    def test_first_opening_starts_a_burst(self):
        """No gap is required before the first burst: a record beginning on a
        defined opening begins on a burst.  EKDIST states this convention in
        Bursts.slice_bursts and dcpyps followed it."""
        t = np.array([0.1, 1.0, 0.2, 1.0, 0.3])
        a = np.array([5.0, 0.0, 5.0, 0.0, 5.0])
        lengths, _ = extract_bursts(t, a, tcrit=0.5)
        assert len(lengths) == 3
        assert lengths == pytest.approx([0.1, 0.2, 0.3])

    def test_no_separators_at_all(self):
        """With nothing to cut on, the record is a single burst."""
        t = np.array([0.1, 0.01, 0.1])
        a = np.array([5.0, 0.0, 5.0])
        lengths, _ = extract_bursts(t, a, tcrit=0.5)
        assert len(lengths) == 1
        assert lengths[0] == pytest.approx(0.21)

    def test_shut_exactly_tcrit_is_within_burst(self):
        """tcrit is the time such that gaps *longer* than it end a burst, so a
        shut interval exactly equal to it does not.  No real record exercises
        this -- none of the four Burzomato 2004 files contains a shut time
        equal to its tcrit -- so the convention is pinned here instead.  EKDIST
        makes the same choice."""
        t = np.array([0.1, 0.5, 0.2])          # middle shut is exactly tcrit
        a = np.array([5.0, 0.0, 5.0])
        lengths, _ = extract_bursts(t, a, tcrit=0.5)
        assert len(lengths) == 1               # one burst, not two
        assert lengths[0] == pytest.approx(0.8)

        lengths, _ = extract_bursts(t, a, tcrit=0.4999)
        assert len(lengths) == 2               # just above, and it cuts

    def test_unusable_interval_ends_a_burst(self):
        """Time-course fitting leaves the last interval with no defined
        length, flagged unusable.  It still ends the burst before it, and its
        meaningless duration is never compared with tcrit."""
        #                o    s     o    unusable shut
        t = np.array([0.1, 0.01, 0.1, 0.00005])
        a = np.array([5.0, 0.0,  5.0, 0.0])
        flags = np.array([0, 0, 0, FLAG_UNUSABLE])
        lengths, _ = extract_bursts(t, a, tcrit=0.5, flags=flags)
        assert len(lengths) == 1
        assert lengths[0] == pytest.approx(0.21)

    def test_unusable_shut_cuts_mid_record(self):
        """An unusable interval in the middle ends one burst and the next
        begins after it, however short its nominal duration."""
        t = np.array([0.1, 0.00004, 0.2])
        a = np.array([5.0, 0.0, 5.0])
        flags = np.array([0, FLAG_UNUSABLE, 0])
        lengths, _ = extract_bursts(t, a, tcrit=0.5, flags=flags)
        assert len(lengths) == 2
        assert lengths == pytest.approx([0.1, 0.2])

    def test_leading_bad_interval_discarded_not_the_burst(self):
        """A bad leading interval is dropped, but the first defined opening
        after it still starts a burst."""
        t = np.array([0.05, 0.1, 1.0, 0.2])
        a = np.array([5.0,  5.0, 0.0, 5.0])
        flags = np.array([FLAG_UNUSABLE, 0, 0, 0])
        lengths, _ = extract_bursts(t, a, tcrit=0.5, flags=flags)
        assert len(lengths) == 2
        assert lengths == pytest.approx([0.1, 0.2])

    def test_empty_record(self):
        lengths, nops = extract_bursts(np.array([]), np.array([]), tcrit=0.5)
        assert len(lengths) == 0
        assert len(nops) == 0

    def test_no_openings_at_all(self):
        t = np.array([1.0, 2.0, 3.0])
        a = np.array([0.0, 0.0, 0.0])
        lengths, _ = extract_bursts(t, a, tcrit=0.5)
        assert len(lengths) == 0


class TestExtractBurstIntervals:

    def test_same_segmentation_as_extract_bursts(self):
        t = np.array([1, 0.1, 0.01, 0.1, 1, 0.2, 0.02, 0.3, 1.0])
        a = np.array([0, 5,   0,    5,   0, 5,   0,    5,   0.0])
        lengths, nops = extract_bursts(t, a, tcrit=0.5)
        seqs = extract_burst_intervals(t, a, tcrit=0.5)
        assert len(seqs) == len(lengths)
        for seq, length in zip(seqs, lengths):
            assert seq.sum() == pytest.approx(length)

    def test_every_burst_has_odd_length(self):
        """Each burst starts and ends on an opening, so its interval sequence
        alternates open/shut/.../open.  The missed-events likelihood requires
        this -- an even-length sequence would end on a shut."""
        t = np.array([1, 0.1, 0.01, 0.1, 1, 0.2, 0.02, 0.3, 1.0])
        a = np.array([0, 5,   0,    5,   0, 5,   0,    5,   0.0])
        for seq in extract_burst_intervals(t, a, tcrit=0.5):
            assert len(seq) % 2 == 1


class TestBurstsFromRecord:
    """The record-level entry point, on the shipped experimental example."""

    @pytest.fixture()
    def record(self, legacy_experimental_scn):
        return from_scn(read(legacy_experimental_scn), tres=25e-6)

    def test_golden_burst_count(self, record):
        """Pins the segmentation of the shipped experimental record.

        Any change to impose_resolution or to the burst convention moves these
        numbers; that is the point of having them.  Measured at tres = 25 us,
        tcrit = 4 ms on examples/scn/glyr_experimental.scn."""
        assert record.n_resolved == 3232
        assert record.periods.n_open == 731
        lengths, nops = bursts_from_record(record, tcrit=4e-3)
        assert len(lengths) == 63
        assert nops.sum() == 731
        assert lengths.mean() == pytest.approx(98.629e-3, abs=1e-6)

    def test_tcrit_sign_is_ignored(self, record):
        """HJCFIT reads a negative tcrit as a flag selecting equilibrium
        vectors over CHS vectors; the magnitude is the critical time.  The same
        number is passed to both, so segmentation must take the magnitude."""
        pos, _ = bursts_from_record(record, tcrit=4e-3)
        neg, _ = bursts_from_record(record, tcrit=-4e-3)
        np.testing.assert_allclose(pos, neg)

    def test_segments_periods_not_resolved_intervals(self, record):
        """impose_resolution emits a fresh open interval at every change of
        fitted amplitude, so a record with sub-conductance levels does not
        alternate open/shut -- this one has 1770 open-open adjacencies in 3232
        resolved intervals.  Segmenting periods instead gives the same bursts
        and the same lengths, but counts an opening once rather than once per
        sub-level, and yields the alternating sequences the missed-events
        likelihood requires."""
        ra = record.resolved_amplitudes != 0.0
        assert int((ra[:-1] & ra[1:]).sum()) == 1770      # the record does have them

        by_period, nops_period = bursts_from_record(record, tcrit=4e-3)
        by_interval, nops_interval = extract_bursts(
            record.resolved_intervals, record.resolved_amplitudes, 4e-3,
            flags=record.resolved_flags,
        )
        assert len(by_period) == len(by_interval)          # same bursts
        np.testing.assert_allclose(by_period, by_interval)  # same lengths
        assert nops_period.sum() == 731                     # openings, merged
        assert nops_interval.sum() == 2501                  # sub-levels counted

    def test_flags_are_supplied_automatically(self, record):
        """The record-level call must pass the flags through; omitting them
        treats an experimental record as if time-course fitting had left no
        undefined intervals in it."""
        p = record.periods
        auto, _ = bursts_from_record(record, tcrit=4e-3)
        explicit, _ = extract_bursts(p.intervals, p.amplitudes, 4e-3, flags=p.flags)
        np.testing.assert_allclose(auto, explicit)

    def test_intervals_only_agrees_on_lengths(self, record):
        lengths, _ = bursts_from_record(record, tcrit=4e-3)
        seqs = bursts_from_record(record, tcrit=4e-3, intervals_only=True)
        assert len(seqs) == len(lengths)
        np.testing.assert_allclose([s.sum() for s in seqs], lengths)

    def test_every_burst_starts_and_ends_open(self, record):
        for seq in bursts_from_record(record, tcrit=4e-3, intervals_only=True):
            assert len(seq) % 2 == 1
            assert len(seq) >= 1
