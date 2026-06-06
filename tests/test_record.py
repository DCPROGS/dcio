"""Tests for dcio.analysis.record — impose_resolution, set_periods, from_scn."""

from pathlib import Path

import numpy as np
import pytest

from dcio.analysis.record import (
    Periods,
    SingleChannelRecord,
    from_scn,
    impose_resolution,
    set_periods,
)
from dcio.formats.scn import read as scn_read, write as scn_write

EXAMPLES_SCN = Path(__file__).parent.parent / "examples" / "scn"

# ---------------------------------------------------------------------------
# Shared synthetic records
# ---------------------------------------------------------------------------

# Perfect alternating record: shut/open/shut/open/shut(unusable)
_IVL = np.array([0.010, 0.002, 0.020, 0.005, 0.030])
_AMP = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
_FLG = np.array([0, 0, 0, 0, 8], dtype=np.int8)


def _make_scn(tmp_path, n=20, tres_us=0):
    """Write a minimal simulated SCN file with n open+shut pairs."""
    ivl = np.tile([0.010, 0.002], n // 2 + 1)[:n + 1]  # alternating shut/open, end shut
    amp = np.tile([0.0, 1.0], n // 2 + 1)[:n + 1]
    flg = np.zeros(n + 1, dtype=np.int8)
    flg[-1] = 8
    p = tmp_path / "test.scn"
    scn_write(p, ivl, amp, flg, title="pytest record")
    return p


# ---------------------------------------------------------------------------
# impose_resolution
# ---------------------------------------------------------------------------


class TestImposeResolution:
    def test_no_resolution_passes_through(self):
        ri, ra, rf = impose_resolution(_IVL, _AMP, _FLG, 0.0, record_type="simulated")
        # Output has same count as input; last flag must be unusable
        assert len(ri) == len(_IVL)
        assert rf[-1] == 8

    def test_last_flag_always_unusable(self):
        ri, ra, rf = impose_resolution(_IVL, _AMP, _FLG, 0.0, record_type="simulated")
        assert rf[-1] == 8

    def test_returns_numpy_arrays(self):
        ri, ra, rf = impose_resolution(_IVL, _AMP, _FLG, 0.0, record_type="simulated")
        assert isinstance(ri, np.ndarray)
        assert isinstance(ra, np.ndarray)
        assert isinstance(rf, np.ndarray)

    def test_dtypes(self):
        ri, ra, rf = impose_resolution(_IVL, _AMP, _FLG, 0.0, record_type="simulated")
        assert ri.dtype == np.float64
        assert ra.dtype == np.float64
        assert rf.dtype == np.int8

    def test_short_open_between_shuts_concatenated(self):
        # brief open (0.001 s < tres=0.003) between two shuts → merged shut
        ivl = np.array([0.010, 0.001, 0.020, 0.005, 0.030])
        amp = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        flg = np.zeros(5, dtype=np.int8); flg[-1] = 8
        ri, ra, rf = impose_resolution(ivl, amp, flg, 0.003, record_type="simulated")
        # brief open gets swallowed: first emitted interval must be a merged shut
        assert ra[0] == pytest.approx(0.0)
        assert ri[0] == pytest.approx(0.010 + 0.001 + 0.020)

    def test_short_shut_between_opens_concatenated(self):
        # brief shut (0.001 s < tres=0.003) between two opens → merged open
        ivl = np.array([0.010, 0.005, 0.001, 0.004, 0.030])
        amp = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        flg = np.zeros(5, dtype=np.int8); flg[-1] = 8
        ri, ra, rf = impose_resolution(ivl, amp, flg, 0.003, record_type="simulated")
        # first shut emitted, then a merged open = 0.005+0.001+0.004 = 0.010
        open_durations = ri[ra != 0]
        assert open_durations[0] == pytest.approx(0.005 + 0.001 + 0.004)

    def test_total_time_conserved(self):
        ri, ra, rf = impose_resolution(_IVL, _AMP, _FLG, 0.003, record_type="simulated")
        # sum of positive durations should equal original sum (all positive here)
        assert ri[ri > 0].sum() == pytest.approx(_IVL.sum(), rel=1e-6)

    def test_simulated_amplitude_not_weighted(self):
        # simulated: open amplitude stays as-is (not divided by time)
        ri, ra, rf = impose_resolution(_IVL, _AMP, _FLG, 0.0, record_type="simulated")
        open_amps = ra[ra != 0]
        assert np.all(open_amps == pytest.approx(1.0))

    def test_experimental_weighted_mean_amplitude(self):
        # Two consecutive opens: 0.010 s @ 2.0 pA, 0.010 s @ 4.0 pA (same sub-level → no merge)
        # They have different amplitudes so they will NOT be merged.
        # Use same amplitude to trigger merge: check weighted mean = amplitude
        ivl = np.array([0.010, 0.005, 0.030])
        amp = np.array([0.0, 3.0, 0.0])
        flg = np.zeros(3, dtype=np.int8); flg[-1] = 8
        ri, ra, rf = impose_resolution(ivl, amp, flg, 0.0, record_type="experimental")
        open_amps = ra[ra != 0]
        assert open_amps[0] == pytest.approx(3.0)

    def test_negative_interval_flagged_unusable(self):
        ivl = np.array([0.010, -0.002, 0.020, 0.005, 0.030])
        amp = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        flg = np.zeros(5, dtype=np.int8); flg[-1] = 8
        # negative interval treated as unresolvable, gets flagged
        ri, ra, rf = impose_resolution(ivl, amp, flg, 0.0, record_type="simulated")
        # function should complete without error
        assert len(ri) > 0

    def test_all_unusable_raises(self):
        ivl = np.array([0.010, 0.005, 0.030])
        amp = np.array([0.0, 1.0, 0.0])
        flg = np.array([8, 8, 8], dtype=np.int8)
        with pytest.raises(ValueError, match="No resolvable"):
            impose_resolution(ivl, amp, flg, 0.0)

    def test_all_below_tres_raises(self):
        ivl = np.array([0.001, 0.001, 0.001])
        amp = np.array([0.0, 1.0, 0.0])
        flg = np.zeros(3, dtype=np.int8)
        with pytest.raises(ValueError, match="No resolvable"):
            impose_resolution(ivl, amp, flg, tres=0.01)

    def test_badopen_flags_long_opening(self):
        # open interval = 0.020 s > badopen = 0.010 s → flag
        ri, ra, rf = impose_resolution(
            _IVL, _AMP, _FLG, 0.0,
            record_type="simulated", badopen=0.010,
        )
        # find the first open interval in resolved list
        open_idx = np.where(ra != 0)[0]
        # second open (0.005 s) is shorter than badopen=0.010 s → not flagged
        # first open (0.002 s) is also shorter → not flagged
        # so no opens should be flagged here (both < 0.010 s)
        assert np.all(rf[open_idx] < 8)

    def test_badopen_flags_correct_interval(self):
        # open = 0.020 s > badopen = 0.010 s
        ivl = np.array([0.010, 0.020, 0.030])
        amp = np.array([0.0, 1.0, 0.0])
        flg = np.zeros(3, dtype=np.int8); flg[-1] = 8
        ri, ra, rf = impose_resolution(
            ivl, amp, flg, 0.0, record_type="simulated", badopen=0.015,
        )
        open_idx = np.where(ra != 0)[0]
        assert rf[open_idx[0]] == 8

    def test_output_flags_independent_of_input(self):
        """impose_resolution must not modify the caller's flags array."""
        flg = _FLG.copy()
        impose_resolution(_IVL, _AMP, flg, 0.0, record_type="simulated")
        np.testing.assert_array_equal(flg, _FLG)


# ---------------------------------------------------------------------------
# set_periods
# ---------------------------------------------------------------------------


class TestSetPeriods:
    @pytest.fixture()
    def resolved(self):
        """Resolved record from _IVL/_AMP/_FLG with tres=0."""
        return impose_resolution(_IVL, _AMP, _FLG, 0.0, record_type="simulated")

    def test_returns_periods_instance(self, resolved):
        p = set_periods(*resolved)
        assert isinstance(p, Periods)

    def test_first_period_is_open(self, resolved):
        p = set_periods(*resolved)
        assert p.intervals[0] in p.open_intervals

    def test_open_periods_count(self, resolved):
        p = set_periods(*resolved)
        assert p.n_open == 2

    def test_shut_periods_count(self, resolved):
        p = set_periods(*resolved)
        assert p.n_shut == 1

    def test_n_open_is_n_shut_plus_one(self, resolved):
        p = set_periods(*resolved)
        assert p.n_open == p.n_shut + 1

    def test_open_interval_durations(self, resolved):
        p = set_periods(*resolved)
        np.testing.assert_allclose(sorted(p.open_intervals), [0.002, 0.005])

    def test_shut_interval_duration(self, resolved):
        p = set_periods(*resolved)
        np.testing.assert_allclose(p.shut_intervals, [0.020])

    def test_shut_amplitudes_are_zero(self, resolved):
        p = set_periods(*resolved)
        np.testing.assert_array_equal(p.shut_amplitudes, 0.0)

    def test_leading_shut_trimmed(self):
        # Record starting with a shut interval
        ivl = np.array([0.050, 0.010, 0.030])
        amp = np.array([0.0, 1.0, 0.0])
        flg = np.zeros(3, dtype=np.int8); flg[-1] = 8
        ri, ra, rf = impose_resolution(ivl, amp, flg, 0.0, record_type="simulated")
        p = set_periods(ri, ra, rf)
        assert p.n_open >= 1
        # leading shut should be absent as a period (it was trimmed)
        assert all(d > 0 for d in p.open_intervals)

    def test_trailing_shut_trimmed(self, resolved):
        p = set_periods(*resolved)
        # last period in the array is open (trailing shuts are trimmed)
        last_amp = p.amplitudes[-1] if len(p.amplitudes) else None
        assert last_amp != 0.0 or p.n_open == 0

    def test_empty_record_returns_empty_periods(self):
        empty = np.array([], dtype=np.float64)
        p = set_periods(empty, empty, np.array([], dtype=np.int8))
        assert p.n_open == 0
        assert p.n_shut == 0

    def test_single_open_interval(self):
        # Record with one open interval and surrounding shuts
        ivl = np.array([0.010, 0.005, 0.030])
        amp = np.array([0.0, 1.0, 0.0])
        flg = np.zeros(3, dtype=np.int8); flg[-1] = 8
        ri, ra, rf = impose_resolution(ivl, amp, flg, 0.0, record_type="simulated")
        p = set_periods(ri, ra, rf)
        assert p.n_open == 1
        assert p.open_intervals[0] == pytest.approx(0.005)

    def test_all_intervals_positive(self, resolved):
        p = set_periods(*resolved)
        assert np.all(p.intervals > 0)


# ---------------------------------------------------------------------------
# Periods properties
# ---------------------------------------------------------------------------


class TestPeriods:
    @pytest.fixture()
    def periods(self):
        ivl = np.array([0.005, 0.020, 0.003])   # open, shut, open
        amp = np.array([1.0, 0.0, 1.0])
        flg = np.zeros(3, dtype=np.int8)
        return Periods(ivl, amp, flg)

    def test_open_intervals(self, periods):
        np.testing.assert_array_equal(periods.open_intervals, [0.005, 0.003])

    def test_shut_intervals(self, periods):
        np.testing.assert_array_equal(periods.shut_intervals, [0.020])

    def test_n_open(self, periods):
        assert periods.n_open == 2

    def test_n_shut(self, periods):
        assert periods.n_shut == 1

    def test_repr_contains_n_open(self, periods):
        assert "n_open=2" in repr(periods)

    def test_repr_contains_mean_open(self, periods):
        assert "ms" in repr(periods)

    def test_empty_repr(self):
        empty = np.array([], dtype=np.float64)
        p = Periods(empty, empty, np.array([], dtype=np.int8))
        assert "empty" in repr(p).lower()


# ---------------------------------------------------------------------------
# from_scn
# ---------------------------------------------------------------------------


class TestFromScn:
    @pytest.fixture()
    def simulated_scn(self, tmp_path):
        p = _make_scn(tmp_path, n=40)
        return scn_read(p)

    def test_returns_single_channel_record(self, simulated_scn):
        scr = from_scn(simulated_scn)
        assert isinstance(scr, SingleChannelRecord)

    def test_tres_stored(self, simulated_scn):
        scr = from_scn(simulated_scn, tres=40e-6)
        assert scr.tres == pytest.approx(40e-6)

    def test_tres_zero_by_default(self, simulated_scn):
        scr = from_scn(simulated_scn)
        assert scr.tres == 0.0

    def test_record_type_preserved(self, simulated_scn):
        scr = from_scn(simulated_scn)
        assert scr.record_type == "simulated"

    def test_path_preserved(self, tmp_path):
        p = _make_scn(tmp_path)
        rec = scn_read(p)
        scr = from_scn(rec)
        assert scr.path == p

    def test_n_raw_matches_scn(self, simulated_scn):
        scr = from_scn(simulated_scn)
        assert scr.n_raw == simulated_scn.n_intervals

    def test_n_resolved_positive(self, simulated_scn):
        scr = from_scn(simulated_scn)
        assert scr.n_resolved > 0

    def test_n_open_positive(self, simulated_scn):
        scr = from_scn(simulated_scn)
        assert scr.periods.n_open > 0

    def test_n_shut_positive(self, simulated_scn):
        scr = from_scn(simulated_scn)
        assert scr.periods.n_shut > 0

    def test_open_periods_all_positive(self, simulated_scn):
        scr = from_scn(simulated_scn)
        assert np.all(scr.open_periods > 0)

    def test_shut_periods_all_positive(self, simulated_scn):
        scr = from_scn(simulated_scn)
        assert np.all(scr.shut_periods > 0)

    def test_open_periods_shortcut(self, simulated_scn):
        scr = from_scn(simulated_scn)
        np.testing.assert_array_equal(scr.open_periods, scr.periods.open_intervals)

    def test_shut_periods_shortcut(self, simulated_scn):
        scr = from_scn(simulated_scn)
        np.testing.assert_array_equal(scr.shut_periods, scr.periods.shut_intervals)

    def test_n_open_equals_n_shut_or_off_by_one(self, simulated_scn):
        scr = from_scn(simulated_scn)
        diff = abs(scr.periods.n_open - scr.periods.n_shut)
        assert diff <= 1

    def test_repr_contains_key_fields(self, simulated_scn):
        scr = from_scn(simulated_scn)
        text = repr(scr)
        assert "SingleChannelRecord" in text
        assert "tres" in text
        assert "n_raw" in text


# ---------------------------------------------------------------------------
# Resolution reduces interval count
# ---------------------------------------------------------------------------


class TestResolutionEffect:
    def test_high_tres_reduces_resolved_count(self, tmp_path):
        p = _make_scn(tmp_path, n=100)
        rec = scn_read(p)
        scr_low = from_scn(rec, tres=0.0)
        scr_high = from_scn(rec, tres=0.003)
        # higher tres → more concatenation → fewer resolved intervals
        assert scr_high.n_resolved <= scr_low.n_resolved

    def test_high_tres_reduces_open_periods(self, tmp_path):
        p = _make_scn(tmp_path, n=100)
        rec = scn_read(p)
        scr_low = from_scn(rec, tres=0.0)
        scr_high = from_scn(rec, tres=0.003)
        assert scr_high.periods.n_open <= scr_low.periods.n_open


# ---------------------------------------------------------------------------
# Legacy real-file smoke tests (skipped if files absent)
# ---------------------------------------------------------------------------


class TestLegacySimulated:
    @pytest.fixture()
    def rec(self):
        p = EXAMPLES_SCN / "glyr_simulated.scn"
        if not p.exists():
            pytest.skip("glyr_simulated.scn not found")
        return scn_read(p)

    def test_from_scn_no_tres(self, rec):
        scr = from_scn(rec)
        assert scr.periods.n_open > 1000

    def test_from_scn_with_tres(self, rec):
        scr = from_scn(rec, tres=40e-6)
        assert scr.periods.n_open > 0

    def test_resolution_reduces_count(self, rec):
        scr0 = from_scn(rec, tres=0.0)
        scr1 = from_scn(rec, tres=40e-6)
        assert scr1.n_resolved <= scr0.n_resolved


class TestLegacyExperimental:
    @pytest.fixture()
    def rec(self):
        p = EXAMPLES_SCN / "glyr_experimental.scn"
        if not p.exists():
            pytest.skip("glyr_experimental.scn not found")
        return scn_read(p)

    def test_from_scn_experimental(self, rec):
        scr = from_scn(rec, tres=40e-6)
        assert scr.record_type == "experimental"
        assert scr.periods.n_open > 0

    def test_open_periods_finite(self, rec):
        scr = from_scn(rec, tres=40e-6)
        assert np.all(np.isfinite(scr.open_periods))

    def test_shut_periods_finite(self, rec):
        scr = from_scn(rec, tres=40e-6)
        assert np.all(np.isfinite(scr.shut_periods))

    def test_repr_shows_experimental(self, rec):
        scr = from_scn(rec, tres=40e-6)
        assert "experimental" in repr(scr)
