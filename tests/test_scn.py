"""Tests for dcio.formats.scn – SCN file I/O."""

from pathlib import Path

import numpy as np
import pytest

from dcio.formats.scn import (
    FLAG_OK,
    FLAG_UNUSABLE,
    SCNHeader,
    SCNRecord,
    read,
    write,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_record(n_open: int = 3, amplitude_pA: float = -48.0, tmp_path: Path = None):
    """Return (path, intervals_s, amplitudes, flags) for a synthetic record.

    Alternating shut/open sequence: shut(10 ms) open(2 ms) …, ending shut.
    """
    n_total = 2 * n_open + 1
    intervals = np.full(n_total, 0.010)      # 10 ms each
    amplitudes = np.zeros(n_total)
    amplitudes[1::2] = amplitude_pA          # odd indices = open
    flags = np.zeros(n_total, dtype=np.int8)
    if tmp_path is not None:
        path = tmp_path / "synthetic.scn"
        write(path, intervals, amplitudes, flags)
        return path, intervals, amplitudes, flags
    return intervals, amplitudes, flags


# ===========================================================================
# write() – argument validation
# ===========================================================================


class TestWriteValidation:
    def test_length_mismatch_raises(self, tmp_path):
        ivl = np.array([0.01, 0.02])
        amp = np.array([0.0])
        flg = np.zeros(2, dtype=np.int8)
        with pytest.raises(ValueError, match="same length"):
            write(tmp_path / "bad.scn", ivl, amp, flg)

    def test_amplitude_overflow_raises(self, tmp_path):
        ivl = np.array([0.01])
        amp = np.array([40000.0])   # exceeds int16 max
        flg = np.zeros(1, dtype=np.int8)
        with pytest.raises(ValueError, match="int16"):
            write(tmp_path / "overflow.scn", ivl, amp, flg)

    def test_returns_resolved_path(self, tmp_path):
        ivl = np.array([0.01, 0.02])
        amp = np.array([0.0, 0.0])
        flg = np.zeros(2, dtype=np.int8)
        result = write(tmp_path / "out.scn", ivl, amp, flg)
        assert isinstance(result, Path)
        assert result.exists()


# ===========================================================================
# Round-trip:  write → read
# ===========================================================================


class TestRoundTrip:
    def test_n_intervals_preserved(self, tmp_path):
        path, ivl, amp, flg = _make_record(n_open=5, tmp_path=tmp_path)
        rec = read(path)
        assert rec.n_intervals == len(ivl)

    def test_intervals_roundtrip_seconds(self, tmp_path):
        path, ivl, amp, flg = _make_record(n_open=4, tmp_path=tmp_path)
        rec = read(path)
        # float32 storage introduces ~1e-4 relative error
        np.testing.assert_allclose(rec.intervals, ivl, rtol=1e-5)

    def test_amplitudes_roundtrip_pA(self, tmp_path):
        amplitude = -62.0
        path, ivl, amp, flg = _make_record(
            n_open=3, amplitude_pA=amplitude, tmp_path=tmp_path
        )
        rec = read(path)
        open_amps = rec.amplitudes[rec.open_mask]
        np.testing.assert_array_equal(open_amps, amplitude)

    def test_shut_amplitudes_are_zero(self, tmp_path):
        path, ivl, amp, flg = _make_record(n_open=3, tmp_path=tmp_path)
        rec = read(path)
        assert np.all(rec.amplitudes[rec.shut_mask] == 0.0)

    def test_flags_roundtrip(self, tmp_path):
        ivl = np.array([0.010, 0.002, 0.020])
        amp = np.array([0.0, -48.0, 0.0])
        flg = np.array([0, FLAG_UNUSABLE, 0], dtype=np.int8)
        path = tmp_path / "flagged.scn"
        write(path, ivl, amp, flg)
        rec = read(path)
        np.testing.assert_array_equal(rec.flags, flg)

    def test_header_version_is_minus103(self, tmp_path):
        path, *_ = _make_record(tmp_path=tmp_path)
        rec = read(path)
        assert rec.header.version == -103

    def test_header_record_type_simulated(self, tmp_path):
        path, *_ = _make_record(tmp_path=tmp_path)
        rec = read(path)
        assert rec.header.record_type == "simulated"

    def test_header_calfac2_default(self, tmp_path):
        path, *_ = _make_record(tmp_path=tmp_path)
        rec = read(path)
        assert rec.header.calfac2 == pytest.approx(1.0)

    def test_custom_calfac_scales_amplitude(self, tmp_path):
        calfac = 0.5
        ivl = np.array([0.010, 0.002, 0.010])
        amp = np.array([0.0, -96.0, 0.0])    # stored as -96, calfac=0.5 → read back -48
        flg = np.zeros(3, dtype=np.int8)
        path = tmp_path / "calfac.scn"
        write(path, ivl, amp, flg, calfac=calfac)
        rec = read(path)
        # amplitude stored as int16(-96), then multiplied by calfac2=0.5 → -48.0
        open_amps = rec.amplitudes[rec.open_mask]
        np.testing.assert_allclose(open_amps, [-48.0], atol=0.01)

    def test_title_stored_in_header(self, tmp_path):
        ivl = np.array([0.01])
        amp = np.array([0.0])
        flg = np.zeros(1, dtype=np.int8)
        path = tmp_path / "titled.scn"
        write(path, ivl, amp, flg, title="Test experiment 2024")
        rec = read(path)
        assert "Test experiment 2024" in rec.header.title

    def test_ffilt_roundtrip(self, tmp_path):
        ivl = np.array([0.01, 0.02])
        amp = np.array([0.0, 0.0])
        flg = np.zeros(2, dtype=np.int8)
        path = tmp_path / "filtered.scn"
        write(path, ivl, amp, flg, ffilt=3000.0)
        rec = read(path)
        assert rec.header.ffilt == pytest.approx(3000.0, rel=1e-5)


# ===========================================================================
# SCNRecord – convenience properties
# ===========================================================================


class TestSCNRecordProperties:
    def test_open_mask_selects_nonzero_amplitude(self, minimal_scn):
        rec = read(minimal_scn)
        assert np.all(rec.amplitudes[rec.open_mask] != 0.0)

    def test_shut_mask_selects_zero_amplitude(self, minimal_scn):
        rec = read(minimal_scn)
        assert np.all(rec.amplitudes[rec.shut_mask] == 0.0)

    def test_usable_mask_excludes_flag8(self, minimal_scn):
        rec = read(minimal_scn)
        # Last interval in minimal_scn has flag=8
        assert not rec.usable_mask[-1]

    def test_n_intervals_matches_array_len(self, minimal_scn):
        rec = read(minimal_scn)
        assert rec.n_intervals == len(rec.intervals)

    def test_repr_contains_key_info(self, minimal_scn):
        rec = read(minimal_scn)
        text = repr(rec)
        assert "SCNRecord" in text
        assert "simulated" in text
        assert "intervals" in text


# ===========================================================================
# read() – error handling
# ===========================================================================


class TestReadErrors:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read(tmp_path / "nonexistent.scn")

    def test_bad_version_raises_value_error(self, tmp_path):
        # Write a file with an invalid version number (999)
        path = tmp_path / "bad_version.scn"
        with open(path, "wb") as fh:
            import struct
            fh.write(struct.pack("<iii", 999, 154, 0))
            fh.write(b"\x00" * 200)
        with pytest.raises(ValueError, match="Unrecognised SCAN version"):
            read(path)

    def test_string_path_accepted(self, minimal_scn):
        rec = read(str(minimal_scn))
        assert isinstance(rec, SCNRecord)

    def test_path_stored_on_record(self, minimal_scn):
        rec = read(minimal_scn)
        assert rec.path == minimal_scn


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_interval_record(self, tmp_path):
        ivl = np.array([0.050])
        amp = np.array([0.0])
        flg = np.zeros(1, dtype=np.int8)
        path = tmp_path / "single.scn"
        write(path, ivl, amp, flg)
        rec = read(path)
        assert rec.n_intervals == 1
        assert rec.intervals[0] == pytest.approx(0.050, rel=1e-5)

    def test_zero_amplitude_all_shut(self, tmp_path):
        n = 10
        ivl = np.linspace(0.005, 0.050, n)
        amp = np.zeros(n)
        flg = np.zeros(n, dtype=np.int8)
        path = tmp_path / "allshut.scn"
        write(path, ivl, amp, flg)
        rec = read(path)
        assert np.all(rec.amplitudes == 0.0)

    def test_negative_amplitudes_roundtrip(self, tmp_path):
        ivl = np.array([0.010, 0.005, 0.010])
        amp = np.array([0.0, -120.0, 0.0])
        flg = np.zeros(3, dtype=np.int8)
        path = tmp_path / "neg_amp.scn"
        write(path, ivl, amp, flg)
        rec = read(path)
        assert rec.amplitudes[1] == pytest.approx(-120.0)

    def test_large_record_performance(self, tmp_path):
        """10 000-interval record should read back without error."""
        n = 10_000
        ivl = np.full(n, 0.001)
        amp = np.where(np.arange(n) % 2 == 0, 0.0, -50.0)
        flg = np.zeros(n, dtype=np.int8)
        path = tmp_path / "large.scn"
        write(path, ivl, amp, flg)
        rec = read(path)
        assert rec.n_intervals == n

    def test_intervals_stored_as_milliseconds(self, tmp_path):
        """Verify that the binary file really stores ms, not seconds."""
        import struct
        ivl_s = np.array([0.012345])   # 12.345 ms
        amp = np.array([0.0])
        flg = np.zeros(1, dtype=np.int8)
        path = tmp_path / "units.scn"
        write(path, ivl_s, amp, flg)
        with open(path, "rb") as fh:
            fh.seek(153)   # ioffset - 1 = 153
            raw_ms = struct.unpack("<f", fh.read(4))[0]
        assert raw_ms == pytest.approx(12.345, rel=1e-4)


# ===========================================================================
# Legacy real-file smoke tests (skipped if files absent)
# ===========================================================================


class TestLegacyFiles:
    def test_legacy_simulated_loads(self, legacy_test_scn):
        rec = read(legacy_test_scn)
        assert rec.n_intervals > 0
        assert rec.header.version in (-103, 103, 104)

    def test_legacy_simulated_intervals_positive(self, legacy_test_scn):
        rec = read(legacy_test_scn)
        assert np.all(rec.intervals >= 0)

    def test_legacy_experimental_loads(self, legacy_experimental_scn):
        rec = read(legacy_experimental_scn)
        assert rec.header.record_type == "experimental"
        assert rec.header.srate > 0

    def test_legacy_experimental_calfac2_nonzero(self, legacy_experimental_scn):
        rec = read(legacy_experimental_scn)
        assert rec.header.calfac2 != 0.0
