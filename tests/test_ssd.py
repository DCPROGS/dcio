"""Tests for dcio.formats.ssd – SSD file I/O."""

from pathlib import Path

import numpy as np
import pytest

from dcio.formats.ssd import (
    SSDHeader,
    SSDRecord,
    read,
    write,
    _HEADER_SIZE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXAMPLES_SSD = Path(__file__).parent.parent / "examples" / "ssd"


def _sine(n=1000, amplitude=0.5, dt_us=20) -> np.ndarray:
    """Return a simple sine-wave signal in pA (amplitude safe for gain=1.0)."""
    t = np.arange(n) * dt_us * 1e-6
    return amplitude * np.sin(2 * np.pi * 100 * t)


# ---------------------------------------------------------------------------
# write() – argument validation
# ---------------------------------------------------------------------------


class TestWriteValidation:
    def test_empty_signal_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            write(tmp_path / "empty.ssd", np.array([]), dt_us=20)

    def test_dt_us_too_small_raises(self, tmp_path):
        with pytest.raises(ValueError, match="dt_us"):
            write(tmp_path / "bad.ssd", np.zeros(10), dt_us=0)

    def test_dt_us_too_large_raises(self, tmp_path):
        with pytest.raises(ValueError, match="dt_us"):
            write(tmp_path / "bad.ssd", np.zeros(10), dt_us=32768)

    def test_overflow_raises(self, tmp_path):
        # gain=1.0, signal=10000 pA → 10000 * 1 * 6553.6 = 65_536_000 >> int16
        with pytest.raises(ValueError, match="int16"):
            write(tmp_path / "overflow.ssd", np.full(10, 10_000.0), dt_us=20)

    def test_returns_resolved_path(self, tmp_path):
        result = write(tmp_path / "out.ssd", np.zeros(100), dt_us=20)
        assert isinstance(result, Path)
        assert result.exists()

    def test_file_is_created(self, tmp_path):
        p = tmp_path / "out.ssd"
        write(p, np.zeros(50), dt_us=10)
        assert p.exists()


# ---------------------------------------------------------------------------
# Round-trip:  write → read
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_n_samples_preserved(self, tmp_path):
        sig = _sine(n=2000)
        write(tmp_path / "sig.ssd", sig, dt_us=20)
        rec = read(tmp_path / "sig.ssd")
        assert rec.n_samples == 2000

    def test_dt_roundtrip(self, tmp_path):
        write(tmp_path / "dt.ssd", np.zeros(100), dt_us=25)
        rec = read(tmp_path / "dt.ssd")
        assert rec.header.dt_us == 25
        assert rec.dt == pytest.approx(25e-6)

    def test_signal_roundtrip_pA(self, tmp_path):
        sig = _sine(n=500)
        write(tmp_path / "sig.ssd", sig, dt_us=20, gain=1.0)
        rec = read(tmp_path / "sig.ssd")
        # int16 quantisation error ≤ 0.5 / 6553.6 ≈ 7.6e-5 pA
        np.testing.assert_allclose(rec.signal, sig, atol=1e-3)

    def test_signal_roundtrip_with_gain(self, tmp_path):
        sig = np.linspace(-50.0, 50.0, 200)
        gain = 0.05   # 50 mV/pA
        write(tmp_path / "g.ssd", sig, dt_us=10, gain=gain)
        rec = read(tmp_path / "g.ssd")
        np.testing.assert_allclose(rec.signal, sig, atol=1.0)  # pA tolerance

    def test_zero_signal_stays_zero(self, tmp_path):
        write(tmp_path / "zero.ssd", np.zeros(100), dt_us=20)
        rec = read(tmp_path / "zero.ssd")
        np.testing.assert_array_equal(rec.signal, 0.0)

    def test_srate_derived_correctly(self, tmp_path):
        write(tmp_path / "s.ssd", np.zeros(10), dt_us=20)
        rec = read(tmp_path / "s.ssd")
        assert rec.header.srate == pytest.approx(50_000.0, rel=1e-4)

    def test_filt_roundtrip(self, tmp_path):
        write(tmp_path / "f.ssd", np.zeros(10), dt_us=20, filt=3000.0)
        rec = read(tmp_path / "f.ssd")
        assert rec.header.filt == pytest.approx(3000.0, rel=1e-4)

    def test_Emem_roundtrip(self, tmp_path):
        write(tmp_path / "e.ssd", np.zeros(10), dt_us=20, Emem=-80.0)
        rec = read(tmp_path / "e.ssd")
        assert rec.header.Emem == pytest.approx(-80.0, rel=1e-4)

    def test_temp_roundtrip(self, tmp_path):
        write(tmp_path / "t.ssd", np.zeros(10), dt_us=20, temp=22.5)
        rec = read(tmp_path / "t.ssd")
        assert rec.header.temp == pytest.approx(22.5, rel=1e-4)

    def test_title_roundtrip(self, tmp_path):
        write(tmp_path / "ti.ssd", np.zeros(10), dt_us=20, title="patch 42 run 3")
        rec = read(tmp_path / "ti.ssd")
        assert "patch 42 run 3" in rec.header.title

    def test_version_is_1002(self, tmp_path):
        write(tmp_path / "v.ssd", np.zeros(10), dt_us=20)
        rec = read(tmp_path / "v.ssd")
        assert rec.header.version == 1002

    def test_calfac_consistent_with_gain(self, tmp_path):
        gain = 0.1
        write(tmp_path / "c.ssd", np.zeros(10), dt_us=20, gain=gain)
        rec = read(tmp_path / "c.ssd")
        expected_calfac = 1.0 / (gain * 6553.6)
        assert rec.header.calfac == pytest.approx(expected_calfac, rel=1e-5)

    def test_data_offset_is_512(self, tmp_path):
        write(tmp_path / "off.ssd", np.zeros(10), dt_us=20)
        rec = read(tmp_path / "off.ssd")
        assert rec.header.data_offset == 512

    def test_path_stored_on_record(self, tmp_path):
        p = tmp_path / "path.ssd"
        write(p, np.zeros(10), dt_us=20)
        rec = read(p)
        assert rec.path == p


# ---------------------------------------------------------------------------
# SSDRecord – convenience properties
# ---------------------------------------------------------------------------


class TestSSDRecordProperties:
    @pytest.fixture()
    def rec(self, tmp_path):
        sig = _sine(n=1000, dt_us=20)
        write(tmp_path / "r.ssd", sig, dt_us=20)
        return read(tmp_path / "r.ssd")

    def test_duration_seconds(self, rec):
        expected = 1000 * 20e-6
        assert rec.duration == pytest.approx(expected, rel=1e-5)

    def test_time_axis_length(self, rec):
        assert len(rec.time_axis) == rec.n_samples

    def test_time_axis_start(self, rec):
        assert rec.time_axis[0] == pytest.approx(0.0)

    def test_time_axis_end(self, rec):
        assert rec.time_axis[-1] == pytest.approx((rec.n_samples - 1) * rec.dt)

    def test_n_samples_matches_signal(self, rec):
        assert rec.n_samples == len(rec.signal)

    def test_repr_contains_key_info(self, rec):
        text = repr(rec)
        assert "SSDRecord" in text
        assert "n_samples" in text
        assert "Hz" in text


# ---------------------------------------------------------------------------
# read() – error handling
# ---------------------------------------------------------------------------


class TestReadErrors:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read(tmp_path / "nonexistent.ssd")

    def test_bad_version_raises_value_error(self, tmp_path):
        p = tmp_path / "bad.ssd"
        with open(p, "wb") as fh:
            import struct
            fh.write(struct.pack("<h", 9999))
            fh.write(b"\x00" * 600)
        with pytest.raises(ValueError, match="Unrecognised SSD version"):
            read(p)

    def test_string_path_accepted(self, tmp_path):
        p = tmp_path / "str.ssd"
        write(p, np.zeros(10), dt_us=20)
        rec = read(str(p))
        assert isinstance(rec, SSDRecord)


# ---------------------------------------------------------------------------
# Binary layout verification
# ---------------------------------------------------------------------------


class TestBinaryLayout:
    def test_file_size_equals_header_plus_data(self, tmp_path):
        n = 300
        p = write(tmp_path / "sz.ssd", np.zeros(n), dt_us=20)
        expected = _HEADER_SIZE + n * 2   # int16 = 2 bytes
        assert p.stat().st_size == expected

    def test_header_size_is_512(self, tmp_path):
        """Data must start at byte 512 regardless of signal length."""
        p = write(tmp_path / "hdr.ssd", np.zeros(1), dt_us=10)
        with open(p, "rb") as fh:
            import struct
            fh.seek(4)              # skip version(2) + first 2 bytes of title
            # check ioff field offset: version(2) + title(70) + date(11)
            #   + time(8) + idt(2) = 93 bytes in
            fh.seek(2 + 70 + 11 + 8 + 2)   # = 93
            ioff = struct.unpack("<i", fh.read(4))[0]
        assert ioff == 512

    def test_int16_samples_stored_correctly(self, tmp_path):
        """Single sample: verify raw int16 in file matches expectation."""
        gain = 1.0
        val_pA = 1.0   # 1 pA × 1 V/pA × 6553.6 = 6553 — well within int16
        p = write(tmp_path / "raw.ssd", np.array([val_pA]), dt_us=20, gain=gain)
        with open(p, "rb") as fh:
            fh.seek(512)
            raw = np.frombuffer(fh.read(2), dtype="<i2")[0]
        expected_raw = round(val_pA * gain * 6553.6)
        assert raw == expected_raw


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_sample(self, tmp_path):
        write(tmp_path / "one.ssd", np.array([0.0]), dt_us=10)
        rec = read(tmp_path / "one.ssd")
        assert rec.n_samples == 1

    def test_negative_signal_roundtrip(self, tmp_path):
        sig = np.full(50, -0.5)   # -0.5 pA, safe with default gain=1.0
        write(tmp_path / "neg.ssd", sig, dt_us=20)
        rec = read(tmp_path / "neg.ssd")
        np.testing.assert_allclose(rec.signal, sig, atol=1e-3)

    def test_large_record(self, tmp_path):
        """500 000-sample record should read back without error."""
        n = 500_000
        sig = np.zeros(n)
        write(tmp_path / "large.ssd", sig, dt_us=20)
        rec = read(tmp_path / "large.ssd")
        assert rec.n_samples == n

    def test_dt_us_boundary_values(self, tmp_path):
        for dt in (1, 32767):
            write(tmp_path / f"dt{dt}.ssd", np.zeros(10), dt_us=dt)
            rec = read(tmp_path / f"dt{dt}.ssd")
            assert rec.header.dt_us == dt

    def test_signal_preserves_shape(self, tmp_path):
        sig = _sine(n=123)
        write(tmp_path / "shape.ssd", sig, dt_us=20)
        rec = read(tmp_path / "shape.ssd")
        assert rec.signal.shape == (123,)


# ---------------------------------------------------------------------------
# Legacy real-file smoke test (skipped if file not found)
# ---------------------------------------------------------------------------


class TestLegacyFiles:
    @pytest.fixture()
    def legacy_ssd(self):
        for ext in ("*.ssd", "*.SSD", "*.dat", "*.DAT"):
            found = list(EXAMPLES_SSD.glob(ext)) if EXAMPLES_SSD.exists() else []
            if found:
                return found[0]
        pytest.skip("No SSD example file found.")

    def test_legacy_loads(self, legacy_ssd):
        rec = read(legacy_ssd)
        assert rec.n_samples > 0

    def test_legacy_signal_is_finite(self, legacy_ssd):
        rec = read(legacy_ssd)
        assert np.all(np.isfinite(rec.signal))
