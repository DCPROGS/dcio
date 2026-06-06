"""Tests for dcio.formats.scan_ini."""

from pathlib import Path

import pytest

from dcio.formats.scan_ini import (
    ISCRIT_FRACTION_AMP,
    ISCRIT_MULTIPLE_RMS,
    INI_SIZE_NEW,
    INI_SIZE_OLD,
    ScanIniRecord,
    read,
)

EXAMPLES_INI = Path(__file__).parent.parent / "examples" / "ini"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def example_ini_path():
    p = EXAMPLES_INI / "SCAN.INI"
    if not p.exists():
        pytest.skip("examples/ini/SCAN.INI not found.")
    return p


@pytest.fixture(scope="module")
def example_ini(example_ini_path):
    return read(example_ini_path)


def _write_ini(path: Path, data: bytes) -> Path:
    """Write raw bytes as a SCAN.INI file."""
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# Basic file reading
# ---------------------------------------------------------------------------


class TestReadExampleFile:
    """Smoke tests against the real 512-byte SCAN.INI example."""

    def test_returns_scan_ini_record(self, example_ini):
        assert isinstance(example_ini, ScanIniRecord)

    def test_file_size_is_512(self, example_ini):
        assert example_ini.ini_size == INI_SIZE_NEW

    def test_path_stored(self, example_ini, example_ini_path):
        assert example_ini.path == example_ini_path

    def test_adcfil_is_string(self, example_ini):
        assert isinstance(example_ini.adcfil, str)

    def test_adcfil_value(self, example_ini):
        # Known value from the example file
        assert example_ini.adcfil == r"C:\fort90\scandemo\consam2.513"

    def test_adcfil_length_at_most_30(self, example_ini):
        assert len(example_ini.adcfil) <= 30

    def test_opendown_is_true(self, example_ini):
        # Confirmed from binary: opendown=TRUE (offset 9 = 1)
        assert example_ini.opendown is True

    def test_invert_is_false(self, example_ini):
        assert example_ini.invert is False

    def test_novlap_is_2048(self, example_ini):
        # Default value, confirmed from binary
        assert example_ini.novlap == 2048

    def test_cjump_is_false(self, example_ini):
        assert example_ini.cjump is False

    def test_iscrit_is_int(self, example_ini):
        assert isinstance(example_ini.iscrit, int)

    def test_smult_is_positive_float(self, example_ini):
        assert example_ini.smult > 0.0

    def test_tmin_is_positive(self, example_ini):
        assert example_ini.tmin > 0.0

    def test_tmin_value(self, example_ini):
        # Confirmed from binary: 20.0 µs (user-changed from default 15)
        assert abs(example_ini.tmin - 20.0) < 0.01

    def test_novlap_is_int(self, example_ini):
        assert isinstance(example_ini.novlap, int)

    def test_iamark_length_10(self, example_ini):
        assert len(example_ini.iamark) == 10

    def test_nampmark_is_zero(self, example_ini):
        assert example_ini.nampmark == 0

    def test_izoom_is_1(self, example_ini):
        assert example_ini.izoom == 1

    def test_minmeth_is_int(self, example_ini):
        assert isinstance(example_ini.minmeth, int)

    def test_nbasemin_is_int(self, example_ini):
        assert isinstance(example_ini.nbasemin, int)

    def test_512_fields_present(self, example_ini):
        # scritvar, smultmin, stpfac are set (even if garbage from conversion)
        assert example_ini.scritvar is not None
        assert example_ini.smultmin is not None
        assert example_ini.stpfac is not None

    def test_savin_is_string(self, example_ini):
        assert isinstance(example_ini.savin, str)

    def test_ndevdat_is_string(self, example_ini):
        assert isinstance(example_ini.ndevdat, str)

    def test_filtfile_is_string(self, example_ini):
        assert isinstance(example_ini.filtfile, str)


# ---------------------------------------------------------------------------
# Synthetic binary construction
# ---------------------------------------------------------------------------


class TestSyntheticBinary:
    """Build minimal synthetic SCAN.INI bytes and verify parsing."""

    @staticmethod
    def _make_ini(size=512, **overrides) -> bytes:
        """Return a zero-filled SCAN.INI with specific fields set."""
        import struct

        data = bytearray(size)

        # savin at 0 (char*1)
        data[0] = ord(overrides.get("savin", "N"))

        # nbuf at 1 (int32)
        struct.pack_into("<i", data, 1, overrides.get("nbuf", 131072))

        # novlap at 5 (int32)
        struct.pack_into("<i", data, 5, overrides.get("novlap", 2048))

        # opendown at 9 (logical4)
        struct.pack_into("<i", data, 9, 1 if overrides.get("opendown", True) else 0)

        # invert at 13 (logical4)
        struct.pack_into("<i", data, 13, 1 if overrides.get("invert", False) else 0)

        # smult at 17 (float32)
        struct.pack_into("<f", data, 17, overrides.get("smult", 0.14))

        # ntrig at 21 (int32)
        struct.pack_into("<i", data, 21, overrides.get("ntrig", 2))

        # tmin at 213 (float32)
        struct.pack_into("<f", data, 213, overrides.get("tmin", 15.0))

        # cjump at 217 (logical4)
        struct.pack_into("<i", data, 217, 1 if overrides.get("cjump", False) else 0)

        # izoom at 221 (int32)
        struct.pack_into("<i", data, 221, overrides.get("izoom", 1))

        # iscrit at 245 (int32)
        struct.pack_into("<i", data, 245, overrides.get("iscrit", 1))

        # adcfil at 139 (char*30)
        adcfil = overrides.get("adcfil", "C:\\CONSAM.DAT")
        encoded = adcfil.encode("ascii")[:30]
        data[139: 139 + len(encoded)] = encoded

        return bytes(data)

    def test_read_synthetic_novlap(self, tmp_path):
        raw = self._make_ini(novlap=4096)
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert rec.novlap == 4096

    def test_read_synthetic_opendown_false(self, tmp_path):
        raw = self._make_ini(opendown=False)
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert rec.opendown is False

    def test_read_synthetic_opendown_true(self, tmp_path):
        raw = self._make_ini(opendown=True)
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert rec.opendown is True

    def test_read_synthetic_invert_true(self, tmp_path):
        raw = self._make_ini(invert=True)
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert rec.invert is True

    def test_read_synthetic_cjump_true(self, tmp_path):
        raw = self._make_ini(cjump=True)
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert rec.cjump is True

    def test_read_synthetic_smult(self, tmp_path):
        raw = self._make_ini(smult=0.20)
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert abs(rec.smult - 0.20) < 1e-5

    def test_read_synthetic_tmin(self, tmp_path):
        raw = self._make_ini(tmin=25.0)
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert abs(rec.tmin - 25.0) < 1e-4

    def test_read_synthetic_iscrit_2(self, tmp_path):
        raw = self._make_ini(iscrit=ISCRIT_MULTIPLE_RMS)
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert rec.iscrit == ISCRIT_MULTIPLE_RMS

    def test_read_synthetic_adcfil(self, tmp_path):
        raw = self._make_ini(adcfil=r"C:\data\test.dat")
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert rec.adcfil == r"C:\data\test.dat"

    def test_read_synthetic_adcfil_30_chars(self, tmp_path):
        # Exactly 30 characters (maximum)
        adcfil = "C:" + "\\x" * 14  # 2 + 28 = 30 chars
        raw = self._make_ini(adcfil=adcfil)
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert len(rec.adcfil) <= 30

    def test_read_synthetic_savin_Y(self, tmp_path):
        raw = self._make_ini(savin="Y")
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert rec.savin == "Y"

    def test_read_synthetic_nbuf(self, tmp_path):
        raw = self._make_ini(nbuf=65536)
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert rec.nbuf == 65536

    def test_size_256_returns_none_for_new_fields(self, tmp_path):
        raw = self._make_ini(size=256)
        p = _write_ini(tmp_path / "SCAN256.INI", raw)
        rec = read(p)
        assert rec.ini_size == INI_SIZE_OLD
        assert rec.scritvar is None
        assert rec.smultmin is None
        assert rec.stpfac is None

    def test_size_512_returns_scritvar(self, tmp_path):
        import struct
        raw = bytearray(self._make_ini(size=512))
        struct.pack_into("<i", raw, 249, 1)  # scritvar = True
        struct.pack_into("<f", raw, 253, 2.5)  # smultmin
        struct.pack_into("<f", raw, 257, 0.1)  # stpfac
        p = _write_ini(tmp_path / "SCAN512.INI", bytes(raw))
        rec = read(p)
        assert rec.scritvar is True
        assert abs(rec.smultmin - 2.5) < 1e-5
        assert abs(rec.stpfac - 0.1) < 1e-5

    def test_izoom_round_trip(self, tmp_path):
        raw = self._make_ini(izoom=4)
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert rec.izoom == 4

    def test_ntrig_round_trip(self, tmp_path):
        raw = self._make_ini(ntrig=3)
        p = _write_ini(tmp_path / "SCAN.INI", raw)
        rec = read(p)
        assert rec.ntrig == 3


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read(tmp_path / "no_such.ini")

    def test_wrong_size_raises_value_error(self, tmp_path):
        p = tmp_path / "bad.ini"
        p.write_bytes(b"\x00" * 100)  # 100 bytes – not a valid size
        with pytest.raises(ValueError, match="expected"):
            read(p)

    def test_wrong_size_message_contains_actual_size(self, tmp_path):
        p = tmp_path / "bad.ini"
        p.write_bytes(b"\x00" * 300)
        with pytest.raises(ValueError, match="300"):
            read(p)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_iscrit_fraction_amp_is_1(self):
        assert ISCRIT_FRACTION_AMP == 1

    def test_iscrit_multiple_rms_is_2(self):
        assert ISCRIT_MULTIPLE_RMS == 2

    def test_ini_size_old_is_256(self):
        assert INI_SIZE_OLD == 256

    def test_ini_size_new_is_512(self):
        assert INI_SIZE_NEW == 512
