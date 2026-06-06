"""Tests for dcio.cli — scn2txt, txt2scn, csv2ssd entry points."""

from pathlib import Path

import numpy as np
import pytest

from dcio.cli.scn2txt import main as scn2txt
from dcio.cli.txt2scn import main as txt2scn
from dcio.cli.csv2ssd import main as csv2ssd
from dcio.formats.scn import read as scn_read, write as scn_write
from dcio.formats.ssd import read as ssd_read

EXAMPLES_SCN = Path(__file__).parent.parent / "examples" / "scn"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_txt(path: Path, amp_col, ivl_col, delimiter="\t") -> Path:
    """Write a minimal two-column interval file."""
    with open(path, "w") as fh:
        for a, t in zip(amp_col, ivl_col):
            fh.write(f"{a}{delimiter}{t}\n")
    return path


def _write_csv(path: Path, times, signal, delimiter=",") -> Path:
    """Write a two-column time/signal CSV."""
    with open(path, "w") as fh:
        for t, s in zip(times, signal):
            fh.write(f"{t}{delimiter}{s}\n")
    return path


def _simple_scn(tmp_path) -> Path:
    """Write a small synthetic SCN file and return its path."""
    ivl = np.array([0.010, 0.002, 0.020, 0.005, 0.030])
    amp = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    flg = np.array([0, 0, 0, 0, 8], dtype=np.int8)
    p = tmp_path / "simple.scn"
    scn_write(p, ivl, amp, flg, title="cli test")
    return p


# ---------------------------------------------------------------------------
# scn2txt
# ---------------------------------------------------------------------------


class TestScn2Txt:
    def test_creates_output_file(self, tmp_path):
        p = _simple_scn(tmp_path)
        assert scn2txt([str(p)]) == 0
        assert (tmp_path / "simple.txt").exists()

    def test_explicit_output_path(self, tmp_path):
        p = _simple_scn(tmp_path)
        out = tmp_path / "out.txt"
        assert scn2txt([str(p), "-o", str(out)]) == 0
        assert out.exists()

    def test_output_has_correct_row_count(self, tmp_path):
        p = _simple_scn(tmp_path)
        out = tmp_path / "out.txt"
        scn2txt([str(p), "-o", str(out)])
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 5  # same as n_intervals

    def test_output_has_three_columns(self, tmp_path):
        p = _simple_scn(tmp_path)
        out = tmp_path / "out.txt"
        scn2txt([str(p), "-o", str(out)])
        first = out.read_text().strip().splitlines()[0].split("\t")
        assert len(first) == 3

    def test_intervals_in_seconds(self, tmp_path):
        p = _simple_scn(tmp_path)
        out = tmp_path / "out.txt"
        scn2txt([str(p), "-o", str(out)])
        rows = [line.split("\t") for line in out.read_text().strip().splitlines()]
        intervals = [float(r[0]) for r in rows]
        assert all(0 < t < 1.0 for t in intervals if t > 0)  # seconds, not ms

    def test_with_tres_creates_resolved_output(self, tmp_path):
        p = _simple_scn(tmp_path)
        out = tmp_path / "res.txt"
        assert scn2txt([str(p), "--tres", "3", "-o", str(out)]) == 0
        assert out.exists()

    def test_with_tres_and_periods(self, tmp_path):
        p = _simple_scn(tmp_path)
        out = tmp_path / "periods.txt"
        assert scn2txt([str(p), "--tres", "3", "--periods", "-o", str(out)]) == 0
        lines = out.read_text().strip().splitlines()
        assert len(lines) > 0

    def test_periods_without_tres_fails(self, tmp_path):
        p = _simple_scn(tmp_path)
        rc = scn2txt([str(p), "--periods"])
        assert rc != 0

    def test_missing_file_fails(self, tmp_path):
        rc = scn2txt([str(tmp_path / "no_such.scn")])
        assert rc != 0

    def test_returns_zero_on_success(self, tmp_path):
        p = _simple_scn(tmp_path)
        assert scn2txt([str(p)]) == 0

    def test_last_flag_is_8(self, tmp_path):
        """SCN convention: last interval always flag=8."""
        p = _simple_scn(tmp_path)
        out = tmp_path / "out.txt"
        scn2txt([str(p), "-o", str(out)])
        last_row = out.read_text().strip().splitlines()[-1].split("\t")
        assert int(last_row[2]) == 8


# ---------------------------------------------------------------------------
# txt2scn
# ---------------------------------------------------------------------------


class TestTxt2Scn:
    @pytest.fixture()
    def simple_txt(self, tmp_path):
        amps = [0, 1, 0, 1, 0]
        ivls = [10.0, 2.0, 20.0, 5.0, 30.0]  # ms
        return _write_txt(tmp_path / "data.txt", amps, ivls)

    def test_creates_scn_file(self, tmp_path, simple_txt):
        assert txt2scn([str(simple_txt)]) == 0
        assert (tmp_path / "data.scn").exists()

    def test_explicit_output_path(self, tmp_path, simple_txt):
        out = tmp_path / "out.scn"
        assert txt2scn([str(simple_txt), "-o", str(out)]) == 0
        assert out.exists()

    def test_output_is_readable_scn(self, tmp_path, simple_txt):
        out = tmp_path / "out.scn"
        txt2scn([str(simple_txt), "-o", str(out)])
        rec = scn_read(out)
        assert rec.n_intervals == 5

    def test_intervals_converted_from_ms_to_seconds(self, tmp_path, simple_txt):
        out = tmp_path / "out.scn"
        txt2scn([str(simple_txt), "-o", str(out)])
        rec = scn_read(out)
        np.testing.assert_allclose(
            sorted(rec.intervals), sorted([0.010, 0.002, 0.020, 0.005, 0.030]),
            atol=1e-5,
        )

    def test_unit_seconds(self, tmp_path):
        amps = [0, 1, 0]
        ivls = [0.010, 0.002, 0.020]  # seconds
        p = _write_txt(tmp_path / "sec.txt", amps, ivls)
        out = tmp_path / "sec.scn"
        txt2scn([str(p), "--unit", "s", "-o", str(out)])
        rec = scn_read(out)
        np.testing.assert_allclose(
            sorted(rec.intervals), sorted([0.010, 0.002, 0.020]), atol=1e-5
        )

    def test_amplitudes_are_zero_or_one(self, tmp_path, simple_txt):
        out = tmp_path / "out.scn"
        txt2scn([str(simple_txt), "-o", str(out)])
        rec = scn_read(out)
        unique_amps = set(rec.amplitudes.tolist())
        assert unique_amps <= {0.0, 1.0}

    def test_custom_title_stored(self, tmp_path, simple_txt):
        out = tmp_path / "out.scn"
        txt2scn([str(simple_txt), "-o", str(out), "--title", "mytest"])
        rec = scn_read(out)
        assert "mytest" in rec.header.title

    def test_comma_delimiter(self, tmp_path):
        amps = [0, 1, 0]
        ivls = [10.0, 2.0, 20.0]
        p = _write_txt(tmp_path / "comma.txt", amps, ivls, delimiter=",")
        out = tmp_path / "comma.scn"
        txt2scn([str(p), "--delimiter", ",", "-o", str(out)])
        rec = scn_read(out)
        assert rec.n_intervals == 3

    def test_missing_file_fails(self, tmp_path):
        rc = txt2scn([str(tmp_path / "no_such.txt")])
        assert rc != 0

    def test_version_is_minus_103(self, tmp_path, simple_txt):
        out = tmp_path / "out.scn"
        txt2scn([str(simple_txt), "-o", str(out)])
        rec = scn_read(out)
        assert rec.header.version == -103

    def test_roundtrip_via_scn2txt(self, tmp_path, simple_txt):
        """txt → scn → txt should recover the same intervals."""
        scn = tmp_path / "rt.scn"
        txt2scn([str(simple_txt), "-o", str(scn)])
        out = tmp_path / "rt.txt"
        scn2txt([str(scn), "-o", str(out)])
        rows = [r.split("\t") for r in out.read_text().strip().splitlines()]
        ivls_back = np.array([float(r[0]) for r in rows])
        expected = np.array([0.010, 0.002, 0.020, 0.005, 0.030])
        np.testing.assert_allclose(sorted(ivls_back), sorted(expected), atol=1e-5)


# ---------------------------------------------------------------------------
# csv2ssd
# ---------------------------------------------------------------------------


class TestCsv2Ssd:
    @pytest.fixture()
    def simple_csv(self, tmp_path):
        n = 500
        dt_s = 20e-6
        times = np.arange(n) * dt_s
        signal = 0.5 * np.sin(2 * np.pi * 100 * times)  # 0.5 pA sine
        return _write_csv(tmp_path / "trace.csv", times, signal)

    def test_creates_ssd_file(self, tmp_path, simple_csv):
        assert csv2ssd([str(simple_csv), "--dt-us", "20"]) == 0
        assert (tmp_path / "trace.ssd").exists()

    def test_explicit_output_path(self, tmp_path, simple_csv):
        out = tmp_path / "out.ssd"
        assert csv2ssd([str(simple_csv), "--dt-us", "20", "-o", str(out)]) == 0
        assert out.exists()

    def test_output_is_readable_ssd(self, tmp_path, simple_csv):
        out = tmp_path / "out.ssd"
        csv2ssd([str(simple_csv), "--dt-us", "20", "-o", str(out)])
        rec = ssd_read(out)
        assert rec.n_samples == 500

    def test_dt_us_stored_correctly(self, tmp_path, simple_csv):
        out = tmp_path / "out.ssd"
        csv2ssd([str(simple_csv), "--dt-us", "20", "-o", str(out)])
        rec = ssd_read(out)
        assert rec.header.dt_us == 20

    def test_signal_roundtrip(self, tmp_path, simple_csv):
        out = tmp_path / "out.ssd"
        csv2ssd([str(simple_csv), "--dt-us", "20", "-o", str(out)])
        rec = ssd_read(out)
        n = 500
        dt_s = 20e-6
        times = np.arange(n) * dt_s
        expected = 0.5 * np.sin(2 * np.pi * 100 * times)
        np.testing.assert_allclose(rec.signal, expected, atol=1e-3)

    def test_infer_dt(self, tmp_path, simple_csv):
        out = tmp_path / "out.ssd"
        csv2ssd([str(simple_csv), "--infer-dt", "--time-col", "0", "-o", str(out)])
        rec = ssd_read(out)
        assert rec.header.dt_us == 20  # inferred from 20e-6 s column

    def test_gain_stored_in_calfac(self, tmp_path, simple_csv):
        out = tmp_path / "out.ssd"
        gain = 0.05
        csv2ssd([str(simple_csv), "--dt-us", "20", "--gain", str(gain), "-o", str(out)])
        rec = ssd_read(out)
        expected_calfac = 1.0 / (gain * 6553.6)
        assert rec.header.calfac == pytest.approx(expected_calfac, rel=1e-4)

    def test_filt_stored(self, tmp_path, simple_csv):
        out = tmp_path / "out.ssd"
        csv2ssd([str(simple_csv), "--dt-us", "20", "--filt", "3000", "-o", str(out)])
        rec = ssd_read(out)
        assert rec.header.filt == pytest.approx(3000.0, rel=1e-3)

    def test_Emem_stored(self, tmp_path, simple_csv):
        out = tmp_path / "out.ssd"
        csv2ssd([str(simple_csv), "--dt-us", "20", "--Emem", "-80", "-o", str(out)])
        rec = ssd_read(out)
        assert rec.header.Emem == pytest.approx(-80.0, rel=1e-3)

    def test_title_stored(self, tmp_path, simple_csv):
        out = tmp_path / "out.ssd"
        csv2ssd([str(simple_csv), "--dt-us", "20", "--title", "cli test", "-o", str(out)])
        rec = ssd_read(out)
        assert "cli test" in rec.header.title

    def test_signal_col_option(self, tmp_path):
        """--signal-col 0 reads from the first column."""
        n, dt_s = 100, 20e-6
        signal = np.zeros(n)
        times = np.arange(n) * dt_s
        p = _write_csv(tmp_path / "sig0.csv", signal, times)  # signal in col 0!
        out = tmp_path / "sig0.ssd"
        csv2ssd([str(p), "--dt-us", "20", "--signal-col", "0", "-o", str(out)])
        rec = ssd_read(out)
        np.testing.assert_array_equal(rec.signal, 0.0)

    def test_single_column_file(self, tmp_path):
        """A file with only one column is read as signal."""
        signal = np.zeros(50)
        p = tmp_path / "single.csv"
        p.write_text("\n".join(str(v) for v in signal))
        out = tmp_path / "single.ssd"
        csv2ssd([str(p), "--dt-us", "10", "--signal-col", "0", "-o", str(out)])
        rec = ssd_read(out)
        assert rec.n_samples == 50

    def test_missing_file_fails(self, tmp_path):
        rc = csv2ssd([str(tmp_path / "no_such.csv"), "--dt-us", "20"])
        assert rc != 0

    def test_dt_us_required(self, tmp_path, simple_csv):
        # Neither --dt-us nor --infer-dt → argparse error → SystemExit
        with pytest.raises(SystemExit):
            csv2ssd([str(simple_csv)])

    def test_dt_us_out_of_range_fails(self, tmp_path, simple_csv):
        rc = csv2ssd([str(simple_csv), "--dt-us", "50000"])
        assert rc != 0


# ---------------------------------------------------------------------------
# Legacy real-file smoke test (scn2txt on an example file)
# ---------------------------------------------------------------------------


class TestLegacyCli:
    @pytest.fixture()
    def example_scn(self):
        p = EXAMPLES_SCN / "glyr_experimental.scn"
        if not p.exists():
            pytest.skip("glyr_experimental.scn not found.")
        return p

    def test_scn2txt_on_real_file(self, tmp_path, example_scn):
        out = tmp_path / "out.txt"
        rc = scn2txt([str(example_scn), "-o", str(out)])
        assert rc == 0
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 3885  # n_intervals in glyr_experimental.scn

    def test_scn2txt_with_tres_on_real_file(self, tmp_path, example_scn):
        out = tmp_path / "out.txt"
        rc = scn2txt([str(example_scn), "--tres", "40", "--periods", "-o", str(out)])
        assert rc == 0
        lines = out.read_text().strip().splitlines()
        assert len(lines) > 0

    @pytest.fixture()
    def example_txt(self):
        p = EXAMPLES_SCN / "glyr_simulated_source.txt"
        if not p.exists():
            pytest.skip("glyr_simulated_source.txt not found.")
        return p

    def test_txt2scn_on_example_source(self, tmp_path, example_txt):
        out = tmp_path / "set1.scn"
        rc = txt2scn([str(example_txt), "-o", str(out)])
        assert rc == 0
        rec = scn_read(out)
        assert rec.n_intervals > 40000  # set1 has ~47k intervals
