"""Viewer for SCAN.INI binary settings files."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from dcio.formats.scan_ini import (
    ISCRIT_FRACTION_AMP,
    ISCRIT_MULTIPLE_RMS,
    ScanIniRecord,
    read,
)

from .base import BaseViewer


class ScanIniViewer(BaseViewer):
    name = "SCAN.INI"
    extensions = ("ini",)
    description = "SCAN application settings file (DCProgs, UCL)"

    @classmethod
    def read(cls, data: bytes, filename: str) -> ScanIniRecord:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".INI") as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        return read(tmp_path)

    @classmethod
    def render(cls, record: ScanIniRecord) -> None:
        _render_scan_ini(record)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_scan_ini(ini: ScanIniRecord) -> None:
    # ── header strip ──────────────────────────────────────────────────────
    size_label = "512 bytes (new)" if ini.ini_size == 512 else "256 bytes (old)"
    col1, col2, col3 = st.columns(3)
    col1.metric("Format", "SCAN.INI")
    col2.metric("File size", size_label)
    col3.metric("Saved on exit", ini.savin)

    st.divider()

    # ── key analysis settings ─────────────────────────────────────────────
    st.subheader("Key analysis settings")

    if ini.iscrit == ISCRIT_FRACTION_AMP:
        threshold_str = f"{ini.smult * 100:.1f} % of mean amplitude"
        criterion_str = "Fraction of mean amplitude (iscrit = 1)"
    elif ini.iscrit == ISCRIT_MULTIPLE_RMS:
        threshold_str = f"{ini.smult:.2f} × RMS noise"
        criterion_str = "Multiple of RMS noise (iscrit = 2)"
    else:
        threshold_str = str(ini.smult)
        criterion_str = f"Unknown (iscrit = {ini.iscrit})"

    key_rows = [
        ("adcfil",    ini.adcfil,                        "Source ADC data file path (≤ 30 chars)"),
        ("opendown",  ini.opendown,                       "Openings appear as downward deflections"),
        ("cjump",     ini.cjump,                          "Concentration-jump (CJUMP) data"),
        ("iscrit",    criterion_str,                      "Threshold criterion"),
        ("smult",     threshold_str,                      "Detection threshold multiplier"),
        ("ntrig",     ini.ntrig,                          "Consecutive points above threshold to confirm transition"),
        ("tmin",      f"{ini.tmin:.1f} µs",               "Min interval — re-fit suggested below this"),
    ]
    _field_table(key_rows)

    # ── buffer / data I/O ─────────────────────────────────────────────────
    with st.expander("Buffer & I/O"):
        buf_rows = [
            ("nbuf",    ini.nbuf,    "Points held in memory per section"),
            ("novlap",  ini.novlap,  "Overlap between adjacent sections (points)"),
            ("nwrit",   ini.nwrit,   "Write to disc every nth transition"),
            ("ndevdat", ini.ndevdat, "Default device / disk identifier"),
            ("savin",   ini.savin,   "'Y' if parameters were saved on last exit"),
        ]
        _field_table(buf_rows)

    # ── display & gain ────────────────────────────────────────────────────
    with st.expander("Display & gain"):
        disp_rows = [
            ("invert",    ini.invert,               "Trace inverted for display"),
            ("dgain",     f"{ini.dgain:.3g}",        "Display gain factor"),
            ("iboff",     ini.iboff,                 "Baseline offset for display (ADC units)"),
            ("bdisp",     f"{ini.bdisp * 100:.1f} %","Baseline position on Y axis"),
            ("xtrig",     f"{ini.xtrig * 100:.1f} %","Trigger position on X axis"),
            ("isub",      ini.isub,                  "Sub-level auto-fit mode (0/1/2)"),
            ("iautosub",  ini.iautosub,              "Auto-fit sub-level mode"),
            ("disp",      ini.disp,                  "Display-only mode (no fitting)"),
            ("disptran",  ini.disptran,              "Show guessed transition points"),
            ("dispderiv", ini.dispderiv,             "Show first derivative on screen"),
            ("dispguess", ini.dispguess,             "Show fit guesses before fitting"),
        ]
        _field_table(disp_rows)

    # ── zoom ──────────────────────────────────────────────────────────────
    with st.expander("Zoom"):
        zoom_rows = [
            ("izoom", ini.izoom,              "Default zoom factor (power of 2)"),
            ("fcz",   f"{ini.fcz * 1000:.0f} Hz", "Filter cut-off when zoomed"),
            ("ampz",  f"{ini.ampz:.2f} pA",   "'Full amplitude' for zoomed detection"),
        ]
        _field_table(zoom_rows)

    # ── fitting parameters ────────────────────────────────────────────────
    with st.expander("Fitting parameters"):
        fit_rows = [
            ("minmeth",  ini.minmeth,           "Minimisation method (1=Simplex, 2=Simplex→DFPMIN, 3=DFPMIN)"),
            ("itsimp",   ini.itsimp,            "Simplex iterations before switching to DFPMIN"),
            ("confac",   f"{ini.confac:.3f}",   "Simplex contraction factor"),
            ("stpfac",   _fmt_opt(ini.stpfac),  "Simplex initial step-size factor"),
            ("errfac",   f"{ini.errfac:.4f}",   "Convergence criterion"),
            ("derivfac", f"{ini.derivfac:.2f}", "Derivative SD multiplier for inflection detection"),
            ("nbasemin", ini.nbasemin,           "Min baseline points for amplitude estimate"),
            ("nshutfit", getattr(ini, "nshutfit", "—"), "Shut points fitted at each end"),
            ("ampfac",   f"{ini.ampfac:.3f}",   "Fraction of full amplitude for 'same amplitude' test"),
            ("tsfac",    f"{ini.tsfac:.2f}",    "Multiple of risetime → 'short' interval"),
            ("tlfac",    f"{ini.tlfac:.2f}",    "Multiple of risetime → 'long' interval"),
            ("tcfac",    f"{ini.tcfac:.2f}",    "Multiple of risetime → 'close' transition"),
            ("facjump",  f"{ini.facjump:.2f}",  "Fraction of filter length to jump after transition"),
        ]
        _field_table(fit_rows)

    # ── variable threshold ────────────────────────────────────────────────
    if ini.ini_size == 512:
        with st.expander("Variable threshold (512-byte format)"):
            var_rows = [
                ("scritvar", ini.scritvar,             "Use reduced threshold for small-amplitude events"),
                ("smultmin", _fmt_opt(ini.smultmin),   "Minimum threshold multiple of RMS when scritvar active"),
            ]
            _field_table(var_rows)

    # ── amplitude markers ─────────────────────────────────────────────────
    if ini.nampmark > 0:
        with st.expander(f"Amplitude markers ({ini.nampmark} set)"):
            amp_rows = [
                (f"iamark[{i}]", f"{v} ADC units", "")
                for i, v in enumerate(ini.iamark[:ini.nampmark])
            ]
            _field_table(amp_rows)

    # ── filter file ───────────────────────────────────────────────────────
    with st.expander("Filter"):
        filt_rows = [
            ("filtfile", ini.filtfile or "—", "Filter definition file name"),
            ("nsetup",   ini.nsetup,           "Filter setup index"),
            ("navtest",  ini.navtest,           "Points averaged before accepting baseline"),
        ]
        _field_table(filt_rows)


def _field_table(rows: list[tuple]) -> None:
    """Render a list of (field, value, description) rows as a st.dataframe."""
    import pandas as pd
    df = pd.DataFrame(rows, columns=["Field", "Value", "Description"])
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Field":       st.column_config.TextColumn(width="small"),
            "Value":       st.column_config.TextColumn(width="medium"),
            "Description": st.column_config.TextColumn(width="large"),
        },
    )


def _fmt_opt(val) -> str:
    """Format an optional float; show '—' for None or NaN."""
    import math
    if val is None:
        return "—"
    try:
        if math.isnan(val) or math.isinf(val):
            return "— (invalid)"
    except TypeError:
        pass
    return f"{val:.4g}"
