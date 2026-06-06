"""Viewer for SCN binary single-channel record files."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from dcio.formats.scn import (
    FLAG_AMPLITUDE_CONSTRAINED,
    FLAG_AMPLITUDE_DUBIOUS,
    FLAG_AMPLITUDE_FIXED,
    FLAG_SWEEP_FIRST,
    FLAG_SWEEP_LAST,
    FLAG_UNUSABLE,
    SCNRecord,
    read,
)

from .base import BaseViewer

# Rows shown in the data table before the user expands
_DEFAULT_PAGE = 500


class ScnViewer(BaseViewer):
    name = "SCN"
    extensions = ("scn",)
    description = "SCAN idealised single-channel record (DCProgs, UCL)"

    @classmethod
    def read(cls, data: bytes, filename: str) -> SCNRecord:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".scn") as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        return read(tmp_path)

    @classmethod
    def render(cls, record: SCNRecord) -> None:
        _render_scn(record)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_VERSION_LABELS = {
    -103: "-103  (simulated – old SCSIM format)",
     103: " 103  (simulated)",
     104: " 104  (experimental)",
}

_FLAG_DESCRIPTIONS = {
    0:                         "OK",
    FLAG_AMPLITUDE_DUBIOUS:    "amplitude dubious",
    FLAG_AMPLITUDE_FIXED:      "amplitude fixed",
    FLAG_AMPLITUDE_CONSTRAINED:"amplitude constrained",
    FLAG_UNUSABLE:             "unusable",
    FLAG_SWEEP_FIRST:          "sweep first (CJUMP)",
    FLAG_SWEEP_LAST:           "sweep last (CJUMP)",
}


def _flag_label(f: int) -> str:
    if f == 0:
        return "OK"
    parts = [desc for bit, desc in _FLAG_DESCRIPTIONS.items() if bit and (f & bit)]
    return " | ".join(parts) if parts else str(f)


def _render_scn(rec: SCNRecord) -> None:
    hdr = rec.header

    # ── top metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Version",    _VERSION_LABELS.get(hdr.version, str(hdr.version)))
    col2.metric("Type",       hdr.record_type.capitalize())
    col3.metric("Intervals",  f"{hdr.n_intervals:,}")
    col4.metric("Calibration", f"{hdr.calfac2:.4g} pA / ADC unit"
                               if hdr.calfac2 else "—")

    st.divider()

    # ── tabs ──────────────────────────────────────────────────────────────
    tab_summary, tab_data, tab_header = st.tabs(["Summary", "Intervals", "Header fields"])

    with tab_summary:
        _render_summary(rec)

    with tab_data:
        _render_data_table(rec)

    with tab_header:
        _render_header_fields(rec)


def _render_summary(rec: SCNRecord) -> None:
    hdr = rec.header

    good_mask  = (rec.flags & FLAG_UNUSABLE) == 0
    open_mask  = rec.open_mask  & good_mask
    shut_mask  = rec.shut_mask  & good_mask

    n_open  = int(open_mask.sum())
    n_shut  = int(shut_mask.sum())
    n_bad   = int((rec.flags & FLAG_UNUSABLE).astype(bool).sum())

    # Durations already in seconds
    mean_open_ms  = rec.intervals[open_mask].mean()  * 1e3 if n_open  else float("nan")
    mean_shut_ms  = rec.intervals[shut_mask].mean()  * 1e3 if n_shut  else float("nan")
    total_open_s  = rec.intervals[open_mask].sum()          if n_open  else 0.0
    total_shut_s  = rec.intervals[shut_mask].sum()          if n_shut  else 0.0

    st.subheader("Interval counts")
    c1, c2, c3 = st.columns(3)
    c1.metric("Open (usable)",  f"{n_open:,}")
    c2.metric("Shut (usable)",  f"{n_shut:,}")
    c3.metric("Unusable",       f"{n_bad:,}")

    st.subheader("Durations")
    c1, c2 = st.columns(2)
    c1.metric("Mean open",  f"{mean_open_ms:.3f} ms" if n_open  else "—")
    c2.metric("Mean shut",  f"{mean_shut_ms:.3f} ms" if n_shut  else "—")
    c1.metric("Total open time",  f"{total_open_s:.3f} s")
    c2.metric("Total shut time",  f"{total_shut_s:.3f} s")

    if n_open:
        open_amp = rec.amplitudes[open_mask]
        c1.metric("Mean open amplitude", f"{open_amp.mean():.2f} pA")

    st.subheader("Flag distribution")
    unique, counts = np.unique(rec.flags.astype(int), return_counts=True)
    flag_df = pd.DataFrame({
        "Flag value": unique,
        "Label":      [_flag_label(int(f)) for f in unique],
        "Count":      counts,
        "Percent":    [f"{100 * c / len(rec.flags):.1f} %" for c in counts],
    })
    st.dataframe(flag_df, use_container_width=True, hide_index=True)

    # ── amplitude histogram ───────────────────────────────────────────────
    if rec.header.record_type == "experimental" and n_open > 0:
        st.subheader("Open amplitude distribution")
        amps = rec.amplitudes[open_mask]
        _amplitude_histogram(amps)


def _amplitude_histogram(amps: np.ndarray) -> None:
    """Simple bar chart of amplitude histogram using Streamlit."""
    counts, edges = np.histogram(amps, bins=40)
    centres = 0.5 * (edges[:-1] + edges[1:])
    df = pd.DataFrame({"Amplitude (pA)": centres, "Count": counts})
    st.bar_chart(df.set_index("Amplitude (pA)"), use_container_width=True)


def _render_data_table(rec: SCNRecord) -> None:
    st.caption(
        f"{rec.header.n_intervals:,} intervals total.  "
        f"Durations in milliseconds, amplitudes in pA."
    )

    # Build full DataFrame (vectorised — fast even for 47k rows)
    ivl_ms = rec.intervals * 1e3
    df = pd.DataFrame({
        "#":           np.arange(1, len(ivl_ms) + 1),
        "Duration ms": np.round(ivl_ms, 4),
        "Amplitude pA": np.round(rec.amplitudes.astype(float), 3),
        "Flag":        rec.flags.astype(int),
        "State":       np.where(rec.open_mask, "open", "shut"),
        "Flag label":  [_flag_label(int(f)) for f in rec.flags],
    })

    total = len(df)
    page_size = st.select_slider(
        "Rows to display",
        options=[100, 500, 1000, 5000, total],
        value=min(_DEFAULT_PAGE, total),
        format_func=lambda x: f"All ({total:,})" if x == total else f"{x:,}",
    )

    st.dataframe(
        df.head(page_size),
        use_container_width=True,
        hide_index=True,
        column_config={
            "#":            st.column_config.NumberColumn(width="small"),
            "Duration ms":  st.column_config.NumberColumn(format="%.4f", width="small"),
            "Amplitude pA": st.column_config.NumberColumn(format="%.3f", width="small"),
            "Flag":         st.column_config.NumberColumn(width="small"),
            "State":        st.column_config.TextColumn(width="small"),
            "Flag label":   st.column_config.TextColumn(width="medium"),
        },
    )

    if page_size < total:
        st.caption(f"Showing first {page_size:,} of {total:,} rows.")

    # Download
    csv = df.to_csv(index=False).encode()
    st.download_button(
        "Download full table as CSV",
        data=csv,
        file_name="intervals.csv",
        mime="text/csv",
    )


def _render_header_fields(rec: SCNRecord) -> None:
    hdr = rec.header

    rows = [
        ("version",     hdr.version,   "SCAN version tag"),
        ("record_type", hdr.record_type, "simulated or experimental"),
        ("n_intervals", hdr.n_intervals, "Number of intervals in data block"),
        ("data_offset", hdr.data_offset, "Byte offset to start of data block"),
        ("title",       hdr.title or "—", "Free-text title"),
        ("date",        hdr.date or "—",  "Date string from header"),
        ("calfac2",     f"{hdr.calfac2:.6g} pA/ADC" if hdr.calfac2 else "—",
                        "Calibration factor (ADC intermediate units → pA)"),
        ("avamp",       f"{hdr.avamp:.4g}" if hdr.avamp else "—",
                        "Mean amplitude at analysis time (ADC units)"),
        ("rms",         f"{hdr.rms:.4g}"  if hdr.rms  else "—",
                        "RMS baseline noise (ADC units)"),
        ("ffilt",       f"{hdr.ffilt:.1f} Hz" if hdr.ffilt and hdr.ffilt > 0 else "—",
                        "Low-pass filter cut-off"),
        ("Emem",        f"{hdr.Emem:.1f} mV" if hdr.Emem else "—",
                        "Membrane potential"),
        ("ipatch",      hdr.ipatch if hdr.ipatch else "—", "Patch number"),
        ("tapeID",      hdr.tapeID or "—", "Tape / file identifier"),
    ]

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
