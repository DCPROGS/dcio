"""DCProgs file viewer — Streamlit application entry point.

Run from the dcio project root::

    streamlit run viewer/app.py

Then open the URL shown in the terminal (usually http://localhost:8501).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from viewer.registry import all_extensions, all_viewers, viewer_for_extension

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DCProgs File Viewer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔬 DCProgs Viewer")
    st.caption("Single-channel electrophysiology file browser")
    st.divider()

    # Accepted extensions
    exts = all_extensions()
    accepted = [f".{e}" for e in exts]

    uploaded = st.file_uploader(
        "Upload a file",
        type=exts,
        help=f"Supported formats: {', '.join(f'.{e}' for e in exts)}",
    )

    st.divider()
    st.subheader("Supported formats")
    for viewer in all_viewers():
        exts_str = ", ".join(f"`.{e}`" for e in viewer.extensions)
        st.markdown(f"**{viewer.name}** {exts_str}  \n{viewer.description}")

    st.divider()
    st.caption("DCProgs · UCL Colquhoun lab")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if uploaded is None:
    # Landing page
    st.title("DCProgs File Viewer")
    st.markdown(
        """
        Upload a file using the sidebar to inspect its contents.

        ### Supported formats

        | Format | Extension | Description |
        |--------|-----------|-------------|
        """
        + "\n".join(
            f"| **{v.name}** | `{'`, `'.join('.' + e for e in v.extensions)}` "
            f"| {v.description} |"
            for v in all_viewers()
        )
    )
    st.info("👈  Use the sidebar to upload a file.")
    st.stop()

# Identify format from extension
suffix = Path(uploaded.name).suffix.lstrip(".").lower()
viewer_cls = viewer_for_extension(suffix)

if viewer_cls is None:
    st.error(
        f"No viewer registered for `.{suffix}` files.  "
        f"Supported: {', '.join(f'.{e}' for e in all_extensions())}"
    )
    st.stop()

# ── file info strip ────────────────────────────────────────────────────────
st.title(uploaded.name)
cols = st.columns([2, 1, 1])
cols[0].caption(f"**Format:** {viewer_cls.name} — {viewer_cls.description}")
cols[1].caption(f"**Size:** {uploaded.size:,} bytes")
cols[2].caption(f"**Viewer:** {viewer_cls.__name__}")

st.divider()

# ── parse & render ─────────────────────────────────────────────────────────
data = uploaded.read()

with st.spinner(f"Reading {uploaded.name} …"):
    try:
        record = viewer_cls.read(data, uploaded.name)
    except Exception as exc:
        st.error(f"**Failed to read file:** {exc}")
        with st.expander("Full traceback"):
            import traceback
            st.code(traceback.format_exc(), language="python")
        st.stop()

try:
    viewer_cls.render(record)
except Exception as exc:
    st.error(f"**Rendering error:** {exc}")
    with st.expander("Full traceback"):
        import traceback
        st.code(traceback.format_exc(), language="python")
