"""dcio-txt2scn — convert a two-column text file to a SCAN SCN file (version –103).

Expected input format (tab- or whitespace-separated, no header)::

    0   1.1015
    1   17.1992
    0   0.3008
    1   49.1992

Column 0: amplitude class — 0 = shut, 1 = open.
Column 1: interval duration (milliseconds by default; use --unit s for seconds).

The output SCN file uses version –103 (simulated) with amplitude 0 pA for shut
and 1 pA for open intervals.  All flags are set to 0 (usable).

Usage examples::

    dcio-txt2scn set1.txt
    dcio-txt2scn set1.txt -o set1.scn
    dcio-txt2scn data.txt --unit s --title "GlyR simulation"
    dcio-txt2scn data.txt --amp-col 0 --ivl-col 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dcio.formats.scn import write as scn_write


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dcio-txt2scn",
        description=(
            "Convert a two-column text file to a SCAN SCN file (version –103).\n\n"
            "Default column layout: col 0 = amplitude class (0/1), col 1 = interval (ms)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", metavar="FILE.txt", help="Input text file.")
    p.add_argument(
        "-o", "--output", metavar="FILE.scn",
        help="Output SCN file (default: input basename with .scn extension).",
    )
    p.add_argument(
        "--unit", choices=["ms", "s"], default="ms",
        help="Unit of the interval column. Default: ms (milliseconds).",
    )
    p.add_argument(
        "--amp-col", metavar="N", type=int, default=0,
        help="Column index for amplitude class (0/1). Default: 0.",
    )
    p.add_argument(
        "--ivl-col", metavar="N", type=int, default=1,
        help="Column index for interval duration. Default: 1.",
    )
    p.add_argument(
        "--delimiter", metavar="CHAR", default=None,
        help="Column delimiter. Default: any whitespace.",
    )
    p.add_argument(
        "--title", metavar="TEXT",
        default="Converted from text file containing event list: amplitudes, intervals.",
        help="Title string stored in the SCN header (max 70 characters).",
    )
    p.add_argument(
        "--calfac", metavar="FLOAT", type=float, default=1.0,
        help="Calibration factor pA/ADC-unit stored in header. Default: 1.0.",
    )
    p.add_argument(
        "--ffilt", metavar="HZ", type=float, default=-1.0,
        help="Filter frequency (Hz) stored in header. Default: –1 (unknown).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"error: file not found: {in_path}", file=sys.stderr)
        return 1

    out_path = Path(args.output) if args.output else in_path.with_suffix(".scn")

    # Load data
    delim = args.delimiter
    try:
        if delim:
            data = np.genfromtxt(in_path, delimiter=delim)
        else:
            data = np.genfromtxt(in_path)
    except Exception as exc:
        print(f"error: could not read {in_path}: {exc}", file=sys.stderr)
        return 1

    if data.ndim != 2 or data.shape[1] < max(args.amp_col, args.ivl_col) + 1:
        print(
            f"error: expected at least {max(args.amp_col, args.ivl_col) + 1} columns, "
            f"got {data.shape[1] if data.ndim == 2 else 1}",
            file=sys.stderr,
        )
        return 1

    amp_class = data[:, args.amp_col].astype(int)   # 0 or 1
    ivl_raw   = data[:, args.ivl_col]               # ms or s

    # Convert intervals to seconds
    intervals  = ivl_raw * 1e-3 if args.unit == "ms" else ivl_raw.copy()
    amplitudes = amp_class.astype(np.float64)        # 0.0 or 1.0 pA
    flags      = np.zeros(len(intervals), dtype=np.int8)

    if np.any(intervals <= 0):
        n_bad = int(np.sum(intervals <= 0))
        print(f"warning: {n_bad} non-positive intervals found; they will be flagged unusable.")
        flags[intervals <= 0] = 8

    scn_write(
        out_path, intervals, amplitudes, flags,
        calfac=args.calfac,
        ffilt=args.ffilt,
        title=args.title,
    )

    print(f"Wrote {len(intervals)} intervals → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
