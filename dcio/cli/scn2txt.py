"""dcio-scn2txt — convert an SCN file to a tab-separated text file.

Usage examples::

    dcio-scn2txt patch.scn
    dcio-scn2txt patch.scn -o result.txt
    dcio-scn2txt patch.scn --tres 40 -o resolved.txt
    dcio-scn2txt patch.scn --tres 40 --periods -o periods.txt

Output columns (default, no --tres):
    interval_s   amplitude_pA   flag

Output columns with --tres (resolved intervals):
    interval_s   amplitude_pA   flag

Output columns with --tres --periods:
    interval_s   amplitude_pA   flag
    (alternating open/shut periods; see dcio.analysis.record.set_periods)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dcio.formats.scn import read as scn_read
from dcio.analysis.record import from_scn


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dcio-scn2txt",
        description=(
            "Convert a SCAN SCN file to a tab-separated text file.\n\n"
            "Default output columns: interval_s  amplitude_pA  flag\n"
            "With --tres: dead-time resolution is applied first.\n"
            "With --tres --periods: write open/shut period lists instead."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", metavar="FILE.scn", help="Input SCN file.")
    p.add_argument(
        "-o", "--output", metavar="FILE.txt",
        help="Output file (default: input basename with .txt extension).",
    )
    p.add_argument(
        "--tres", metavar="MICROSECONDS", type=float, default=0.0,
        help="Dead time in microseconds.  Activates impose_resolution. Default: 0 (off).",
    )
    p.add_argument(
        "--periods", action="store_true",
        help=(
            "Write open/shut period lists instead of the resolved interval list. "
            "Requires --tres."
        ),
    )
    p.add_argument(
        "--badopen", metavar="SECONDS", type=float, default=0.0,
        help="Flag unusable any opening longer than this (seconds). Default: 0 (off).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"error: file not found: {in_path}", file=sys.stderr)
        return 1

    out_path = Path(args.output) if args.output else in_path.with_suffix(".txt")

    if args.periods and args.tres == 0.0:
        print("error: --periods requires --tres > 0", file=sys.stderr)
        return 1

    rec = scn_read(in_path)

    if args.tres > 0.0 or args.periods:
        scr = from_scn(rec, tres=args.tres * 1e-6, badopen=args.badopen)
        if args.periods:
            ivl = scr.periods.intervals
            amp = scr.periods.amplitudes
            flg = scr.periods.flags
        else:
            ivl = scr.resolved_intervals
            amp = scr.resolved_amplitudes
            flg = scr.resolved_flags
    else:
        ivl = rec.intervals
        amp = rec.amplitudes
        flg = rec.flags

    with open(out_path, "w") as fh:
        for t, a, f in zip(ivl, amp, flg):
            fh.write(f"{t:.16e}\t{a:.6f}\t{int(f)}\n")

    n = len(ivl)
    label = "periods" if args.periods else "intervals"
    print(f"Wrote {n} {label} → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
