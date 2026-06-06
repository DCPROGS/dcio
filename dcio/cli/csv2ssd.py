"""dcio-csv2ssd — convert a CSV or text file to a CONSAM SSD file.

The input file must contain at least one numeric column holding the signal
in pA.  A second column with sample times can be used to infer the sampling
interval automatically.

Usage examples::

    dcio-csv2ssd trace.csv --dt-us 20
    dcio-csv2ssd trace.csv --dt-us 20 --gain 0.05 --filt 3000
    dcio-csv2ssd trace.csv --infer-dt --time-col 0 --signal-col 1
    dcio-csv2ssd trace.csv --dt-us 20 -o output.ssd --Emem -80 --temp 22

Input format (default: comma-separated, signal in column 1)::

    0.000000,  -1.234
    0.000020,  -1.187
    0.000040,  -1.309
    …

Single-column files (signal only) are also supported with --signal-col 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dcio.formats.ssd import write as ssd_write


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dcio-csv2ssd",
        description=(
            "Convert a CSV or text file to a CONSAM SSD file.\n\n"
            "Provide --dt-us or --infer-dt (requires a time column)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", metavar="FILE.csv", help="Input CSV or text file.")
    p.add_argument(
        "-o", "--output", metavar="FILE.ssd",
        help="Output SSD file (default: input basename with .ssd extension).",
    )

    # Sampling interval
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--dt-us", metavar="MICROSECONDS", type=float,
        help="Sampling interval in microseconds (e.g. 20 for 50 kHz).",
    )
    grp.add_argument(
        "--infer-dt", action="store_true",
        help="Infer dt from the time column (requires --time-col).",
    )

    # Column layout
    p.add_argument(
        "--signal-col", metavar="N", type=int, default=1,
        help="Column index for the signal in pA. Default: 1.",
    )
    p.add_argument(
        "--time-col", metavar="N", type=int, default=0,
        help="Column index for sample times (used with --infer-dt). Default: 0.",
    )
    p.add_argument(
        "--delimiter", metavar="CHAR", default=",",
        help="Column delimiter. Default: comma.",
    )
    p.add_argument(
        "--skip-header", metavar="N", type=int, default=0,
        help="Number of header rows to skip. Default: 0.",
    )

    # Physical parameters
    p.add_argument(
        "--gain", metavar="V/pA", type=float, default=1.0,
        help="Transducer gain in V/pA. Default: 1.0.",
    )
    p.add_argument(
        "--filt", metavar="HZ", type=float, default=1000.0,
        help="Low-pass filter cut-off in Hz stored in header. Default: 1000.",
    )
    p.add_argument(
        "--Emem", metavar="MV", type=float, default=0.0,
        help="Membrane potential in mV stored in header. Default: 0.",
    )
    p.add_argument(
        "--temp", metavar="CELSIUS", type=float, default=0.0,
        help="Bath temperature in °C stored in header. Default: 0.",
    )
    p.add_argument(
        "--title", metavar="TEXT", default="",
        help="Title string stored in the SSD header (max 70 characters).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"error: file not found: {in_path}", file=sys.stderr)
        return 1

    out_path = Path(args.output) if args.output else in_path.with_suffix(".ssd")

    # Load data
    delim = args.delimiter if args.delimiter.strip() else None
    try:
        data = np.genfromtxt(
            in_path,
            delimiter=delim,
            skip_header=args.skip_header,
        )
    except Exception as exc:
        print(f"error: could not read {in_path}: {exc}", file=sys.stderr)
        return 1

    if data.ndim == 1:
        # Single-column file — treat as signal
        data = data.reshape(-1, 1)
        signal_col = 0
    else:
        signal_col = args.signal_col

    if signal_col >= data.shape[1]:
        print(
            f"error: --signal-col {signal_col} out of range "
            f"(file has {data.shape[1]} columns)",
            file=sys.stderr,
        )
        return 1

    signal = data[:, signal_col].astype(np.float64)

    # Determine sampling interval
    if args.infer_dt:
        time_col = args.time_col
        if data.ndim == 1 or time_col >= data.shape[1]:
            print(
                f"error: --infer-dt requires a time column; "
                f"--time-col {time_col} is out of range",
                file=sys.stderr,
            )
            return 1
        times = data[:, time_col]
        if len(times) < 2:
            print("error: need at least 2 rows to infer dt", file=sys.stderr)
            return 1
        dt_s = float(np.median(np.diff(times)))
        dt_us = dt_s * 1e6
        print(f"Inferred dt = {dt_us:.2f} µs from time column {time_col}")
    else:
        dt_us = args.dt_us

    dt_us_int = int(round(dt_us))
    if not (1 <= dt_us_int <= 32767):
        print(
            f"error: dt_us={dt_us_int} out of valid range 1–32767 µs",
            file=sys.stderr,
        )
        return 1

    try:
        ssd_write(
            out_path,
            signal,
            dt_us_int,
            gain=args.gain,
            filt=args.filt,
            Emem=args.Emem,
            temp=args.temp,
            title=args.title,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    srate = 1e6 / dt_us_int
    print(
        f"Wrote {len(signal)} samples "
        f"(dt={dt_us_int} µs, {srate/1e3:.1f} kHz) → {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
