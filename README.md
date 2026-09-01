# dcio

I/O library for electrophysiology file formats used by the
[DCProgs](http://www.ucl.ac.uk/Pharmacology/dcpr95.html) single-channel
analysis suite.

## Supported formats

| Format | Extension | Read | Write | Description |
|--------|-----------|------|-------|-------------|
| SCAN   | `.scn`    | ✓    | ✓     | Idealised single-channel records |

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10 and NumPy ≥ 1.24.

## Quick start

```python
from dcio.formats.scn import read, write
import numpy as np

# --- Load an existing file ---
rec = read("myrecording.scn")
print(rec)
# SCNRecord('myrecording.scn')
#   version      : -103 (simulated)
#   total        : 1842 intervals
#   usable       : 1841
#   open / shut  : 921 / 920
#   mean open    : 2.3147 ms
#   mean shut    : 8.0412 ms
#   calfac2      : 1 pA/ADC
#   filter       : 3000 Hz

# Intervals are in seconds; amplitudes in pA
print(rec.intervals[:5])   # array([0.01031, 0.00198, 0.02047, ...])
print(rec.amplitudes[:5])  # array([0., -48., 0., -48., 0.])

# Filter out unusable intervals
from dcio.formats.scn import FLAG_UNUSABLE
good = rec.usable_mask
open_durations_ms = rec.intervals[rec.open_mask & good] * 1e3

# --- Write a new file ---
intervals_s = np.array([0.010, 0.002, 0.020, 0.001, 0.050])
amplitudes  = np.array([0,     -48,   0,     -48,   0    ], dtype=float)
flags       = np.zeros(5, dtype=np.int8)

write("output.scn", intervals_s, amplitudes, flags,
      title="My simulated record", ffilt=3000.0)
```

## SCNRecord attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `intervals` | `ndarray[float64]` | Interval durations in **seconds** |
| `amplitudes` | `ndarray[float64]` | Amplitudes in **pA** (0 = shut) |
| `flags` | `ndarray[int8]` | Property flags; `& 8 != 0` → unusable |
| `header` | `SCNHeader` | Parsed file metadata |
| `path` | `Path` | Source file path |
| `open_mask` | `ndarray[bool]` | True for open intervals |
| `shut_mask` | `ndarray[bool]` | True for shut intervals |
| `usable_mask` | `ndarray[bool]` | True when `flags & 8 == 0` |

## Record analysis

Beyond file I/O, `dcio.analysis` carries the record layer shared by the rest of
the DCPROGS stack: applying a dead time, grouping the result into open and shut
periods, and cutting it into bursts.

```python
from dcio.formats.scn import read
from dcio.analysis import from_scn, bursts_from_record

record = from_scn(read("myrecording.scn"), tres=25e-6)   # 25 us dead time
lengths, n_openings = bursts_from_record(record, tcrit=4e-3)

print(len(lengths), "bursts,", n_openings.mean(), "openings each")
```

`bursts_from_record` segments the record's **periods**, not its resolved
intervals. `impose_resolution` emits a fresh open interval at every change of
fitted amplitude, so a record idealised with sub-conductance levels contains
runs of consecutive openings; segmenting those directly gives bursts that do
not alternate open/shut. Burst count and lengths are the same either way, the
number of openings per burst is not.

Two conventions are worth stating, because implementations in this stack have
differed on them:

- no gap longer than `tcrit` is required before the first burst -- the first
  defined opening starts one;
- an unusable interval ends the burst before it, and its (meaningless)
  duration is never compared with `tcrit`.

Both follow SCAN's time-course fitting, which leaves the final interval of a
record with no defined length. Dropping the runs at each end instead loses two
bursts from every record.

## Running tests

```bash
cd dcio          # from the dcprogs root
pytest
```

Tests cover write validation, round-trip fidelity, flag handling, edge cases,
and optional smoke tests against legacy SCN files in `examples/scn/`.

## Documentation

See [`docs/scn_format.md`](docs/scn_format.md) for a complete description of
the binary file format, header fields, and unit conventions.
