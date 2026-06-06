# SCN example files

These files can be loaded with `dcio.formats.scn.read()`.

## Experimental record — `glyr_experimental.scn`

Version 104 (SCAN experimental).  Real single-channel recording: GlyR α2,
1 mM taurine, –100 mV.  3 885 intervals.  Carries a full header (membrane
potential, filter frequency, time resolution) and a trailing sentinel
interval flagged unusable.

## Simulated record — `glyr_simulated.scn`

Version –103 (SCAN simulated).  Idealised interval list with unit amplitudes:
0 pA = shut, 1 pA = open.  47 783 intervals.  Suitable for large-scale
algorithm tests and benchmarking.

## Source text — `glyr_simulated_source.txt`

Tab-separated input that was converted to `glyr_simulated.scn`:

```
column 0  – amplitude class (0 = shut, 1 = open)
column 1  – interval duration in milliseconds
```

Used by `dcio-txt2scn` integration tests.

## Quick start

```python
from dcio.formats.scn import read
from dcio.analysis.record import from_scn

# Experimental record — apply 40 µs dead time
rec = read("examples/scn/glyr_experimental.scn")
scr = from_scn(rec, tres=40e-6)
print(scr)

open_ivl = rec.intervals[rec.open_mask]   # open intervals in seconds
shut_ivl = rec.intervals[rec.shut_mask]   # shut intervals in seconds

# Simulated record
rec2 = read("examples/scn/glyr_simulated.scn")
print(f"{rec2.n_intervals} intervals, {rec2.open_mask.sum()} open")
```
