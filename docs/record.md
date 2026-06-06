# Single-channel record analysis

`dcio.analysis.record` provides dead-time filtering and open/shut period
extraction for idealised single-channel records loaded from SCN files.

---

## Overview

An SCN file contains a raw idealised interval list: alternating open and shut
durations with amplitudes and flags, as idealised by SCAN.  Before fitting
dwell-time distributions the list must be post-processed in two steps:

1. **Impose dead time** (`impose_resolution`) — intervals shorter than a
   threshold *t*<sub>res</sub> are concatenated into neighbouring intervals.
   The dead time is set by the recording bandwidth; a typical value is
   40–100 µs.

2. **Extract periods** (`set_periods`) — consecutive openings at different
   sub-conductance levels are merged into *open periods*; the alternating
   open/shut sequence is split into two separate arrays ready for fitting.

The main entry point is `from_scn`, which performs both steps from an
`SCNRecord` and returns a `SingleChannelRecord`.

---

## Quick start

```python
from dcio.formats.scn import read
from dcio.analysis.record import from_scn

rec = read("examples/scn/glyr_experimental.scn")

# Apply 40 µs dead time
scr = from_scn(rec, tres=40e-6)

print(scr)
# SingleChannelRecord('glyr_experimental.scn')
#   record_type : experimental
#   tres        : 40.0 µs
#   n_raw       : 3885
#   n_resolved  : ...
#   open periods: ...  mean ... ms
#   shut periods: ...  mean ... ms

# Dwell-time distributions
import numpy as np
open_ms  = scr.open_periods  * 1e3    # seconds → ms
shut_ms  = scr.shut_periods  * 1e3

print(f"Mean open:  {open_ms.mean():.3f} ms")
print(f"Mean shut:  {shut_ms.mean():.3f} ms")
```

---

## `impose_resolution`

```python
impose_resolution(
    intervals, amplitudes, flags, tres,
    *, record_type="experimental", badopen=0.0
) -> (intervals, amplitudes, flags)
```

Concatenates intervals shorter than `tres` into their neighbours.

### Concatenation rules

| State | Sub-resolution interval | Action |
|-------|------------------------|--------|
| Shut | Any | Extend shut duration by adding the brief interval |
| Open | Brief shut (terminated by open) | Extend open, add duration to open accumulator |
| Open | Brief open | Extend open, add *amplitude × time* to accumulator |

At a resolvable transition:

* **Open → shut**: emit the accumulated open interval with amplitude =
  *Σ(aᵢ tᵢ) / Σtᵢ* (time-weighted mean) for experimental records, or the
  raw amplitude for simulated records.
* **Shut → open**: emit the accumulated shut duration.

### Amplitude conventions

| `record_type` | Open amplitude |
|---|---|
| `"simulated"` | Raw amplitude from file (typically 0 or 1 pA) |
| `"experimental"` | Time-weighted mean across any sub-level changes |

### Flags

The last interval in the output is always flagged unusable (flag bit 3 = 8).
An unfinished opening at the end of the record gets duration −1 as a sentinel.

### `badopen`

If `badopen > 0`, any open interval longer than `badopen` seconds is flagged
unusable.  This discards spuriously long openings (e.g. missed closings).

---

## `set_periods`

```python
set_periods(intervals, amplitudes, flags) -> Periods
```

Groups the resolved interval list into alternating open and shut periods.

### Algorithm

1. **Trim** leading shut, trailing unfinished opening (−1 sentinel), and
   trailing shuts.
2. Start with the first interval as an open period.
3. Each resolvable shut that follows a good open emits the accumulated open
   period and starts a new shut period.
4. Any consecutive opening at a different sub-conductance amplitude (from
   experimental data) extends the current open period.
5. A single unusable opening (flag ≥ 8) contaminates the entire open period;
   it is discarded and the preceding shut is also marked unusable.

### Output layout

```
Periods.intervals = [open₀, shut₀, open₁, shut₁, …, openₙ]
                     [0::2] = open periods
                     [1::2] = shut periods
```

The sequence always starts and ends with an open period.

---

## `Periods` dataclass

| Attribute | Type | Description |
|-----------|------|-------------|
| `intervals` | `ndarray` (s) | All period durations, alternating open/shut |
| `amplitudes` | `ndarray` (pA) | Time-weighted mean amplitude per period; 0 for shut |
| `flags` | `ndarray` (int8) | Flag bytes (bit 3 set = unusable) |
| `open_intervals` | property | `intervals[0::2]` |
| `open_amplitudes` | property | `amplitudes[0::2]` |
| `open_flags` | property | `flags[0::2]` |
| `shut_intervals` | property | `intervals[1::2]` |
| `shut_amplitudes` | property | `amplitudes[1::2]` |
| `shut_flags` | property | `flags[1::2]` |
| `n_open` | property | Number of open periods |
| `n_shut` | property | Number of shut periods |

---

## `SingleChannelRecord` dataclass

| Attribute | Type | Description |
|-----------|------|-------------|
| `intervals` | `ndarray` (s) | Raw intervals from SCN file |
| `amplitudes` | `ndarray` (pA) | Raw amplitudes |
| `flags` | `ndarray` (int8) | Raw flags |
| `record_type` | `str` | `'simulated'` or `'experimental'` |
| `tres` | `float` (s) | Dead time applied (0 = none) |
| `path` | `Path` or `None` | Source file path |
| `resolved_intervals` | `ndarray` (s) | After `impose_resolution` |
| `resolved_amplitudes` | `ndarray` (pA) | After `impose_resolution` |
| `resolved_flags` | `ndarray` (int8) | After `impose_resolution` |
| `periods` | `Periods` | Open/shut period lists |
| `open_periods` | property | `periods.open_intervals` |
| `shut_periods` | property | `periods.shut_intervals` |
| `n_raw` | property | `len(intervals)` |
| `n_resolved` | property | `len(resolved_intervals)` |

---

## `from_scn`

```python
from_scn(rec: SCNRecord, tres: float = 0.0, *, badopen: float = 0.0)
    -> SingleChannelRecord
```

Convenience wrapper that calls `impose_resolution` then `set_periods` and
packages the results into a `SingleChannelRecord`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rec` | — | `SCNRecord` from `dcio.formats.scn.read` |
| `tres` | `0.0` | Dead time in seconds |
| `badopen` | `0.0` | Max open duration before flagging unusable; 0 = disabled |

---

## Dead time selection

The dead time should be matched to the recording bandwidth.  A common heuristic
is *t*<sub>res</sub> = rise time of the filter × 1.5:

| Filter (–3 dB) | Rise time (10–90%) | Typical *t*<sub>res</sub> |
|---|---|---|
| 10 kHz | ~33 µs | 40–50 µs |
| 3 kHz | ~112 µs | 150 µs |
| 1 kHz | ~333 µs | 500 µs |

---

## Flag bits

| Bit | Value | Meaning |
|-----|-------|---------|
| 0 | 1 | Amplitude dubious |
| 1 | 2 | Amplitude fixed by fitting |
| 2 | 4 | Amplitude constrained |
| 3 | 8 | **Duration unusable** — excluded from analysis |

Intervals with `flag >= 8` are skipped by `impose_resolution` when finding the
first usable interval, and propagate to contaminate any period they fall in.

---

## References

* Colquhoun D & Sigworth FJ (1983) *Fitting and statistical analysis of
  single-channel records.*  In: Sakmann B & Neher E (eds)
  *Single-Channel Recording*, Plenum Press, New York.
* Hawkes AG, Jalali A & Colquhoun D (1992) *Asymptotic distributions of
  apparent open times and shut times in a single channel record allowing
  for the omission of brief events.*  Phil Trans R Soc B 337:383–404.
