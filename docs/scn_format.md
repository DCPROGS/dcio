# SCAN Binary File Format (.scn)

SCAN is the single-channel analysis program from the
[DCProgs](http://www.ucl.ac.uk/Pharmacology/dcpr95.html) suite (Colquhoun lab,
UCL).  An SCN file stores an **idealised** single-channel record produced by
fitting a raw patch-clamp electrophysiology trace.  The record is a sequence of
open and shut intervals with associated amplitudes and quality flags.

---

## File versions

| `iscanver` | Meaning | Header size |
|------------|---------|-------------|
| `-103` | Simulated / old format | 153 bytes |
| `103` | Simulated (newer) | 153 bytes |
| `104` | Experimental recording | variable (~350 bytes) |

Versions `-103` and `103` share the same short header layout; version `104` has
an extended header with recording conditions, display parameters, and filter
settings.

---

## Overall layout

```
┌─────────────────────────────┐
│         Header              │  153 bytes (short) or ~350 bytes (full)
├─────────────────────────────┤
│  float32[nint]  intervals   │  4 × nint bytes  (milliseconds)
│  int16[nint]    amplitudes  │  2 × nint bytes  (ADC intermediate units)
│  int8[nint]     flags       │  1 × nint bytes  (property bits)
└─────────────────────────────┘
```

Total data size = **7 × nint** bytes.

All multi-byte integers are **little-endian** (the SCAN program ran on
DOS/Windows x86 systems).

---

## Short header (versions −103 and 103)

| Offset (bytes) | Type | Field | Description |
|----------------|------|-------|-------------|
| 0 | `int32` | `iscanver` | Version tag (`-103` or `103`) |
| 4 | `int32` | `ioffset` | 1-based byte offset to data block (`154`) |
| 8 | `int32` | `nint` | Number of intervals |
| 12 | `char[70]` | `title` | Free-text description |
| 82 | `char[11]` | `date` | Date string, e.g. `"25-Jan-2012"` |
| 93 | `char[24]` | `tapeID` | Tape / file identifier |
| 117 | `int32` | `ipatch` | Patch number |
| 121 | `float32` | `Emem` | Membrane potential (mV) |
| 125 | `int32` | `unknown1` | (unused) |
| 129 | `float32` | `avamp` | Average amplitude (ADC units) |
| 133 | `float32` | `rms` | RMS baseline noise (ADC units) |
| 137 | `float32` | `ffilt` | Low-pass filter cut-off (Hz; `-1` = unset) |
| 141 | `float32` | `calfac2` | Calibration factor: pA per ADC intermediate unit |
| 145 | `float32` | `treso` | Temporal resolution used in analysis (s) |
| 149 | `float32` | `tresg` | Group temporal resolution (s) |

**Total: 153 bytes** → data starts at byte 153 (0-based) = `ioffset − 1`.

---

## Full header (version 104)

Version 104 extends the short header with the fields below (shown in read
order after the common prefix through `date`).

| Type | Field | Description |
|------|-------|-------------|
| `char[6]` | `defname` | Default file name prefix |
| `char[24]` | `tapeID` | Tape identifier |
| `int32` | `ipatch` | Patch number |
| `int32` | `npatch` | Number of patches |
| `float32` | `Emem` | Membrane potential (mV) |
| `float32` | `temper` | Temperature (°C) |
| `char[30]` | `adcfil` | ADC filter file name |
| `char[35]` | `qfile1` | Associated raw data file |
| `int32` | `cjump` | Logical: data from concentration-jump file |
| `int32` | `nfits` | Number of fits |
| `int32` | `ntmax` | Maximum number of transitions |
| `int32` | `nfmax` | Maximum number of fits |
| `int32` | `nbuf` | Points per disk read (default 131072) |
| `int32` | `novlap` | Overlap points between sections (default 2048) |
| `float32` | `srate` | Sampling rate (Hz) |
| `float32` | `finter` | Sampling interval (µs) |
| `float32` | `tsect` | Section duration (µs) |
| `int32` | `ioff` | First data point offset |
| `int32` | `ndat` | Total number of data points |
| `int32` | `nsec` | Number of sections |
| `int32` | `nrlast` | Points in the last section |
| `float32` | `avtot` | Total average |
| `int32` | `navamp` | Number of amplitude estimates |
| `float32` | `avamp` | Average amplitude (ADC units) |
| `float32` | `rms` | RMS baseline noise (ADC units) |
| `int32` | `nwrit` | Auto-save interval (transitions) |
| `int32` | `nwsav` | Auto-save counter |
| `int32` | `newpar` | Logical: new parameters set |
| `int32` | `opendown` | Logical: openings go downward |
| `int32` | `invert` | Logical: invert trace |
| `int32` | `usepots` | Logical: use potential steps |
| `int32` | `disp` | Logical: display only (no step-response) |
| `float32` | `smult` | Scrit multiplier |
| `int32` | `scrit` | Critical level type (1 = %amp, 2 = ×RMS) |
| `int32` | `vary` | Vary fit parameters |
| `int32` | `ntrig` | Consecutive points beyond Scrit for trigger (default 2) |
| `int32` | `navtest` | Points averaged before curlev is used |
| `float32` | `dgain` | Display gain (default 1.0) |
| `int32` | `iboff` | Baseline display offset (ADC units) |
| `float32` | `expfac` | Expand factor (default 2.0) |
| `float32` | `bdisp` | Baseline screen position (0.25 or 0.75) |
| `int32` | `ibflag` | Baseline flag |
| `int32` | `iautosub` | Auto-avoid sublevels |
| `float32` | `xtrig` | Trigger position on x-axis (default 0.2) |
| `char[2]` | `ndev` | Disk partition (e.g. `"C:"`) |
| `char[11]` | `cdate` | Creation date |
| `char[8]` | `adctime` | ADC time string |
| `int32` | `nsetup` | Setup number |
| `char[20]` | `filtfile` | Filter file name |
| `float32` | `ffilt` | Low-pass filter cut-off (Hz, −3 dB) |
| `int32` | `npfilt` | Points to skip after transition |
| `float32` | `sfac1` | Scale factor 1 (ADC → pixel) |
| `float32` | `sfac2` | Scale factor 2 (ADC → intermediate) |
| `float32` | `sfac3` | Scale factor 3 (intermediate → pixel) |
| `int32` | `nscale` | Scale exponent |
| `float32` | `calfac` | Raw calibration factor (pA / ADC unit) |
| `float32` | `calfac1` | calfac / sfac1 (pixel display units → pA) |
| `float32` | `calfac2` | calfac / sfac2 (intermediate units → pA) |

---

## Data block

### Interval durations (`float32`, milliseconds)

Each element is a 32-bit IEEE 754 float representing the duration of one
interval in **milliseconds**.  The `dcio` reader converts to **seconds**
on load.

### Amplitudes (`int16`, ADC intermediate units)

The ADC intermediate unit is an internal SCAN scaling step:

```
amplitude_pA = iampl × calfac2
```

Shut intervals have `iampl = 0`.  Open intervals carry the mean
single-channel current.  For **simulated** files written with
`calfac2 = 1.0`, the stored integers are directly in pA.

### Flags (`int8`, property bitmask)

Each byte encodes the quality of the corresponding interval:

| Bit | Decimal | Meaning |
|-----|---------|---------|
| 0 | 1 | Amplitude dubious |
| 1 | 2 | Amplitude fixed |
| 2 | 4 | Amplitude of opening constrained |
| 3 | 8 | **Duration unusable** |

A flag value of `0` means the interval is fully usable.  Any interval with
`flags & 8 != 0` should be excluded from kinetic analysis.  For experimental
files (version 104), SCAN marks the final sentinel interval unusable
automatically.

---

## Sentinel interval (version 104 only)

Experimental SCN files (version 104) end with a sentinel shut interval
appended by SCAN to mark the end of the record.  `dcio` detects this by
stripping any trailing open intervals until the last element has amplitude
zero, then marking it with `FLAG_UNUSABLE` (`8`).

Simulated files (versions −103 / 103) do not use a sentinel.

---

## Units summary

| Quantity | File storage | `dcio` API |
|----------|-------------|--------------|
| Interval duration | `float32` milliseconds | `float64` **seconds** |
| Amplitude | `int16` ADC intermediate units | `float64` **picoamperes** |
| Filter cut-off | `float32` Hz | `float32` **Hz** (header field) |
| Membrane potential | `float32` mV | `float32` **mV** (header field) |

---

## References

* Colquhoun D & Sigworth FJ (1983) *Fitting and statistical analysis of
  single-channel records.*  In: Sakmann B & Neher E (eds) *Single-Channel
  Recording*, Plenum Press, New York.
* DCProgs software suite: <http://www.ucl.ac.uk/Pharmacology/dcpr95.html>
