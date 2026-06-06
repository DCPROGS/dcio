# CONSAM SSD File Format (.ssd / .dat)

CONSAM is the continuous single-channel analysis program from the
[DCProgs](http://www.ucl.ac.uk/Pharmacology/dcpr95.html) suite (Colquhoun lab,
UCL).  An SSD file stores a **raw continuous** patch-clamp electrophysiology
trace: a time series of ADC samples acquired at a fixed sampling rate.

SSD files are the *raw data* counterpart to SCN files, which store idealised
(fitted) interval lists derived from the same traces.

---

## Overall layout

```
┌──────────────────────────────┐  ← byte 0
│         Header               │
│         (512 bytes)          │
│   zero-padded after fields   │
├──────────────────────────────┤  ← byte 512  (ioff)
│   int16[n_samples]           │  2 × n_samples bytes
│   ADC trace data             │
└──────────────────────────────┘
```

The header is always exactly **512 bytes** (`ioff = 512`).  The data block
starts immediately at byte 512 and continues to the end of the file.

All multi-byte numerics are **little-endian**.

---

## Header fields

Fields are packed sequentially starting at byte 0 with no alignment padding.

| Byte offset | Bytes | Type | Field | Description |
|-------------|-------|------|-------|-------------|
| 0 | 2 | `int16` | `version` | Format version: `1002` (current) or `1001` (old) |
| 2 | 70 | `char[70]` | `title` | Free-text description, padded with `.` |
| 72 | 11 | `char[11]` | `date` | Acquisition date `DD-Mon-YYYY`, e.g. `03-Aug-2018` |
| 83 | 8 | `char[8]` | `time` | Acquisition time `HH-MM-SS`, e.g. `09-41-19` |
| 91 | 2 | `int16` | `idt` | Sampling interval in **microseconds** |
| 93 | 4 | `int32` | `ioff` | Byte offset to data block (always `512`) |
| 97 | 4 | `int32` | `ilen` | Number of samples |
| 101 | 2 | `int16` | `inc` | Sample increment (`1` = every sample stored) |
| 103 | 2 | `int16` | `id1` | Channel identifier 1 (application-specific) |
| 105 | 2 | `int16` | `id2` | Channel identifier 2 (application-specific) |
| 107 | 3 | `char[3]` | `cs` | Channel-type string (e.g. `"H"` = holding current) |
| 110 | 4 | `float32` | `calfac` | ADC → pA calibration factor (see below) |
| 114 | 4 | `float32` | `srate` | Sample rate in Hz (`1e6 / idt`) |
| 118 | 4 | `float32` | `filt` | Low-pass filter cut-off in Hz |
| 122 | 4 | `float32` | `filt1` | Secondary filter frequency in Hz (often `0`) |
| 126 | 4 | `float32` | `calfac1` | Secondary calibration factor (often `0`) |
| 130 | 11 | `char[11]` | `expdate` | Experiment date `DD-Mon-YYYY` |
| 141 | 6 | `char[6]` | `defname` | Default file-name prefix |
| 147 | 24 | `char[24]` | `tapeID` | Tape / file identifier |
| 171 | 4 | `int32` | `ipatch` | Patch number |
| 175 | 4 | `int32` | `npatch` | Number of patches in series |
| 179 | 4 | `float32` | `Emem` | Membrane potential in mV |
| 183 | 4 | `float32` | `temp` | Bath temperature in °C |

**Total field bytes: 187.**  Bytes 187–511 are zero-padded.

---

## Data block

### Samples (`int16`)

Each sample is a 16-bit signed integer representing the ADC output at that
instant.  Samples are stored in acquisition order with no inter-sample gaps.

### Amplitude calibration

The conversion chain from raw ADC to physical units (pA) is:

```
signal_pA = int16_sample × calfac
calfac     = 1.0 / (gain × 6553.6)
```

where:

* **gain** (V/pA) is the transducer gain set on the patch-clamp amplifier
  (e.g. 50 mV/pA → `gain = 0.05`).
* **6553.6** ≈ 2¹⁶ / 10 maps the ±5 V ADC input range to the full `int16`
  range (±32 767).

The inverse — converting physical units to ADC for writing — is:

```
int16_sample = round(signal_pA × gain × 6553.6)
```

### Sampling interval

`idt` is the sampling interval in **microseconds** stored as `int16`.
Permitted range: 1 … 32 767 µs (i.e. sampling rates from ~30 Hz to 1 MHz).
The sample rate in Hz is always `1e6 / idt`.

Typical patch-clamp values:

| `idt` (µs) | Sample rate |
|------------|-------------|
| 10 | 100 kHz |
| 20 | 50 kHz |
| 50 | 20 kHz |
| 100 | 10 kHz |

---

## Version notes

| `version` | Notes |
|-----------|-------|
| `1002` | Current format; produced by CONSAM and `dcio`. |
| `1001` | Old format; field layout identical, minor internal differences. |

`dcio` reads both versions but only writes version `1002`.

---

## `ilen` field ambiguity

In some legacy files `ilen` stores the number of **bytes** in the data block
(i.e. `2 × n_samples`) rather than the number of samples.  `dcio` stores
`n_samples` in `ilen` (consistent with `csv2ssd.py`) and reads the actual
sample count from the file size rather than relying on `ilen`.

---

## Units summary

| Quantity | File storage | `dcio` API |
|----------|-------------|------------|
| Trace samples | `int16` ADC units | `float64` **pA** |
| Sampling interval | `int16` microseconds | `float64` **seconds** (`dt` attribute) |
| Sample rate | `float32` Hz | `float32` **Hz** (header field) |
| Filter cut-off | `float32` Hz | `float32` **Hz** (header field) |
| Membrane potential | `float32` mV | `float32` **mV** (header field) |

---

## References

* Colquhoun D & Sigworth FJ (1983) *Fitting and statistical analysis of
  single-channel records.*  In: Sakmann B & Neher E (eds) *Single-Channel
  Recording*, Plenum Press, New York.
* DCProgs software suite: <http://www.ucl.ac.uk/Pharmacology/dcpr95.html>
