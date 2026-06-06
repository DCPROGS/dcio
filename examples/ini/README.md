# SCAN.INI example file

This file can be loaded with `dcio.formats.scan_ini.read()`.

## `SCAN.INI`

A 512-byte binary settings file saved by the SCAN single-channel analysis
program (DCProgs, UCL Colquhoun lab).

**Recording context**: analysis of GlyR channel data; ADC data file was
`C:\fort90\scandemo\consam2.513`.  The file was originally created in
256-byte format and auto-converted to 512 bytes by SCAN on first load.
The three fields added in the 512-byte format (`scritvar`, `smultmin`,
`stpfac`) contain invalid values inherited from the conversion and are
corrected to defaults by SCAN at runtime.

**Key parameter values** (from the saved session):

| Field     | Value | Meaning                                    |
|-----------|-------|--------------------------------------------|
| `adcfil`  | `C:\fort90\scandemo\consam2.513` | Source ADC file |
| `opendown`| `True`  | Openings are downward deflections        |
| `iscrit`  | `1`   | Threshold = fraction of mean amplitude     |
| `smult`   | ~0.16 | ~16% of mean amplitude                    |
| `cjump`   | `False` | Continuous-sample (not C-jump) data     |
| `novlap`  | `2048`  | Default overlap between sections         |

## Quick start

```python
from dcio.formats.scan_ini import read, ISCRIT_FRACTION_AMP, ISCRIT_MULTIPLE_RMS

ini = read("examples/ini/SCAN.INI")

print(f"ADC file  : {ini.adcfil}")
print(f"Open down : {ini.opendown}")
print(f"C-jump    : {ini.cjump}")

if ini.iscrit == ISCRIT_FRACTION_AMP:
    pct = ini.smult * 100
    print(f"Threshold : {pct:.1f}% of mean amplitude")
else:
    print(f"Threshold : {ini.smult:.1f} × RMS noise")
```
