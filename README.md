# swot_rc
SWOT Rating Curves 2026

# Dependencies
- H2iVDI (https://github.com/DassHydro/H2iVDI) must be installed to compute filtered cross-section geometry
- arviz
- matplotlib
- pandas
- pymc
- tqdm
- xarray

# Usage — Unified script (`swot_rc.py`)

The unified script `swot_rc.py` combines all three equation types into a single entry point.
Select the equation with the `--equation` argument.

## Equation types

| Type | Equation | Description |
|---|---|---|
| `classic` | Q = α (H − z0)^β | Power-law rating curve |
| `lowfroude` | Q = k (A0 + dA)^(5/3) W^(−2/3) S^(1/2) | Manning-like (requires H2iVDI) |
| `sfd` | Q = α (H − z0)^β S^δ | Stage-fall-discharge |

## Calibration

```
$ python swot_rc.py --equation {classic,lowfroude,sfd} DATA_FILES... -o OUTPUT_DIR [-r ROOT_DIR]
```

Examples:

```
$ python swot_rc.py --equation classic   sos/* -o out_RC_classic
$ python swot_rc.py --equation lowfroude sos/* -o out_RC_lf
$ python swot_rc.py --equation sfd       sos/* -o out_RC_sfd
```

## Scoring

```
$ python swot_rc.py --equation {classic,lowfroude,sfd} DATA_FILES... --score --score_csv_file CALIBRATION_CSV
```

Examples:

```
$ python swot_rc.py --equation classic   sos/* --score --score_csv_file out_RC_classic/swot_rc.csv
$ python swot_rc.py --equation lowfroude sos/* --score --score_csv_file out_RC_lf/swot_rc.csv
$ python swot_rc.py --equation sfd       sos/* --score --score_csv_file out_RC_sfd/swot_rc.csv
```
