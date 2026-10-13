# RIS total-field data

All samples are stored directly inside folders whose names specify the array
size and incident angle. Each folder contains sequentially numbered files,
`sample_000001.fld`, `sample_000002.fld`, and so on. Each file contains the input and target fields for one RIS configuration.

```text
training/
    10by10_0deg/
        sample_000001.fld
        sample_000002.fld
        ...
    10by10_10deg/
    ...
    16by16_50deg/
test/
    10by10_30deg/
        sample_000001.fld
        ...
    10by10_45deg/
    ...
    24by24_30deg/
```

## Training inventory

| Array size | Incident angles (degrees) | Samples per angle | Total samples |
| --- | --- | --- | --- |
| 10x10 | 0, 10, 20, 30, 40, 50 | 2,000 | 12,000 |
| 16x16 | 0, 10, 20, 30, 40, 50 | 2,000 | 12,000 |

There are 24,000 published training samples. Each size/angle group contains
2,000 samples. Sequential filenames are assigned in random order.
The training script can select a further random subset using `data.samples_per_group` in [config.json](../config.json).

## Test inventory

| Folder | Samples |
| --- | --- |
| 10by10_0deg | 50 |
| 10by10_30deg | 50 |
| 10by10_45deg | 50 |
| 12by12_25deg | 50 |
| 12by12_30deg | 50 |
| 14by14_10deg | 50 |
| 14by14_25deg | 50 |
| 16by16_30deg | 50 |
| 16by16_45deg | 50 |
| 20by20_25deg | 50 |
| 24by24_30deg | 50 |

There are 550 test samples, with sequential filenames assigned in random order.

`manifest.json` records the counts and sizes of the training and test groups.

## Sample format

Files contain one header line followed by whitespace-separated numeric rows.
Each sample has `size * size` rows and 17 columns, listed with zero-based indices:

| Columns | Values |
| --- | --- |
| 0 | Binary configuration state |
| 1, 2 | One-based x and y grid indices |
| 3, 4 | Real and imaginary incident Ex |
| 5, 6 | Real and imaginary incident Ey |
| 7, 8 | Real and imaginary incident Ez |
| 9, 10 | Real and imaginary total Ex |
| 11, 12 | Real and imaginary total Ey |
| 13, 14 | Real and imaginary total Hx |
| 15, 16 | Real and imaginary total Hy |

The inputs are `[state, Re(Ex_inc), Im(Ex_inc), Re(Ey_inc), Im(Ey_inc)]`.
With the default magnetic scale of 377, targets are `[Re(Ex_tot), Im(Ex_tot), Re(Ey_tot), Im(Ey_tot),
377*Re(Hx_tot), 377*Im(Hx_tot), 377*Re(Hy_tot), 377*Im(Hy_tot)]`.
Channels follow the stored sample row order and are reshaped to `(size, size)`.
Electric and magnetic total fields are read directly from the samples by `fields.py`.

## License and attribution

The dataset and accompanying files in this directory are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE).

Copyright (c) 2026 Yuanzhi Liu and Costas D. Sarris.

When sharing this dataset or an adapted version, credit Yuanzhi Liu and
Costas D. Sarris for **RIS CNN Training and Test Data**, provide a link to
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and indicate any
changes. See the [repository citation section](../../README.md#citation)
for suggested paper citations.
