# Local dataset layout

Datasets are intentionally excluded from Git. Place them under `Dataset/`:

```text
Dataset/
  PNU-NOMV/
    data/
      records/        Eight `*_operating.dat` files
      features/       Optional DOI-release feature shards
  XJTU-SY/
    XJTU-SY_Bearing_Datasets/
  PRONOSTIA/
    ... provider archive/files ...
  NASA-IMS/
    ... provider archives ...
```

The public benchmark inventory script records the exact provider files it finds.
Do not rename archive members or combine duplicate PRONOSTIA full/truncated runs.
The NASA archive provenance conflict documented by the experiment manifest must
remain visible rather than being silently relabeled.

PNU scripts default to `Dataset/PNU-NOMV/data/records`. Alternatively, pass an
explicit `--input` path. The DOI feature shards are supplied for verification,
but full reproduction starts from the immutable record files.
