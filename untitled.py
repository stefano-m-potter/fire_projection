#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Join per-dataset predictor Parquet files into a single
per-(year, month) Parquet dataset.

Inputs (existing):
  /explore/nobackup/people/spotter5/anna_v/v2/parquet_predictors/<DATASET>/<DATASET>_<YEAR>.parquet

Output (this script):
  /explore/nobackup/people/spotter5/anna_v/v2/parquet_predictors_joined/<YEAR>/predictors_<YEAR>_<MM>.parquet

Join keys:
  x, y, year, month

Predictor columns:
  All non-key columns from each dataset, prefixed with "<DATASET>__".
"""

import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================
# CONFIG
# ============================================================

IN_ROOT = Path("/explore/nobackup/people/spotter5/anna_v/v2/parquet_predictors")
OUT_ROOT = Path("/explore/nobackup/people/spotter5/anna_v/v2/parquet_predictors_joined")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DATASET_NAMES = [
    "ALT",
    "ERA5",
    "SMAP_L4",
    "LAI_FPAR",
    "LST",
    "MODIS_AllBands",
    "TerraClimate",
    "HiHydroSoil",
    "MERIT_DEM_TPI",
    # NEW datasets
    "SoilGrids",
    "MOD44B",
    "Permafrost_probability_Obu",
    "CO2_CONT",
]

# ============================================================
# HELPERS
# ============================================================

def load_year_tables(year):
    tables_by_ds = {}
    months_by_ds = {}

    for ds_name in DATASET_NAMES:
        ds_dir = IN_ROOT / ds_name
        if not ds_dir.exists():
            continue

        path = ds_dir / f"{ds_name}_{year}.parquet"
        if not path.exists():
            continue

        print(f"  [LOAD] {path}")
        table = pq.read_table(path)
        tables_by_ds[ds_name] = table

        if "month" in table.column_names:
            months = set(table.column("month").to_pylist())
            months = {m for m in months if m is not None and m >= 1}
        else:
            months = set()
        months_by_ds[ds_name] = months

    return tables_by_ds, months_by_ds


def table_month_to_df(table, month, ds_name):
    import pyarrow.compute as pc

    mask = pc.equal(table["month"], pa.scalar(month, type=table["month"].type))
    filtered = table.filter(mask)

    if filtered.num_rows == 0:
        return None

    df = filtered.to_pandas()
    key_cols = ["x", "y", "year", "month"]
    for k in key_cols:
        if k not in df.columns:
            raise ValueError(f"Expected key column '{k}' not found in dataset {ds_name}")

    df = df.set_index(key_cols)

    df = df.rename(columns={col: f"{ds_name}__{col}" for col in df.columns})

    return df


# ============================================================
# MAIN (smoke test for 2004 only)
# ============================================================

def main():
    target_year = 2004
    print(f"[SMOKE TEST] Running for year {target_year} only.")

    year_out_dir = OUT_ROOT / f"{target_year:04d}"
    year_out_dir.mkdir(parents=True, exist_ok=True)

    tables_by_ds, months_by_ds = load_year_tables(target_year)
    if not tables_by_ds:
        print(f"[WARN] No dataset tables found for year {target_year}, exiting.")
        return

    months_all = sorted({m for ms in months_by_ds.values() for m in ms})
    if not months_all:
        print(f"[WARN] No months found in any dataset for year {target_year}, exiting.")
        return

    print(f"  Months in year {target_year}: {months_all}")

    for month in months_all:
        out_path = year_out_dir / f"predictors_{target_year}_{month:02d}.parquet"

        # ============================================
        # SKIP IF OUTPUT FILE ALREADY EXISTS
        # ============================================
        if out_path.exists():
            print(f"  [SKIP exists] {out_path}")
            continue

        print(f"  [MONTH] {target_year}-{month:02d} → {out_path.name}")

        combined_df = None

        for ds_name, table in tables_by_ds.items():
            if month not in months_by_ds.get(ds_name, set()):
                continue

            df_ds = table_month_to_df(table, month, ds_name)
            if df_ds is None or df_ds.empty:
                continue

            if combined_df is None:
                combined_df = df_ds
            else:
                combined_df = combined_df.join(df_ds, how="outer")

        if combined_df is None or combined_df.empty:
            print(f"    [WARN] No data for {target_year}-{month:02d}, skipping.")
            continue

        combined_df = combined_df.reset_index()

        table_out = pa.Table.from_pandas(combined_df, preserve_index=False)
        pq.write_table(table_out, out_path, compression="snappy", use_dictionary=True)
        print(f"    [WRITE] {out_path}")

    print("\n[DONE] Smoke test for 2004 completed.")


if __name__ == "__main__":
    main()
