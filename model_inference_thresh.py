#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch predict burned probabilities and classes for ALL neg-fraction focal models
trained with IoU-at-best-threshold learning curves.

Paired with training script:

  OUT_ROOT = "/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/neg_ratio_experiments_globaltest"
  OUT_DIR  = os.path.join(OUT_ROOT, "option4_focal_loss_10x_negative_auc_thresh")

Training outputs:

  Models:
    OUT_DIR/models/focal_model_negfrac{neg_fraction_pct:03d}_{backend}.pkl

  Summary CSV:
    OUT_DIR/option4_focal_globaltest_neg_fraction_sweep_metrics_auc_thresh.csv

Summary CSV columns used here:
  - neg_fraction, neg_fraction_pct, backend
  - threshold (probability)
  - best_iteration

We reconstruct the predictor list from:
  cems_with_fraction_balanced_10x.parquet

For each model (neg fraction) and each YEAR × MONTH, we:
  - Read features from the full partitioned Parquet dataset
  - Predict probabilities and classes
  - Write rasters to:

    /explore/nobackup/people/spotter5/clelland_fire_ml/
        predictions_option4_focal_10x_negative_auc_thresh_negfrac{pct:03d}_mcd/
            proba/cems_pred_proba_{YYYY}_{MM}.tif
            class/cems_pred_class_{YYYY}_{MM}_thr{thr}.tif
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow as pa

import rasterio as rio
from rasterio.warp import transform as rio_transform
from rasterio.transform import rowcol

import lightgbm as lgb
import joblib

# ============================================================
# CONFIG
# ============================================================

# Years & months to process
YEARS  = list(range(2001, 2024))   # change as needed
MONTHS = list(range(1, 13))        # 1..12

# Where your ORIGINAL template monthly TIFFs live (for georeference/shape)
IN_TIF_DIR = "/explore/nobackup/people/spotter5/clelland_fire_ml/training_e5l_cems_mcd_with_fraction"

# The Parquet dataset built from those TIFFs (partitioned by year=/month=)
PARQUET_DATASET_DIR = "/explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_with_fraction_dataset_mcd"

# The 10x balanced training parquet used for focal models
TRAIN_PARQUET_10X = "/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/cems_with_fraction_balanced_10x.parquet"

# Focal training output directory (models + summary CSV) — MATCHES NEW TRAINING SCRIPT
FOCAL_OUT_ROOT = "/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/neg_ratio_experiments_globaltest"
FOCAL_OUT_DIR  = os.path.join(FOCAL_OUT_ROOT, "option4_focal_loss_10x_negative_auc_thresh")

MODELS_DIR = os.path.join(FOCAL_OUT_DIR, "models")
SUMMARY_CSV = os.path.join(
    FOCAL_OUT_DIR, "option4_focal_globaltest_neg_fraction_sweep_metrics_auc_thresh.csv"
)

# Root where prediction rasters for all neg-fraction models will be stored
PRED_ROOT = "/explore/nobackup/people/spotter5/clelland_fire_ml"

# Parquet lon/lat handling
PARQUET_COORDS_ARE_EPSG4326 = True

# Batch size for reading Parquet
PARQUET_BATCH_ROWS = 1_000_000

# ============================================================
# Helpers
# ============================================================

name_re = re.compile(r"cems_e5l_mcd_(\d{4})_(\d{1,2})", re.IGNORECASE)

def parse_year_month(fname: str):
    m = name_re.search(fname)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def find_template_tif(year: int, month: int) -> Path:
    """Find the template GeoTIFF for the given year/month."""
    patterns = [
        f"cems_e5l_mcd_{year}_{month}_with_fraction.tif",
        f"cems_e5l_mcd_{year}_{month}.tif",
        f"cems_e5l_mcd_{year}_{month:02d}_with_fraction.tif",
        f"cems_e5l_mcd_{year}_{month:02d}.tif",
    ]
    for p in patterns:
        cand = Path(IN_TIF_DIR) / p
        if cand.exists():
            return cand

    # fallback: search by regex
    for tif in Path(IN_TIF_DIR).glob(f"cems_e5l_mcd_{year}_*.tif"):
        y, m = parse_year_month(tif.name)
        if y == year and m == month:
            return tif
    return None

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def ensure_predictor_frame(df: pd.DataFrame, predictors: list):
    """
    Ensure df has all predictor columns with correct types/order.
    - Missing predictors -> NaN
    - 'b1' treated as categorical (like in training)
    - Other columns coerced to numeric if needed
    """
    for c in predictors:
        if c not in df.columns:
            df[c] = np.nan
    X = df[predictors].copy()

    # Handle land cover as categorical if present
    if "b1" in X.columns:
        X["b1"] = X["b1"].astype("category")

    # Coerce other non-numeric to numeric
    for c in X.columns:
        if c == "b1" and pd.api.types.is_categorical_dtype(X[c]):
            continue
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    return X

def write_geotiff(path, array2d, template_ds, nodata=np.nan, dtype="float32"):
    profile = template_ds.profile.copy()
    profile.update(
        count=1,
        dtype=dtype,
        nodata=nodata if (isinstance(nodata, (int, float)) and not np.isnan(nodata)) else None,
        compress="deflate",
        predictor=2,
        tiled=True
    )
    with rio.open(path, "w", **profile) as dst:
        dst.write(array2d.astype(dtype), 1)

# ============================================================
# 1) Recover predictor list from the 10x training parquet
# ============================================================

print(f"Reading training parquet (10x balanced) from:\n  {TRAIN_PARQUET_10X}")
df_train_meta = pd.read_parquet(TRAIN_PARQUET_10X)

if "fraction" not in df_train_meta.columns:
    raise ValueError("Expected column 'fraction' in TRAIN_PARQUET_10X.")

drop_cols = {"fraction", "burned", "bin", "year", "month", "latitude", "longitude"}
predictors = [c for c in df_train_meta.columns if c not in drop_cols]

print("\nRecovered predictor columns from training parquet:")
print(f"  #predictors: {len(predictors)}")
print("  First few predictors:", predictors[:10])

# ============================================================
# 2) Load summary CSV with thresholds and iterations
# ============================================================

if not os.path.exists(SUMMARY_CSV):
    raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_CSV}")

summary_df = pd.read_csv(SUMMARY_CSV)
if summary_df.empty:
    raise ValueError(f"Summary CSV is empty: {SUMMARY_CSV}")

print(f"\nLoaded summary CSV with {len(summary_df)} rows:")
print(summary_df[["neg_fraction", "neg_fraction_pct", "backend", "threshold", "best_iteration"]])

# ============================================================
# 3) Prepare Parquet dataset handle once
# ============================================================

dataset = ds.dataset(PARQUET_DATASET_DIR, format="parquet", partitioning="hive")
print(f"\nUsing prediction Parquet dataset:\n  {PARQUET_DATASET_DIR}")

# ============================================================
# 4) Loop over all neg-fraction models
# ============================================================

for idx, row in summary_df.iterrows():
    neg_fraction      = float(row["neg_fraction"])
    neg_fraction_pct  = int(row["neg_fraction_pct"])
    backend           = str(row["backend"])
    THRESH_PROBA      = float(row["threshold"])
    best_iter         = int(row.get("best_iteration", -1))

    print("\n" + "=" * 80)
    print(f"MODEL {idx+1}/{len(summary_df)}: neg_fraction={neg_fraction:.1f} "
          f"(pct={neg_fraction_pct}%), backend={backend}")
    print(f"  threshold (proba): {THRESH_PROBA:.3f}")
    print(f"  best_iteration   : {best_iter}")

    model_path = os.path.join(
        MODELS_DIR,
        f"focal_model_negfrac{neg_fraction_pct:03d}_{backend}.pkl"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"  Loading model from: {model_path}")
    booster = joblib.load(model_path)

    # Set up output dirs for this neg fraction
    OUT_PRED_DIR = os.path.join(
        PRED_ROOT,
        f"predictions_option4_focal_10x_negative_auc_thresh_negfrac{neg_fraction_pct:03d}_mcd"
    )
    os.makedirs(os.path.join(OUT_PRED_DIR, "proba"), exist_ok=True)
    os.makedirs(os.path.join(OUT_PRED_DIR, "class"), exist_ok=True)

    print(f"  Output dir: {OUT_PRED_DIR}")

    # ========================================================
    # Predict for requested YEARS × MONTHS for this model
    # ========================================================
    for YEAR in YEARS:
        for MONTH in MONTHS:
            template_path = find_template_tif(YEAR, MONTH)
            if template_path is None:
                print(f"[SKIP] No template TIF in {IN_TIF_DIR} for {YEAR}-{MONTH:02d}")
                continue

            print(f"\n[negfrac{neg_fraction_pct:03d} {backend}] "
                  f"=== Predicting {YEAR}-{MONTH:02d} ===")

            with rio.open(template_path) as tpl:
                height, width = tpl.height, tpl.width
                transform = tpl.transform
                dst_crs = tpl.crs

                proba_arr = np.full((height, width), np.nan, dtype="float32")
                class_arr = np.full((height, width), 255, dtype="uint8")  # 255 nodata

                # Filter Parquet dataset for this year/month
                filt = (ds.field("year") == YEAR) & (ds.field("month") == MONTH)
                needed_cols = list(set(predictors + ["longitude", "latitude", "year", "month"]))

                scanner = ds.Scanner.from_dataset(
                    dataset,
                    columns=needed_cols,
                    filter=filt,
                    batch_size=PARQUET_BATCH_ROWS
                )

                total_rows = 0
                written_rows = 0

                for batch in scanner.to_batches():
                    if batch.num_rows == 0:
                        continue

                    tbl = pa.Table.from_batches([batch])
                    df = tbl.to_pandas()
                    total_rows += len(df)
                    if len(df) == 0:
                        continue

                    if "longitude" not in df.columns or "latitude" not in df.columns:
                        raise ValueError(
                            "Parquet dataset must include 'longitude' and 'latitude' columns."
                        )

                    lons = df["longitude"].to_numpy(dtype="float64", copy=False)
                    lats = df["latitude"].to_numpy(dtype="float64", copy=False)

                    # Build predictors frame with correct order/types
                    X = ensure_predictor_frame(df, predictors)

                    # Predict probabilities
                    if backend == "lightgbm":
                        # booster is a lightgbm.Booster
                        if isinstance(best_iter, int) and best_iter > 0:
                            raw_pred = booster.predict(X, num_iteration=best_iter)
                        else:
                            raw_pred = booster.predict(X)
                        proba = sigmoid(raw_pred)
                    elif backend == "xgboost":
                        import xgboost as xgb
                        dmat = xgb.DMatrix(X, enable_categorical=True)
                        if isinstance(best_iter, int) and best_iter > 0:
                            raw_pred = booster.predict(
                                dmat,
                                iteration_range=(0, best_iter + 1)
                            )
                        else:
                            raw_pred = booster.predict(dmat)
                        proba = sigmoid(raw_pred)
                    else:
                        raise ValueError(f"Unknown backend in summary: {backend}")

                    # Map coordinates to row/col in template CRS
                    if PARQUET_COORDS_ARE_EPSG4326 and (dst_crs is not None) and (
                        dst_crs.to_string().upper() not in ("EPSG:4326", "OGC:CRS84")
                    ):
                        xs, ys = rio_transform("EPSG:4326", dst_crs, lons, lats)
                    else:
                        xs, ys = lons, lats

                    # Convert to row/col (chunked to avoid overhead)
                    step = 500_000
                    npts = len(xs)
                    rows_all = np.empty(npts, dtype=np.int64)
                    cols_all = np.empty(npts, dtype=np.int64)
                    for s in range(0, npts, step):
                        e = s + step
                        rr, cc = rowcol(transform, xs[s:e], ys[s:e], op=round)
                        rows_all[s:e] = rr
                        cols_all[s:e] = cc

                    # Keep only points inside raster bounds
                    mask_in = (
                        (rows_all >= 0) & (rows_all < height) &
                        (cols_all >= 0) & (cols_all < width)
                    )
                    if not np.any(mask_in):
                        continue

                    rr = rows_all[mask_in]
                    cc = cols_all[mask_in]
                    pp = proba[mask_in]

                    # Write to arrays
                    proba_arr[rr, cc] = pp.astype("float32")
                    class_arr[rr, cc] = (pp >= THRESH_PROBA).astype("uint8")
                    written_rows += int(mask_in.sum())

                # Save rasters
                out_proba = Path(OUT_PRED_DIR) / "proba" / f"cems_pred_proba_{YEAR}_{MONTH:02d}.tif"
                out_class = Path(OUT_PRED_DIR) / "class" / f"cems_pred_class_{YEAR}_{MONTH:02d}_thr{THRESH_PROBA:.3f}.tif"

                write_geotiff(str(out_proba), proba_arr, tpl, nodata=np.nan, dtype="float32")
                write_geotiff(str(out_class), class_arr, tpl, nodata=255, dtype="uint8")

                print(f"[negfrac{neg_fraction_pct:03d} {backend} {YEAR}-{MONTH:02d}] "
                      f"wrote {written_rows:,}/{total_rows:,} rows ->")
                print(f"  - {out_proba}")
                print(f"  - {out_class}")

print("\n✅ Done. All requested predictions written for all IoU-threshold-based focal models.")
