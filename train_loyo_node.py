#!/usr/bin/env python3
import os
import sys
import gc
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import pyarrow.dataset as ds
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG & ARGPARSE
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True, help="The LOYO year to test")
args = parser.parse_args()

TEST_YEAR = args.year
RANDOM_STATE = 42

# --- PATHS ---
DATASET_DIR = Path("/explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_new_fwi_with_fraction_dataset_pred_mask_new_fwi")
OUT_DIR = Path("/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/xgb_loyo_regularized_new_fwi")
MODELS_DIR = OUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- FEATURES ---
FEATURES = [
    "DEM", "slope", "aspect", "b1", "relative_humidity",
    "total_precipitation_sum", "temperature_2m", "temperature_2m_min",
    "temperature_2m_max", "build_up_index", "drought_code",
    "duff_moisture_code", "fine_fuel_moisture_code",
    "fire_weather_index", "initial_fire_spread_index",
]
FRACTION_COL = "fraction"
LABEL_COL = "burned"

# ============================================================
# HELPERS
# ============================================================
def find_area_match_threshold(y_true, y_probs):
    n_burned = np.sum(y_true)
    n_total = len(y_true)
    if n_burned == 0: return 0.99 
    target_percentile = 100.0 * (1.0 - (n_burned / n_total))
    return float(np.percentile(y_probs, target_percentile))

def prepare_df_cleaned(df: pd.DataFrame):
    df = df.copy()
    df[FRACTION_COL] = pd.to_numeric(df[FRACTION_COL], errors="coerce").astype("float32")
    df = df[df[FRACTION_COL].notna() & (df[FRACTION_COL] != 0.5)].copy()
    df[LABEL_COL] = (df[FRACTION_COL] > 0.5).astype("uint8")
    df["b1"] = pd.to_numeric(df["b1"], errors="coerce").round().astype("Int64").astype("category")
    df = df.dropna(subset=FEATURES + [LABEL_COL, "year"])
    return df

def main():
    print(f"--- Training LOYO Year: {TEST_YEAR} ---")
    
    # Load data
    dset = ds.dataset(str(DATASET_DIR), format="parquet", partitioning="hive")
    table = dset.to_table(columns=FEATURES + [FRACTION_COL, "year"])
    df_all = prepare_df_cleaned(table.to_pandas())

    # Split: Train on everything EXCEPT TEST_YEAR
    df_test = df_all[df_all["year"] == TEST_YEAR].copy()
    df_tv = df_all[df_all["year"] != TEST_YEAR]

    X_tv = df_tv[FEATURES]
    y_tv = df_tv[LABEL_COL].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.20, random_state=RANDOM_STATE, stratify=y_tv
    )

    # Weights for imbalance
    scale_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

    # DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval   = xgb.DMatrix(X_val,   label=y_val,   enable_categorical=True)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda", # Uses the GPU assigned by Slurm
        "learning_rate": 0.05,
        "scale_pos_weight": scale_weight,
        "max_depth": 4,
        "min_child_weight": 100,
        "gamma": 5.0,
        "subsample": 0.5,
        "colsample_bytree": 0.5,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dval, "val")],
        early_stopping_rounds=200,
        verbose_eval=100
    )

    # Threshold calculation
    val_probs = booster.predict(dval)
    val_thr = find_area_match_threshold(y_val, val_probs)
    
    # Save results
    save_path = MODELS_DIR / f"xgb_loyo_{TEST_YEAR}.json"
    booster.save_model(str(save_path))
    
    # Save threshold to a small sidecar file
    meta = {"year": TEST_YEAR, "threshold": val_thr}
    with open(MODELS_DIR / f"xgb_loyo_{TEST_YEAR}_meta.json", "w") as f:
        json.dump(meta, f)

    print(f"✅ Finished {TEST_YEAR}. Threshold: {val_thr:.4f}")

if __name__ == "__main__":
    main()