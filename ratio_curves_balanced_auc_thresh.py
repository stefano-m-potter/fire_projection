#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BalancedBagging + XGBoost — neg:pos sweep (100, 90, ..., 10) using:

- Train/Val dataset with ~100:1 neg:pos:
    /explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_trainval_100x

- 10% fixed global test set with true ~4169:1 neg:pos:
    /explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_test_true_10pct

- Tuned BalancedBagging + XGB params loaded from:
    .../option4_balanced_bagging_xgb_aucpr/tuned_balanced_bagging_xgb_params.json

For each desired neg:pos ratio R in [100, 90, ..., 10]:

  - Use ALL positives from Train/Val 100x pool.
  - Sample negatives to target ≈ R:1 neg:pos (cannot exceed pool's actual ratio).
  - Stratified Train vs Val split (same effective 20% overall Val as before).
  - Train BalancedBaggingClassifier(XGBClassifier) with tuned params.
  - On Val:
        * Predict probabilities
        * Find best F1 threshold
        * Compute IoU on Train + Val at that threshold
  - Using that threshold:
        * Compute final test metrics on the true-ratio test set
        * Plot ROC and PR curves
        * Save artifacts + summary CSV.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyarrow.dataset as ds

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    jaccard_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)

from imblearn.ensemble import BalancedBaggingClassifier
from xgboost import XGBClassifier
import joblib

# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

# These are the datasets you already created
TRAINVAL_DIR = "/explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_trainval_100x"
TEST_DIR     = "/explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_test_true_10pct"

# Where tuned BalancedBagging+XGB params were saved from the tuning script
OUT_ROOT_OLD  = "/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/neg_ratio_experiments_globaltest"
PARAMS_DIR    = os.path.join(OUT_ROOT_OLD, "option4_balanced_bagging_xgb_aucpr")
BEST_PARAMS_JSON = os.path.join(PARAMS_DIR, "tuned_balanced_bagging_xgb_params.json")

# New output directory for this experiment
OUT_ROOT = OUT_ROOT_OLD
OUT_DIR  = os.path.join(OUT_ROOT, "balancedbagging_trueTest_train100_negpos_sweep")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS_DIR = os.path.join(OUT_DIR, "models")
FIGS_DIR   = os.path.join(OUT_DIR, "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

TOP_N_IMPORT = 30

# Ratios we want to explore (neg:pos in Train/Val subset)
NEG_POS_RATIOS = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

# Same semantics as before: 10% test overall, 20% val overall => ~22.22% of TrainVal
TEST_SIZE_GLOBAL = 0.10   # informational (already applied upstream)
VAL_SIZE_OVERALL = 0.20
VAL_SIZE_INNER   = VAL_SIZE_OVERALL / (1.0 - TEST_SIZE_GLOBAL)

# ============================================================
# Helpers
# ============================================================

def prepare_trainval_and_predictors(df: pd.DataFrame):
    """
    From the Train/Val 100x dataframe:

      - Ensure `burned` exists (0/1 label; recompute from fraction if needed).
      - Define predictors (drop fraction/burned/coords/etc).
      - Handle `b1` categorical.
      - Coerce non-numeric predictors to numeric.
      - Drop rows with NaNs in predictors (and NaN b1 if categorical).
    """
    df = df.copy()

    if "fraction" in df.columns:
        df["fraction"] = df["fraction"].astype("float32").clip(0, 1)

    if "burned" not in df.columns:
        if "fraction" not in df.columns:
            raise ValueError("Need either 'burned' or 'fraction' in Train/Val dataset.")
        df["burned"] = (df["fraction"] > 0.5).astype("uint8")

    df = df.replace([np.inf, -np.inf], np.nan)

    drop_cols = {"fraction", "burned", "bin", "year", "month", "latitude", "longitude"}
    predictors = [c for c in df.columns if c not in drop_cols]

    X = df[predictors].copy()
    y = df["burned"].astype("uint8")

    # Treat land cover as categorical if present
    if "b1" in X.columns and not pd.api.types.is_categorical_dtype(X["b1"]):
        X["b1"] = X["b1"].astype("category")
        print("\nTreating 'b1' as pandas 'category' in Train/Val.")

    # Coerce non-numeric predictors (except categorical b1) to numeric
    coerced = 0
    for c in X.columns:
        if c == "b1" and pd.api.types.is_categorical_dtype(X[c]):
            continue
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
            coerced += 1

    if coerced:
        num_cols = [
            c for c in X.columns
            if not (c == "b1" and pd.api.types.is_categorical_dtype(X["b1"]))
        ]
        mask = X[num_cols].notna().all(axis=1)
        if "b1" in X.columns and pd.api.types.is_categorical_dtype(X["b1"]):
            mask &= X["b1"].notna()

        before = len(X)
        X = X.loc[mask].copy()
        y = y.loc[mask].copy()
        print(f"Dropped {before - len(X):,} Train/Val rows with NaNs after coercion.")

    return X, y, predictors


def prepare_test(df: pd.DataFrame, predictors):
    """
    Prepare X_test, y_test using the same predictor set as Train/Val.
    """
    df = df.copy()

    if "fraction" in df.columns:
        df["fraction"] = df["fraction"].astype("float32").clip(0, 1)

    if "burned" not in df.columns:
        if "fraction" not in df.columns:
            raise ValueError("Need either 'burned' or 'fraction' in Test dataset.")
        df["burned"] = (df["fraction"] > 0.5).astype("uint8")

    df = df.replace([np.inf, -np.inf], np.nan)

    missing_preds = [c for c in predictors if c not in df.columns]
    if missing_preds:
        raise ValueError(f"Test dataset is missing predictor columns: {missing_preds}")

    X = df[predictors].copy()
    y = df["burned"].astype("uint8")

    # Handle categorical b1 identical to Train/Val
    if "b1" in X.columns and not pd.api.types.is_categorical_dtype(X["b1"]):
        X["b1"] = X["b1"].astype("category")
        print("\nTreating 'b1' as pandas 'category' in Test.")

    # Coerce any non-numeric predictor (except categorical b1) to numeric
    coerced = 0
    for c in X.columns:
        if c == "b1" and pd.api.types.is_categorical_dtype(X[c]):
            continue
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
            coerced += 1

    if coerced:
        num_cols = [
            c for c in X.columns
            if not (c == "b1" and pd.api.types.is_categorical_dtype(X["b1"]))
        ]
        mask = X[num_cols].notna().all(axis=1)
        if "b1" in X.columns and pd.api.types.is_categorical_dtype(X["b1"]):
            mask &= X["b1"].notna()

        before = len(X)
        X = X.loc[mask].copy()
        y = y.loc[mask].copy()
        print(f"Dropped {before - len(X):,} Test rows with NaNs after coercion.")

    return X, y


# ============================================================
# LOAD TUNED BALANCED-BAGGING + XGB PARAMS
# ============================================================

print(f"Loading tuned BalancedBagging+XGB params from:\n  {BEST_PARAMS_JSON}")
with open(BEST_PARAMS_JSON, "r") as f:
    tuned = json.load(f)

bb_n_estimators  = tuned["balanced_bagging_n_estimators"]
bb_max_samples   = tuned["balanced_bagging_max_samples"]
xgb_params       = tuned["xgb_params"]
xgb_num_boost    = tuned.get("xgb_num_boost_rounds", xgb_params.get("n_estimators", 600))

print("\nUsing BalancedBagging+XGB tuned params:")
print(f"  balanced_bagging_n_estimators = {bb_n_estimators}")
print(f"  balanced_bagging_max_samples  = {bb_max_samples}")
print(f"  xgb_num_boost_rounds          = {xgb_num_boost}")
print("  xgb_params:")
for k, v in xgb_params.items():
    print(f"    {k}: {v}")

# Make sure n_estimators is present in xgb_params (but DO NOT override it in the constructor)
if "n_estimators" not in xgb_params:
    xgb_params["n_estimators"] = xgb_num_boost

# ============================================================
# LOAD TRAINVAL 100x + TEST TRUE-RATIO
# ============================================================

print(f"\nLoading Train/Val 100x dataset from:\n  {TRAINVAL_DIR}")
tv_dataset = ds.dataset(TRAINVAL_DIR, format="parquet")
tv_table   = tv_dataset.to_table()
df_tv_full = tv_table.to_pandas()
print(f"Train/Val 100x raw size: {len(df_tv_full):,} rows")

print(f"\nLoading true-ratio 10% Test dataset from:\n  {TEST_DIR}")
test_dataset = ds.dataset(TEST_DIR, format="parquet")
test_table   = test_dataset.to_table()
df_test_full = test_table.to_pandas()
print(f"Test raw size: {len(df_test_full):,} rows")

# Prepare TrainVal + predictors
X_tv_full, y_tv_full, predictors = prepare_trainval_and_predictors(df_tv_full)
print(f"\nTrain/Val 100x after cleaning: {len(X_tv_full):,} rows")
print(f"Number of predictors: {len(predictors)}")
print("Train/Val class counts:")
print(pd.Series(y_tv_full).value_counts())
print(pd.Series(y_tv_full).value_counts(normalize=True).mul(100))

# Prepare Test
X_test, y_test = prepare_test(df_test_full, predictors)
print(f"\nTest after cleaning: {len(X_test):,} rows")
print("Test class counts:")
print(pd.Series(y_test).value_counts())
print(pd.Series(y_test).value_counts(normalize=True).mul(100))

test_pos = int((y_test == 1).sum())
test_neg = int((y_test == 0).sum())
print(f"\nTest positives (1): {test_pos:,}")
print(f"Test negatives (0): {test_neg:,}")

# Split TrainVal into pos / neg pool (starting from ~100:1)
tv_data = X_tv_full.copy()
tv_data["burned"] = y_tv_full

pos_pool = tv_data[tv_data["burned"] == 1]
neg_pool = tv_data[tv_data["burned"] == 0]

n_pos_pool = len(pos_pool)
n_neg_pool = len(neg_pool)
actual_ratio_pool = n_neg_pool / max(n_pos_pool, 1)

print("\nTrain/Val 100x pool (starting point) class counts:")
print(tv_data["burned"].value_counts())
print(tv_data["burned"].value_counts(normalize=True).mul(100))
print(f"\nPositives in pool: {n_pos_pool:,}")
print(f"Negatives in pool: {n_neg_pool:,}")
print(f"Actual neg:pos ratio in TrainVal pool ≈ {actual_ratio_pool:.2f}:1")

summary_rows = []

# ============================================================
# NEG:POS SWEEP LOOP (100, 90, ..., 10)
# ============================================================

for i, target_ratio in enumerate(NEG_POS_RATIOS, start=1):
    print("\n" + "=" * 80)
    print(f"=== Sweep {i}/{len(NEG_POS_RATIOS)} — target neg:pos ≈ {target_ratio}:1 ===")

    # Determine how many negatives we can actually use
    if target_ratio >= actual_ratio_pool:
        # Can't upsample negatives, so just use all we have
        n_neg_target = n_neg_pool
        eff_ratio = actual_ratio_pool
        print(f"Target ratio {target_ratio}:1 exceeds pool ratio {actual_ratio_pool:.2f}:1; "
              f"using all negatives (eff_ratio ≈ {eff_ratio:.2f}:1).")
        neg_subset = neg_pool
    else:
        n_neg_target = int(round(target_ratio * n_pos_pool))
        n_neg_target = min(n_neg_target, n_neg_pool)
        eff_ratio = n_neg_target / max(n_pos_pool, 1)
        print(f"Sampling {n_neg_target:,} negatives for target ratio {target_ratio}:1 "
              f"(eff_ratio ≈ {eff_ratio:.2f}:1).")
        neg_subset = neg_pool.sample(
            n=n_neg_target,
            random_state=RANDOM_STATE + i
        )

    # Combine all positives + sampled negatives
    tv_subset = pd.concat([pos_pool, neg_subset], axis=0)
    tv_subset = tv_subset.sample(frac=1.0, random_state=RANDOM_STATE + 100 + i).reset_index(drop=True)

    print("TrainVal subset class counts (for this neg:pos target):")
    print(tv_subset["burned"].value_counts())
    print(tv_subset["burned"].value_counts(normalize=True).mul(100))

    if tv_subset["burned"].nunique() < 2:
        print(f"[SKIP] target_ratio={target_ratio}: only one class present.")
        continue

    # Train vs Val split within this TrainVal subset
    train_sub, val_sub = train_test_split(
        tv_subset,
        test_size=VAL_SIZE_INNER,
        random_state=RANDOM_STATE,
        stratify=tv_subset["burned"],
    )

    print("\nTrain/Val subset split sizes:")
    print(f"  Train: {len(train_sub):,}")
    print(f"  Val  : {len(val_sub):,}")
    print("Train subset class counts:")
    print(train_sub["burned"].value_counts())
    print("Val subset class counts:")
    print(val_sub["burned"].value_counts())

    X_train = train_sub[predictors].copy()
    y_train = train_sub["burned"].astype("uint8")
    X_val   = val_sub[predictors].copy()
    y_val   = val_sub["burned"].astype("uint8")

    n_pos_train = int((y_train == 1).sum())
    n_neg_train = int((y_train == 0).sum())
    n_pos_val   = int((y_val   == 1).sum())
    n_neg_val   = int((y_val   == 0).sum())

    print(f"\nTrain subset positives: {n_pos_train:,}, negatives: {n_neg_train:,}")
    print(f"Val subset positives  : {n_pos_val:,}, negatives: {n_neg_val:,}")

    # ----------------- BUILD AND TRAIN BALANCED-BAGGING MODEL -----------------
    base_estimator = XGBClassifier(**xgb_params)

    clf = BalancedBaggingClassifier(
        estimator=base_estimator,
        n_estimators=bb_n_estimators,
        max_samples=bb_max_samples,
        sampling_strategy="auto",  # rebalance within each bootstrap
        replacement=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining BalancedBagging + XGB...")
    clf.fit(X_train, y_train)

    # ----------------- FIND BEST THRESHOLD ON VAL (MAX F1) -----------------
    print("\nFinding best threshold on Val (max F1)...")
    val_proba = clf.predict_proba(X_val)[:, 1]
    prec_val, rec_val, thr_val = precision_recall_curve(y_val, val_proba)
    prec_ = prec_val[:-1]
    rec_  = rec_val[:-1]
    f1_vals = 2 * prec_ * rec_ / (prec_ + rec_ + 1e-12)
    best_idx = int(np.argmax(f1_vals))
    best_thr = float(thr_val[best_idx])
    best_val_f1 = float(f1_vals[best_idx])

    # IoU on VAL at best threshold
    y_val_hat = (val_proba >= best_thr).astype("uint8")
    best_val_iou = jaccard_score(y_val, y_val_hat, average="binary", zero_division=0)

    # IoU on TRAIN at the same threshold
    train_proba = clf.predict_proba(X_train)[:, 1]
    y_train_hat = (train_proba >= best_thr).astype("uint8")
    train_iou = jaccard_score(y_train, y_train_hat, average="binary", zero_division=0)

    print(f"Best threshold (Val max F1): {best_thr:.3f}")
    print(f"Val F1 at best_thr        : {best_val_f1:.4f}")
    print(f"Val IoU at best_thr       : {best_val_iou:.4f}")
    print(f"Train IoU at best_thr     : {train_iou:.4f}")

    # ----------------- PLOT SIMPLE TRAIN vs VAL IoU BAR PLOT -----------------
    plt.figure(figsize=(6, 4))
    plt.bar(["Train", "Val"], [train_iou, best_val_iou])
    plt.ylabel("IoU (Jaccard)")
    plt.title(
        f"BalancedBagging+XGB — IoU at best Val F1 threshold\n"
        f"Neg:pos target {target_ratio}:1 (eff ≈ {eff_ratio:.2f}:1)"
    )
    plt.tight_layout()
    iou_fig_out = os.path.join(
        FIGS_DIR,
        f"iou_bar_balancedbagging_negpos{int(target_ratio):03d}.png"
    )
    plt.savefig(iou_fig_out, dpi=150)
    plt.close()
    print(f"Saved IoU bar plot: {iou_fig_out}")

    # ----------------- FINAL TEST METRICS -----------------
    print("\nEvaluating on fixed true-ratio Test set...")

    y_test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_hat   = (y_test_proba >= best_thr).astype("uint8")

    test_iou  = jaccard_score(y_test, y_test_hat, average="binary", zero_division=0)
    test_prec = precision_score(y_test, y_test_hat, zero_division=0)
    test_rec  = recall_score(y_test, y_test_hat, zero_division=0)
    test_f1   = f1_score(y_test, y_test_hat, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    test_auc_roc = roc_auc_score(y_test, y_test_proba)

    prec_curve_test, rec_curve_test, _ = precision_recall_curve(y_test, y_test_proba)
    test_auc_pr = average_precision_score(y_test, y_test_proba)

    print("\n==== FINAL TEST METRICS (BalancedBagging+XGB, fixed true-ratio test) ====")
    print(f"Target neg:pos (TrainVal) : {target_ratio}:1")
    print(f"Eff neg:pos in TrainVal   : {eff_ratio:.2f}:1")
    print(f"Threshold (Val best F1)   : {best_thr:.3f}")
    print(f"IoU (Jaccard)             : {test_iou:.4f}")
    print(f"Precision                 : {test_prec:.4f}")
    print(f"Recall                    : {test_rec:.4f}")
    print(f"F1 Score                  : {test_f1:.4f}")
    print(f"ROC AUC                   : {test_auc_roc:.4f}")
    print(f"PR AUC (Avg Precision)    : {test_auc_pr:.4f}")

    # ---- ROC curve ----
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {test_auc_roc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"BalancedBagging+XGB — ROC (neg:pos target {target_ratio}:1)\n"
        f"Eff neg:pos ≈ {eff_ratio:.2f}:1 in TrainVal subset"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    roc_fig_out = os.path.join(
        FIGS_DIR,
        f"roc_curve_balancedbagging_negpos{int(target_ratio):03d}.png"
    )
    plt.savefig(roc_fig_out, dpi=150)
    plt.close()
    print(f"Saved ROC curve: {roc_fig_out}")

    # ---- PR curve ----
    plt.figure(figsize=(6, 5))
    plt.plot(rec_curve_test, prec_curve_test, label=f"PR (AUC = {test_auc_pr:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"BalancedBagging+XGB — PR Curve (neg:pos target {target_ratio}:1)\n"
        f"Eff neg:pos ≈ {eff_ratio:.2f}:1 in TrainVal subset"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pr_fig_out = os.path.join(
        FIGS_DIR,
        f"pr_curve_balancedbagging_negpos{int(target_ratio):03d}.png"
    )
    plt.savefig(pr_fig_out, dpi=150)
    plt.close()
    print(f"Saved PR curve: {pr_fig_out}")

    # ----------------- FEATURE IMPORTANCE -----------------
    # BalancedBagging should expose feature_importances_ if base estimators do.
    if hasattr(clf, "feature_importances_"):
        gain_imp = np.array(clf.feature_importances_, dtype=float)
    else:
        # Fallback: average over base estimators
        all_imp = []
        for est in clf.estimators_:
            if hasattr(est, "feature_importances_"):
                all_imp.append(est.feature_importances_)
        if all_imp:
            gain_imp = np.mean(np.vstack(all_imp), axis=0)
        else:
            gain_imp = np.zeros(len(predictors), dtype=float)

    # Normalize
    gain_imp = gain_imp / (gain_imp.sum() + 1e-12)

    feat_names = np.array(predictors)
    order = np.argsort(gain_imp)[::-1][:TOP_N_IMPORT]

    plt.figure(figsize=(9, max(5, 0.28 * len(order))))
    plt.barh(feat_names[order][::-1], gain_imp[order][::-1])
    plt.xlabel("Relative Importance")
    plt.title(
        f"BalancedBagging+XGB — Feature Importance (Top {len(order)})\n"
        f"Neg:pos target {target_ratio}:1 (eff ≈ {eff_ratio:.2f}:1)"
    )
    plt.tight_layout()
    fi_fig_out = os.path.join(
        FIGS_DIR,
        f"feature_importance_balancedbagging_negpos{int(target_ratio):03d}.png"
    )
    plt.savefig(fi_fig_out, dpi=150)
    plt.close()
    print(f"Saved feature importance plot: {fi_fig_out}")

    # ----------------- SAVE MODEL -----------------
    model_path = os.path.join(
        MODELS_DIR,
        f"balancedbagging_xgb_model_negpos{int(target_ratio):03d}.pkl"
    )
    joblib.dump(clf, model_path)
    print(f"Saved model for neg:pos={target_ratio}:1 to: {model_path}")

    # ----------------- APPEND SUMMARY ROW -----------------
    summary_rows.append(
        dict(
            neg_pos_target      = target_ratio,
            eff_neg_pos_ratio   = round(eff_ratio, 3),
            n_pos_pool          = n_pos_pool,
            n_neg_pool          = n_neg_pool,
            n_pos_train         = n_pos_train,
            n_neg_train         = n_neg_train,
            n_pos_val           = n_pos_val,
            n_neg_val           = n_neg_val,
            n_pos_test          = test_pos,
            n_neg_test          = test_neg,
            threshold           = round(best_thr, 3),
            train_iou_best_thr  = round(train_iou, 4),
            val_iou_best_thr    = round(best_val_iou, 4),
            val_f1_best_thr     = round(best_val_f1, 4),
            test_iou            = round(test_iou, 4),
            test_precision      = round(test_prec, 4),
            test_recall         = round(test_rec, 4),
            test_f1             = round(test_f1, 4),
            test_auc_roc        = round(test_auc_roc, 4),
            test_auc_pr         = round(test_auc_pr, 4),
            bb_n_estimators     = bb_n_estimators,
            bb_max_samples      = bb_max_samples,
        )
    )

# ============================================================
# SAVE SUMMARY CSV
# ============================================================

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(
        OUT_DIR,
        "balancedbagging_trueTest_train100_negpos_sweep_metrics_auc_thresh.csv"
    )
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved BalancedBagging+XGB neg:pos sweep summary to:\n  {summary_csv}")
else:
    print("\nNo neg:pos runs were executed; summary not saved.")

print("\n✅ Done. BalancedBagging+XGB neg:pos sweeps (100→10) complete.")
