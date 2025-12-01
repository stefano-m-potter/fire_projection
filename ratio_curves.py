# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Option 4 (Focal loss, global fixed test set) — single model:
# - Uses partitioned parquet dataset: parquet_cems_with_fraction_dataset
# - Label: burned = 1 if fraction > 0.5, else 0
# - Train/Val/Test split with stratification so class proportions are similar
#   across all three sets.
# - Try LightGBM core API with custom focal loss via fobj.
# - If LightGBM lacks `fobj`, fall back to XGBoost with the same focal loss.
# - Threshold is selected to MAXIMIZE F1 on the validation set (no recall floor).
# - Threshold stored as a true probability in [0,1]; metrics rounded in CSV.

# Outputs under:
# .../neg_ratio_experiments_globaltest/option4_focal_loss_all_data/
# """

# import os, inspect
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import lightgbm as lgb

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     precision_recall_curve,
#     jaccard_score,
#     precision_score,
#     recall_score,
#     f1_score,
# )

# # ----------------- CONFIG -----------------
# PARQUET_IN    = "/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/cems_with_fraction_balanced_10x.parquet"
# RANDOM_STATE  = 42

# TEST_SIZE_GLOBAL = 0.10      # fraction of full data reserved as fixed test set
# VAL_SIZE_OVERALL = 0.20      # overall fraction of data used as validation

# THRESH_INIT   = 0.50         # only for IoU logging
# TOP_N_IMPORT  = 30

# FOCAL_ALPHA   = 0.25
# FOCAL_GAMMA   = 2.0

# LGB_PARAMS = dict(
#     boosting_type="gbdt",
#     learning_rate=0.05,
#     num_leaves=48,
#     min_data_in_leaf=100,
#     feature_fraction=0.75,
#     bagging_fraction=0.75,
#     bagging_freq=5,
#     lambda_l2=2.0,
#     n_jobs=-1,
#     metric="aucpr",
# )

# OUT_ROOT = "/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/neg_ratio_experiments_globaltest"
# OUT_DIR  = os.path.join(OUT_ROOT, "option4_focal_loss_10x_negative")
# os.makedirs(OUT_DIR, exist_ok=True)

# # ----------------- Helpers -----------------
# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-x))

# def lgb_has_fobj():
#     try:
#         import inspect as _inspect
#         sig = _inspect.signature(lgb.train)
#         return "fobj" in sig.parameters
#     except Exception:
#         return False

# # --- FOCAL LOSS for LightGBM (margin -> grad/hess) ---
# def focal_loss_lgb(y_pred, dataset):
#     y_true = dataset.get_label()
#     p = sigmoid(y_pred)
#     p = np.clip(p, 1e-7, 1 - 1e-7)
#     a, g = FOCAL_ALPHA, FOCAL_GAMMA

#     # Stable approximate focal gradients
#     grad_pos = a * ((1 - p) ** g) * (g * (-np.log(p)) * (1 - p) - 1) * (p * (1 - p))
#     grad_neg = (1 - a) * (p ** g) * (g * (-np.log(1 - p)) * p + 1) * (p * (1 - p))
#     grad = np.where(y_true > 0.5, grad_pos, grad_neg)

#     # Approximate hessian with logistic hessian
#     hess = p * (1 - p)
#     return grad, hess

# def iou_metric_lgb(y_pred, dataset):
#     y_true = dataset.get_label()
#     y_hat  = (sigmoid(y_pred) >= THRESH_INIT).astype(np.uint8)
#     iou    = jaccard_score(y_true, y_hat, average="binary", zero_division=0)
#     return "IoU", float(iou), True

# # ----------------- LOAD & PREP -----------------
# print(f"Loading partitioned parquet dataset from: {PARQUET_IN}")
# df = pd.read_parquet(PARQUET_IN)
# if "fraction" not in df.columns:
#     raise ValueError("Expected column 'fraction' in dataset.")

# df["fraction"] = df["fraction"].astype("float32").clip(0, 1)
# before = len(df)
# df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()
# print(f"Dropped {before - len(df):,} rows with NaNs/±inf; {len(df):,} remain.")

# # Label: burned = 1 if fraction > 0.5, else 0
# df["burned"] = (df["fraction"] > 0.5).astype(np.uint8)
# print("\nClass counts (burned = 1 if fraction>0.5):")
# print(df["burned"].value_counts(dropna=False))

# drop_cols = {"fraction", "burned", "bin", "year", "month", "latitude", "longitude"}
# predictors = [c for c in df.columns if c not in drop_cols]

# X_full = df[predictors].copy()
# y_full = df["burned"].astype(np.uint8)

# # Treat land cover as categorical if present
# if "b1" in X_full.columns and not pd.api.types.is_categorical_dtype(X_full["b1"]):
#     X_full["b1"] = X_full["b1"].astype("category")
#     print("\nTreating 'b1' as pandas 'category'.")

# # Coerce any non-numeric predictors (except categorical b1) to numeric
# coerced = 0
# for c in X_full.columns:
#     if c == "b1" and pd.api.types.is_categorical_dtype(X_full[c]):
#         continue
#     if not np.issubdtype(X_full[c].dtype, np.number):
#         X_full[c] = pd.to_numeric(X_full[c], errors="coerce")
#         coerced += 1

# if coerced:
#     pre = len(X_full)
#     num_cols = [
#         c for c in X_full.columns
#         if not (c == "b1" and pd.api.types.is_categorical_dtype(X_full["b1"]))
#     ]
#     mask = X_full[num_cols].notna().all(axis=1)
#     if "b1" in X_full.columns and pd.api.types.is_categorical_dtype(X_full["b1"]):
#         mask &= X_full["b1"].notna()
#     X_full = X_full.loc[mask].copy()
#     y_full = y_full.loc[X_full.index]
#     print(f"Coerced {coerced} column(s); dropped {pre - len(X_full):,} rows post-coercion.")
# print(f"\nPredictor columns: {len(X_full.columns)}")

# data = X_full.copy()
# data["burned"] = y_full

# # ----------------- SPLITS: train/val/test with stratification -----------------
# # 1) TrainVal vs Test (fixed global test set, stratified)
# idx_trainval, idx_test = train_test_split(
#     data.index,
#     test_size=TEST_SIZE_GLOBAL,
#     random_state=RANDOM_STATE,
#     stratify=data["burned"],
# )
# trainval = data.loc[idx_trainval].copy()
# test     = data.loc[idx_test].copy()

# print("\nGlobal split sizes (true distribution in test):")
# print(f"  Train/Val pool: {len(trainval):,}")
# print(f"  Test (fixed)  : {len(test):,}")
# print("\nTest set class counts:")
# print(test["burned"].value_counts())

# X_test = test[predictors].copy()
# y_test = test["burned"].astype(np.uint8)

# # 2) Train vs Val (within TrainVal, stratified)
# val_size_inner = VAL_SIZE_OVERALL / (1.0 - TEST_SIZE_GLOBAL)
# train, val = train_test_split(
#     trainval,
#     test_size=val_size_inner,
#     random_state=RANDOM_STATE,
#     stratify=trainval["burned"],
# )

# print("\nTrain/Val split sizes (within TrainVal pool):")
# print(f"  Train: {len(train):,}")
# print(f"  Val  : {len(val):,}")
# print("\nTrain class counts:")
# print(train["burned"].value_counts())
# print("\nVal class counts:")
# print(val["burned"].value_counts())

# X_train = train[predictors].copy()
# y_train = train["burned"].astype(np.uint8)
# X_val   = val[predictors].copy()
# y_val   = val["burned"].astype(np.uint8)

# # Test set class counts (constant)
# test_pos = int((y_test == 1).sum())
# test_neg = int((y_test == 0).sum())
# print(f"\nTest set positives (1): {test_pos:,}")
# print(f"Test set negatives (0): {test_neg:,}")

# USE_LGB_FOBJ = lgb_has_fobj()
# if not USE_LGB_FOBJ:
#     print("\n[INFO] LightGBM build lacks `fobj` support on train(); falling back to XGBoost for focal loss.")
#     import xgboost as xgb  # imported only when needed

# # ----------------- TRAIN SINGLE MODEL -----------------
# evals_result = {}
# if USE_LGB_FOBJ:
#     # ---------- LightGBM path with custom fobj ----------
#     train_set = lgb.Dataset(X_train, label=y_train)
#     val_set   = lgb.Dataset(X_val, label=y_val, reference=train_set)
#     params = LGB_PARAMS.copy()
#     params["seed"] = RANDOM_STATE
#     params["objective"] = "binary"  # overridden by fobj

#     booster = lgb.train(
#         params,
#         train_set,
#         num_boost_round=10000,
#         valid_sets=[train_set, val_set],
#         valid_names=["train", "validation"],
#         fobj=focal_loss_lgb,
#         feval=iou_metric_lgb,
#         callbacks=[
#             lgb.early_stopping(stopping_rounds=50),
#             lgb.log_evaluation(period=50),
#             lgb.record_evaluation(evals_result),
#         ]
#     )

#     # Learning curve (IoU)
#     if "IoU" in evals_result.get("train", {}):
#         plt.figure(figsize=(8,5))
#         plt.plot(evals_result["train"]["IoU"], label="Train IoU (focal)")
#         plt.plot(evals_result["validation"]["IoU"], label="Validation IoU (focal)")
#         plt.xlabel("Boosting Rounds"); plt.ylabel("IoU (Jaccard)")
#         plt.title(
#             "Option 4 (Focal-LGB): Train vs Val IoU\n"
#             f"THRESH_INIT={THRESH_INIT:.2f}"
#         )
#         plt.legend(); plt.grid(True); plt.tight_layout()
#         iou_fig_out = os.path.join(
#             OUT_DIR,
#             "iou_curve_focal_all_data.png"
#         )
#         plt.savefig(iou_fig_out, dpi=150); plt.close()
#         print(f"Saved focal IoU curve: {iou_fig_out}")

#     # Validation probabilities (convert margins to probs)
#     y_val_proba = sigmoid(
#         booster.predict(X_val, num_iteration=booster.best_iteration)
#     )
#     # ---------- end LightGBM path ----------

# else:
#     # ---------- XGBoost fallback with custom objective ----------
#     import xgboost as xgb

#     def focal_loss_xgb(preds, dtrain):
#         y = dtrain.get_label()
#         p = sigmoid(preds)
#         p = np.clip(p, 1e-7, 1 - 1e-7)
#         a, g = FOCAL_ALPHA, FOCAL_GAMMA
#         grad_pos = a * ((1 - p) ** g) * (g * (-np.log(p)) * (1 - p) - 1) * (p * (1 - p))
#         grad_neg = (1 - a) * (p ** g) * (g * (-np.log(1 - p)) * p + 1) * (p * (1 - p))
#         grad = np.where(y > 0.5, grad_pos, grad_neg)
#         hess = p * (1 - p)
#         return grad, hess

#     def iou_metric_xgb(preds, dtrain):
#         y = dtrain.get_label()
#         y_hat = (sigmoid(preds) >= THRESH_INIT).astype(np.uint8)
#         iou = jaccard_score(y, y_hat, average="binary", zero_division=0)
#         return "IoU", float(iou)

#     dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
#     dval   = xgb.DMatrix(X_val,   label=y_val,   enable_categorical=True)

#     params_xgb = dict(
#         booster="gbtree",
#         eta=0.05,
#         max_depth=0,        # use `max_leaves` with tree_method=hist
#         max_leaves=48,
#         subsample=0.75,
#         colsample_bytree=0.75,
#         reg_lambda=2.0,
#         tree_method="hist",   # or "gpu_hist" if you want GPU
#         objective="reg:logistic",  # overridden by custom obj
#         eval_metric="aucpr",
#         seed=RANDOM_STATE,
#         nthread=-1,
#     )

#     watchlist = [(dtrain, "train"), (dval, "validation")]
#     booster = xgb.train(
#         params_xgb,
#         dtrain,
#         num_boost_round=10000,
#         evals=watchlist,
#         obj=focal_loss_xgb,
#         feval=iou_metric_xgb,
#         verbose_eval=50,
#         early_stopping_rounds=50,
#     )

#     # Validation probabilities (margins -> probs)
#     y_val_proba = sigmoid(
#         booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
#     )
#     # ---------- end XGBoost path ----------

# # ----------------- THRESHOLD SELECTION (VAL) — MAXIMIZE F1 -----------------
# prec, rec, thr = precision_recall_curve(y_val, y_val_proba)
# # thr has length len(prec) - 1; drop last point to align
# prec_ = prec[:-1]
# rec_  = rec[:-1]
# f1_vals = 2 * prec_ * rec_ / (prec_ + rec_ + 1e-12)
# best_idx = int(np.argmax(f1_vals))
# best_thr = float(thr[best_idx])
# best_f1_val = float(f1_vals[best_idx])

# print("\nValidation threshold selection (focal, all_data):")
# print(f"  Best threshold (max F1): {best_thr:.3f}")
# print(f"  Val F1 @ best_thr      : {best_f1_val:.3f}")

# # ----------------- TEST METRICS (FIXED GLOBAL TEST) -----------------
# if USE_LGB_FOBJ:
#     y_test_proba = sigmoid(
#         booster.predict(X_test, num_iteration=booster.best_iteration)
#     )
# else:
#     import xgboost as xgb
#     dtest = xgb.DMatrix(X_test, enable_categorical=True)
#     y_test_proba = sigmoid(
#         booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
#     )

# y_test_hat = (y_test_proba >= best_thr).astype(np.uint8)
# test_iou  = jaccard_score(y_test, y_test_hat, average="binary", zero_division=0)
# test_prec = precision_score(y_test, y_test_hat, zero_division=0)
# test_rec  = recall_score(y_test, y_test_hat, zero_division=0)
# test_f1   = f1_score(y_test, y_test_hat, zero_division=0)

# print("\n==== FINAL TEST METRICS (focal, fixed global test set, all_data) ====")
# print(f"Threshold (max F1 on val): {best_thr:.3f}")
# print(f"IoU (Jaccard)            : {test_iou:.2f}")
# print(f"Precision                : {test_prec:.2f}")
# print(f"Recall                   : {test_rec:.2f}")
# print(f"F1 Score                 : {test_f1:.2f}")

# # ----------------- FEATURE IMPORTANCE -----------------
# if USE_LGB_FOBJ:
#     gain_imp = booster.feature_importance(importance_type="gain")
#     feat_names = np.array(X_train.columns)
# else:
#     fmap = booster.get_score(importance_type="gain")
#     feat_names = np.array(X_train.columns)
#     gain_imp = np.array(
#         [fmap.get(f"f{i}", 0.0) for i in range(len(feat_names))],
#         dtype=float,
#     )

# gain_imp = gain_imp / (gain_imp.sum() + 1e-12)
# order = np.argsort(gain_imp)[::-1][:TOP_N_IMPORT]

# plt.figure(figsize=(9, max(5, 0.28 * len(order))))
# plt.barh(feat_names[order][::-1], gain_imp[order][::-1])
# plt.xlabel("Relative Gain Importance")
# plt.title(
#     f"Option 4 (Focal, {('LGB' if USE_LGB_FOBJ else 'XGB')}): "
#     f"Feature Importance (Top {len(order)}) — all_data"
# )
# plt.tight_layout()
# fi_fig_out = os.path.join(
#     OUT_DIR,
#     "feature_importance_focal_all_data.png"
# )
# plt.savefig(fi_fig_out, dpi=150)
# plt.close()
# print(f"Saved focal feature importance plot: {fi_fig_out}")

# # ----------------- SUMMARY CSV (single row) -----------------
# summary_row = dict(
#     focal_alpha=FOCAL_ALPHA,
#     focal_gamma=FOCAL_GAMMA,
#     threshold=round(best_thr, 3),
#     val_f1=round(best_f1_val, 3),
#     test_pos=test_pos,
#     test_neg=test_neg,
#     test_iou=round(test_iou, 2),
#     test_precision=round(test_prec, 2),
#     test_recall=round(test_rec, 2),
#     test_f1=round(test_f1, 2),
#     best_iteration=int(
#         getattr(booster, "best_iteration", getattr(booster, "best_ntree_limit", 0))
#     ),
#     backend=("lightgbm" if USE_LGB_FOBJ else "xgboost"),
# )

# summary_df = pd.DataFrame([summary_row])
# summary_csv = os.path.join(
#     OUT_DIR,
#     "option4_focal_globaltest_singlemodel_all_data_metrics.csv"
# )
# summary_df.to_csv(summary_csv, index=False)
# print(f"\nSaved Option 4 (focal, all_data) single-model summary to: {summary_csv}")



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Option 4 (Focal loss, global fixed test set) — negative-fraction sweep:

- Input parquet (already balanced to ~10x negatives):
    cems_with_fraction_balanced_10x.parquet

- Label: burned = 1 if fraction > 0.5, else 0

- Step 1: On the FULL 10x dataset:
    * Split into TrainVal vs Test (stratified) once.
    * Test is fixed and reused for all experiments.

- Step 2: On the TrainVal pool ONLY:
    * For each negative fraction f in [1.0, 0.9, ..., 0.1]:
        - Keep ALL positives
        - Sample f * (all negatives in TrainVal)
        - Stratified Train vs Val split
        - Train focal-loss model
        - Pick threshold to MAXIMIZE F1 on validation
        - Evaluate on the SAME fixed test set
        - Save model + IoU curve + feature importance

Outputs under:
.../neg_ratio_experiments_globaltest/option4_focal_loss_10x_negative/

Summary CSV:
  option4_focal_globaltest_neg_fraction_sweep_metrics.csv
"""

import os, inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    jaccard_score,
    precision_score,
    recall_score,
    f1_score,
)

# ----------------- CONFIG -----------------
PARQUET_IN    = "/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/cems_with_fraction_balanced_10x.parquet"
RANDOM_STATE  = 42

TEST_SIZE_GLOBAL = 0.10      # fraction of full data reserved as fixed test set
VAL_SIZE_OVERALL = 0.20      # overall fraction of data used as validation

THRESH_INIT   = 0.50         # only for IoU logging
TOP_N_IMPORT  = 30

FOCAL_ALPHA   = 0.25
FOCAL_GAMMA   = 2.0

# Fractions of negatives in TrainVal pool to keep (all positives kept)
NEG_FRAC_STEPS = [1.0, 0.9, 0.8, 0.7, 0.6,
                  0.5, 0.4, 0.3, 0.2, 0.1]

LGB_PARAMS = dict(
    boosting_type="gbdt",
    learning_rate=0.05,
    num_leaves=48,
    min_data_in_leaf=100,
    feature_fraction=0.75,
    bagging_fraction=0.75,
    bagging_freq=5,
    lambda_l2=2.0,
    n_jobs=-1,
    metric="aucpr",
)

OUT_ROOT = "/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/neg_ratio_experiments_globaltest"
OUT_DIR  = os.path.join(OUT_ROOT, "option4_focal_loss_10x_negative")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------- Helpers -----------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def lgb_has_fobj():
    try:
        import inspect as _inspect
        sig = _inspect.signature(lgb.train)
        return "fobj" in sig.parameters
    except Exception:
        return False

# --- FOCAL LOSS for LightGBM (margin -> grad/hess) ---
def focal_loss_lgb(y_pred, dataset):
    y_true = dataset.get_label()
    p = sigmoid(y_pred)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    a, g = FOCAL_ALPHA, FOCAL_GAMMA

    # Stable approximate focal gradients
    grad_pos = a * ((1 - p) ** g) * (g * (-np.log(p)) * (1 - p) - 1) * (p * (1 - p))
    grad_neg = (1 - a) * (p ** g) * (g * (-np.log(1 - p)) * p + 1) * (p * (1 - p))
    grad = np.where(y_true > 0.5, grad_pos, grad_neg)

    # Approximate hessian with logistic hessian
    hess = p * (1 - p)
    return grad, hess

def iou_metric_lgb(y_pred, dataset):
    y_true = dataset.get_label()
    y_hat  = (sigmoid(y_pred) >= THRESH_INIT).astype(np.uint8)
    iou    = jaccard_score(y_true, y_hat, average="binary", zero_division=0)
    return "IoU", float(iou), True

# ----------------- LOAD & PREP -----------------
print(f"Loading balanced 10x parquet dataset from: {PARQUET_IN}")
df = pd.read_parquet(PARQUET_IN)
if "fraction" not in df.columns:
    raise ValueError("Expected column 'fraction' in dataset.")

df["fraction"] = df["fraction"].astype("float32").clip(0, 1)
before = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()
print(f"Dropped {before - len(df):,} rows with NaNs/±inf; {len(df):,} remain.")

# Label: burned = 1 if fraction > 0.5, else 0
df["burned"] = (df["fraction"] > 0.5).astype(np.uint8)
print("\nClass counts (burned = 1 if fraction>0.5):")
print(df["burned"].value_counts(dropna=False))
print(df["burned"].value_counts(normalize=True).mul(100))

drop_cols = {"fraction", "burned", "bin", "year", "month", "latitude", "longitude"}
predictors = [c for c in df.columns if c not in drop_cols]

X_full = df[predictors].copy()
y_full = df["burned"].astype(np.uint8)

# Treat land cover as categorical if present
if "b1" in X_full.columns and not pd.api.types.is_categorical_dtype(X_full["b1"]):
    X_full["b1"] = X_full["b1"].astype("category")
    print("\nTreating 'b1' as pandas 'category'.")

# Coerce any non-numeric predictors (except categorical b1) to numeric
coerced = 0
for c in X_full.columns:
    if c == "b1" and pd.api.types.is_categorical_dtype(X_full[c]):
        continue
    if not np.issubdtype(X_full[c].dtype, np.number):
        X_full[c] = pd.to_numeric(X_full[c], errors="coerce")
        coerced += 1

if coerced:
    pre = len(X_full)
    num_cols = [
        c for c in X_full.columns
        if not (c == "b1" and pd.api.types.is_categorical_dtype(X_full["b1"]))
    ]
    mask = X_full[num_cols].notna().all(axis=1)
    if "b1" in X_full.columns and pd.api.types.is_categorical_dtype(X_full["b1"]):
        mask &= X_full["b1"].notna()
    X_full = X_full.loc[mask].copy()
    y_full = y_full.loc[X_full.index]
    print(f"Coerced {coerced} column(s); dropped {pre - len(X_full):,} rows post-coercion.")
print(f"\nPredictor columns: {len(X_full.columns)}")

data = X_full.copy()
data["burned"] = y_full

# ----------------- FIXED GLOBAL TEST SPLIT -----------------
idx_trainval, idx_test = train_test_split(
    data.index,
    test_size=TEST_SIZE_GLOBAL,
    random_state=RANDOM_STATE,
    stratify=data["burned"],
)
trainval = data.loc[idx_trainval].copy()
test     = data.loc[idx_test].copy()

print("\nGlobal split sizes (true distribution in test):")
print(f"  Train/Val pool: {len(trainval):,}")
print(f"  Test (fixed)  : {len(test):,}")
print("\nTest set class counts:")
print(test["burned"].value_counts())
print(test["burned"].value_counts(normalize=True).mul(100))

X_test = test[predictors].copy()
y_test = test["burned"].astype(np.uint8)

test_pos = int((y_test == 1).sum())
test_neg = int((y_test == 0).sum())
print(f"\nTest set positives (1): {test_pos:,}")
print(f"Test set negatives (0): {test_neg:,}")

# Split TrainVal into pos / neg for downsampling experiments
pos_tv = trainval[trainval["burned"] == 1]
neg_tv = trainval[trainval["burned"] == 0]
n_pos_tv, n_neg_tv = len(pos_tv), len(neg_tv)
print("\nTrain/Val pool class counts (before any downsampling):")
print(trainval["burned"].value_counts())
print(trainval["burned"].value_counts(normalize=True).mul(100))
print(f"\nTrain/Val positives: {n_pos_tv:,}")
print(f"Train/Val negatives: {n_neg_tv:,}")
neg_per_pos_initial = n_neg_tv / max(n_pos_tv, 1)
print(f"Initial neg:pos ratio in TrainVal ~ {neg_per_pos_initial:.2f}:1")

USE_LGB_FOBJ = lgb_has_fobj()
if not USE_LGB_FOBJ:
    print("\n[INFO] LightGBM build lacks `fobj` support on train(); falling back to XGBoost for focal loss.")
    import xgboost as xgb  # imported only when needed

summary_rows = []

# ----------------- NEGATIVE FRACTION SWEEP -----------------
for step_idx, frac in enumerate(NEG_FRAC_STEPS):
    frac_pct = int(round(frac * 100))
    print("\n" + "=" * 80)
    print(f"=== Negative Fraction Step: {frac:.1f} (approx {frac_pct}% of TrainVal negatives kept) ===")

    # Number of negatives to sample in TrainVal
    neg_target = int(round(frac * n_neg_tv))
    neg_target = max(1, neg_target)
    # Effective neg:pos ratio in the TrainVal subset (before Train/Val split)
    eff_ratio = neg_target / max(n_pos_tv, 1)
    print(f"Target negatives in TrainVal subset: {neg_target:,}")
    print(f"Effective neg:pos ratio in TrainVal subset ~ {eff_ratio:.2f}:1")

    # Sample negatives and combine with ALL positives from TrainVal
    neg_tv_sample = neg_tv.sample(neg_target, random_state=RANDOM_STATE + step_idx)
    tv_subset = (
        pd.concat([pos_tv, neg_tv_sample], axis=0)
        .sample(frac=1.0, random_state=RANDOM_STATE + 100 + step_idx)
        .reset_index(drop=True)
    )

    print("TrainVal subset class counts (for this fraction):")
    print(tv_subset["burned"].value_counts())
    print(tv_subset["burned"].value_counts(normalize=True).mul(100))

    X_tv = tv_subset[predictors].copy()
    y_tv = tv_subset["burned"].astype(np.uint8)

    # Train vs Val split within this TrainVal subset
    val_size_inner = VAL_SIZE_OVERALL / (1.0 - TEST_SIZE_GLOBAL)
    train_sub, val_sub = train_test_split(
        tv_subset,
        test_size=val_size_inner,
        random_state=RANDOM_STATE,
        stratify=tv_subset["burned"],
    )

    print("\nTrain/Val subset split sizes:")
    print(f"  Train: {len(train_sub):,}")
    print(f"  Val  : {len(val_sub):,}")
    print("\nTrain subset class counts:")
    print(train_sub["burned"].value_counts())
    print("\nVal subset class counts:")
    print(val_sub["burned"].value_counts())

    X_train = train_sub[predictors].copy()
    y_train = train_sub["burned"].astype(np.uint8)
    X_val   = val_sub[predictors].copy()
    y_val   = val_sub["burned"].astype(np.uint8)

    # ----------------- TRAIN SINGLE MODEL FOR THIS FRACTION -----------------
    evals_result = {}
    backend = "lightgbm"

    if USE_LGB_FOBJ:
        # ---------- LightGBM path with custom fobj ----------
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set   = lgb.Dataset(X_val, label=y_val, reference=train_set)
        params = LGB_PARAMS.copy()
        params["seed"] = RANDOM_STATE
        params["objective"] = "binary"  # overridden by fobj

        booster = lgb.train(
            params,
            train_set,
            num_boost_round=10000,
            valid_sets=[train_set, val_set],
            valid_names=["train", "validation"],
            fobj=focal_loss_lgb,
            feval=iou_metric_lgb,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50),
                lgb.record_evaluation(evals_result),
            ]
        )

        # Learning curve (IoU)
        if "IoU" in evals_result.get("train", {}):
            plt.figure(figsize=(8,5))
            plt.plot(evals_result["train"]["IoU"], label="Train IoU (focal)")
            plt.plot(evals_result["validation"]["IoU"], label="Validation IoU (focal)")
            plt.xlabel("Boosting Rounds"); plt.ylabel("IoU (Jaccard)")
            plt.title(
                f"Option 4 (Focal-LGB): Train vs Val IoU\n"
                f"Neg fraction={frac:.1f} (≈{eff_ratio:.2f}:1 in TrainVal subset)"
            )
            plt.legend(); plt.grid(True); plt.tight_layout()
            iou_fig_out = os.path.join(
                OUT_DIR,
                f"iou_curve_focal_negfrac{frac_pct:03d}.png"
            )
            plt.savefig(iou_fig_out, dpi=150); plt.close()
            print(f"Saved focal IoU curve: {iou_fig_out}")

        # Validation probabilities (convert margins to probs)
        y_val_proba = sigmoid(
            booster.predict(X_val, num_iteration=booster.best_iteration)
        )

    else:
        # ---------- XGBoost fallback with custom objective ----------
        import xgboost as xgb
        backend = "xgboost"

        def focal_loss_xgb(preds, dtrain):
            y = dtrain.get_label()
            p = sigmoid(preds)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            a, g = FOCAL_ALPHA, FOCAL_GAMMA
            grad_pos = a * ((1 - p) ** g) * (g * (-np.log(p)) * (1 - p) - 1) * (p * (1 - p))
            grad_neg = (1 - a) * (p ** g) * (g * (-np.log(1 - p)) * p + 1) * (p * (1 - p))
            grad = np.where(y > 0.5, grad_pos, grad_neg)
            hess = p * (1 - p)
            return grad, hess

        def iou_metric_xgb(preds, dtrain):
            y = dtrain.get_label()
            y_hat = (sigmoid(preds) >= THRESH_INIT).astype(np.uint8)
            iou = jaccard_score(y, y_hat, average="binary", zero_division=0)
            return "IoU", float(iou)

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval   = xgb.DMatrix(X_val,   label=y_val,   enable_categorical=True)

        params_xgb = dict(
            booster="gbtree",
            eta=0.05,
            max_depth=0,        # use `max_leaves` with tree_method=hist
            max_leaves=48,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=2.0,
            tree_method="hist",   # or "gpu_hist" if you want GPU
            objective="reg:logistic",  # overridden by custom obj
            eval_metric="aucpr",
            seed=RANDOM_STATE,
            nthread=-1,
        )

        watchlist = [(dtrain, "train"), (dval, "validation")]
        booster = xgb.train(
            params_xgb,
            dtrain,
            num_boost_round=10000,
            evals=watchlist,
            obj=focal_loss_xgb,
            feval=iou_metric_xgb,
            verbose_eval=50,
            early_stopping_rounds=50,
        )

        # Validation probabilities (margins -> probs)
        y_val_proba = sigmoid(
            booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        )

    # ----------------- THRESHOLD SELECTION (VAL) — MAXIMIZE F1 -----------------
    prec, rec, thr = precision_recall_curve(y_val, y_val_proba)
    prec_ = prec[:-1]
    rec_  = rec[:-1]
    f1_vals = 2 * prec_ * rec_ / (prec_ + rec_ + 1e-12)
    best_idx = int(np.argmax(f1_vals))
    best_thr = float(thr[best_idx])
    best_f1_val = float(f1_vals[best_idx])

    print("\nValidation threshold selection (focal, neg fraction sweep):")
    print(f"  Neg fraction (TrainVal)     : {frac:.1f} (≈{eff_ratio:.2f}:1)")
    print(f"  Best threshold (max F1)     : {best_thr:.3f}")
    print(f"  Val F1 @ best_thr           : {best_f1_val:.3f}")

    # ----------------- TEST METRICS (FIXED GLOBAL TEST) -----------------
    if USE_LGB_FOBJ:
        y_test_proba = sigmoid(
            booster.predict(X_test, num_iteration=booster.best_iteration)
        )
    else:
        import xgboost as xgb
        dtest = xgb.DMatrix(X_test, enable_categorical=True)
        y_test_proba = sigmoid(
            booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
        )

    y_test_hat = (y_test_proba >= best_thr).astype(np.uint8)
    test_iou  = jaccard_score(y_test, y_test_hat, average="binary", zero_division=0)
    test_prec = precision_score(y_test, y_test_hat, zero_division=0)
    test_rec  = recall_score(y_test, y_test_hat, zero_division=0)
    test_f1   = f1_score(y_test, y_test_hat, zero_division=0)

    print("\n==== FINAL TEST METRICS (focal, fixed global test set) ====")
    print(f"Neg fraction (TrainVal)    : {frac:.1f} (≈{eff_ratio:.2f}:1)")
    print(f"Threshold (max F1 on val)  : {best_thr:.3f}")
    print(f"IoU (Jaccard)              : {test_iou:.2f}")
    print(f"Precision                  : {test_prec:.2f}")
    print(f"Recall                     : {test_rec:.2f}")
    print(f"F1 Score                   : {test_f1:.2f}")

    # ----------------- FEATURE IMPORTANCE -----------------
    if USE_LGB_FOBJ:
        gain_imp = booster.feature_importance(importance_type="gain")
        feat_names = np.array(X_train.columns)
    else:
        fmap = booster.get_score(importance_type="gain")
        feat_names = np.array(X_train.columns)
        gain_imp = np.array(
            [fmap.get(f"f{i}", 0.0) for i in range(len(feat_names))],
            dtype=float,
        )

    gain_imp = gain_imp / (gain_imp.sum() + 1e-12)
    order = np.argsort(gain_imp)[::-1][:TOP_N_IMPORT]

    plt.figure(figsize=(9, max(5, 0.28 * len(order))))
    plt.barh(feat_names[order][::-1], gain_imp[order][::-1])
    plt.xlabel("Relative Gain Importance")
    plt.title(
        f"Option 4 (Focal, {backend}): Feature Importance (Top {len(order)})\n"
        f"Neg fraction={frac:.1f} (≈{eff_ratio:.2f}:1 in TrainVal subset)"
    )
    plt.tight_layout()
    fi_fig_out = os.path.join(
        OUT_DIR,
        f"feature_importance_focal_negfrac{frac_pct:03d}.png"
    )
    plt.savefig(fi_fig_out, dpi=150)
    plt.close()
    print(f"Saved focal feature importance plot: {fi_fig_out}")

    # ----------------- SAVE MODEL -----------------
    model_path = os.path.join(
        MODELS_DIR,
        f"focal_model_negfrac{frac_pct:03d}_{backend}.pkl"
    )
    joblib.dump(booster, model_path)
    print(f"Saved model for neg fraction {frac:.1f} to: {model_path}")

    # ----------------- APPEND SUMMARY ROW -----------------
    summary_rows.append(
        dict(
            neg_fraction=frac,
            neg_fraction_pct=frac_pct,
            eff_neg_pos_ratio=round(eff_ratio, 3),
            n_pos_train=int((train_sub["burned"] == 1).sum()),
            n_neg_train=int((train_sub["burned"] == 0).sum()),
            focal_alpha=FOCAL_ALPHA,
            focal_gamma=FOCAL_GAMMA,
            threshold=round(best_thr, 3),
            val_f1=round(best_f1_val, 3),
            test_pos=test_pos,
            test_neg=test_neg,
            test_iou=round(test_iou, 2),
            test_precision=round(test_prec, 2),
            test_recall=round(test_rec, 2),
            test_f1=round(test_f1, 2),
            best_iteration=int(
                getattr(booster, "best_iteration", getattr(booster, "best_ntree_limit", 0))
            ),
            backend=backend,
        )
    )

# ----------------- SUMMARY CSV (multi-row) -----------------
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(
        OUT_DIR,
        "option4_focal_globaltest_neg_fraction_sweep_metrics.csv"
    )
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved Option 4 (focal) neg-fraction sweep summary to: {summary_csv}")
else:
    print("\nNo neg-fraction runs were executed; summary not saved.")

