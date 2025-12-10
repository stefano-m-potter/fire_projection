import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    jaccard_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

# Datasets
TRAINVAL_DIR = "/explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_trainval_100x"
TEST_DIR     = "/explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_test_true_10pct"

# Tuned params source
OUT_ROOT_OLD  = "/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/neg_ratio_experiments_globaltest"
PARAMS_DIR    = os.path.join(OUT_ROOT_OLD, "option4_focal_loss_10x_negative_auc_thresh")
BEST_PARAMS_JSON = os.path.join(PARAMS_DIR, "tuned_xgb_focal_params.json")

# NEW Output directory (Updated for Filtering Experiment)
OUT_ROOT = OUT_ROOT_OLD
OUT_DIR  = os.path.join(OUT_ROOT, "xgb_focal_trueTest_train100_negpos_sweep_PRAUC_FIX_FILTERED")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS_DIR = os.path.join(OUT_DIR, "models")
FIGS_DIR   = os.path.join(OUT_DIR, "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

TOP_N_IMPORT = 30
NEG_POS_RATIOS = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

TEST_SIZE_GLOBAL = 0.10
VAL_SIZE_OVERALL = 0.20
VAL_SIZE_INNER   = VAL_SIZE_OVERALL / (1.0 - TEST_SIZE_GLOBAL)

# ============================================================
# Helpers
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def get_focal_wrapper(alpha, gamma):
    """
    Creates a closure for Focal Loss with a dynamic Alpha.
    This ensures we balance the loss correctly as we downsample negatives.
    """
    def focal_loss_fixed(preds, dtrain):
        y = dtrain.get_label()
        p = sigmoid(preds)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        
        # Gradient computation
        grad_pos = alpha * ((1 - p) ** gamma) * (gamma * (-np.log(p)) * (1 - p) - 1) * (p * (1 - p))
        grad_neg = (1 - alpha) * (p ** gamma) * (gamma * (-np.log(1 - p)) * p + 1) * (p * (1 - p))
        grad = np.where(y > 0.5, grad_pos, grad_neg)
        
        # Hessian computation
        hess = p * (1 - p)
        return grad, hess
    
    return focal_loss_fixed

def apply_phys_filters(df, stage_name="Dataset"):
    """
    Applies strict physical threshold filtering.
    Removes pixels with:
      - relative_humidity > 80
      - temperature_2m < 270
      - fine_fuel_moisture_code < 50
      - initial_spread_index < 2
    """
    if df.empty:
        return df
        
    initial_len = len(df)
    
    # We construct a mask for items to DROP
    # Note: If a column doesn't exist, we skip that filter to prevent errors,
    # but we assume these columns exist based on requirements.
    
    drop_mask = pd.Series(False, index=df.index)
    
    if 'relative_humidity' in df.columns:
        drop_mask |= (df['relative_humidity'] > 80)
        
    if 'temperature_2m' in df.columns:
        drop_mask |= (df['temperature_2m'] < 270)
        
    if 'fine_fuel_moisture_code' in df.columns:
        drop_mask |= (df['fine_fuel_moisture_code'] < 50)
        
    if 'initial_spread_index' in df.columns:
        drop_mask |= (df['initial_spread_index'] < 2)
    
    df_filtered = df[~drop_mask].copy()
    dropped_count = initial_len - len(df_filtered)
    
    if dropped_count > 0:
        print(f"[{stage_name}] Dropped {dropped_count:,} rows due to physical filters (RH>80, Temp<270, FFMC<50, ISI<2).")
        
    return df_filtered

def prepare_trainval_and_predictors(df: pd.DataFrame):
    df = df.copy()
    
    # --- APPLY FILTERS BEFORE PROCESSING ---
    df = apply_phys_filters(df, stage_name="Train/Val")
    
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

    if "b1" in X.columns and not pd.api.types.is_categorical_dtype(X["b1"]):
        X["b1"] = X["b1"].astype("category")
        print("\nTreating 'b1' as pandas 'category' in Train/Val.")

    coerced = 0
    for c in X.columns:
        if c == "b1" and pd.api.types.is_categorical_dtype(X[c]):
            continue
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
            coerced += 1

    if coerced:
        num_cols = [c for c in X.columns if not (c == "b1" and pd.api.types.is_categorical_dtype(X["b1"]))]
        mask = X[num_cols].notna().all(axis=1)
        if "b1" in X.columns and pd.api.types.is_categorical_dtype(X["b1"]):
            mask &= X["b1"].notna()
        before = len(X)
        X = X.loc[mask].copy()
        y = y.loc[mask].copy()
        print(f"Dropped {before - len(X):,} Train/Val rows with NaNs after coercion.")

    return X, y, predictors

def prepare_test(df: pd.DataFrame, predictors):
    df = df.copy()
    
    # --- APPLY FILTERS BEFORE PROCESSING ---
    df = apply_phys_filters(df, stage_name="Test")
    
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

    if "b1" in X.columns and not pd.api.types.is_categorical_dtype(X["b1"]):
        X["b1"] = X["b1"].astype("category")
        print("\nTreating 'b1' as pandas 'category' in Test.")

    coerced = 0
    for c in X.columns:
        if c == "b1" and pd.api.types.is_categorical_dtype(X[c]):
            continue
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
            coerced += 1

    if coerced:
        num_cols = [c for c in X.columns if not (c == "b1" and pd.api.types.is_categorical_dtype(X["b1"]))]
        mask = X[num_cols].notna().all(axis=1)
        if "b1" in X.columns and pd.api.types.is_categorical_dtype(X["b1"]):
            mask &= X["b1"].notna()
        before = len(X)
        X = X.loc[mask].copy()
        y = y.loc[mask].copy()
        print(f"Dropped {before - len(X):,} Test rows with NaNs after coercion.")

    return X, y

# ============================================================
# LOAD DATA & CONFIG
# ============================================================

print(f"Loading tuned XGBoost focal params from:\n  {BEST_PARAMS_JSON}")
with open(BEST_PARAMS_JSON, "r") as f:
    tuned_params = json.load(f)

# Extract Gamma and Rounds (Alpha will be dynamic now)
FOCAL_GAMMA      = tuned_params.get("focal_gamma", 2.0)
NUM_BOOST_ROUNDS = tuned_params.get("num_boost_rounds", 600)

# Remove keys so we can inject them cleanly later
for k in ["focal_alpha", "focal_gamma", "num_boost_rounds"]:
    tuned_params.pop(k, None)

print("\nBase Params (Metric will be forced to 'aucpr'):")
print(tuned_params)

print(f"\nLoading Train/Val 100x dataset from:\n  {TRAINVAL_DIR}")
tv_dataset = ds.dataset(TRAINVAL_DIR, format="parquet")
df_tv_full = tv_dataset.to_table().to_pandas()

print(f"\nLoading true-ratio 10% Test dataset from:\n  {TEST_DIR}")
test_dataset = ds.dataset(TEST_DIR, format="parquet")
df_test_full = test_dataset.to_table().to_pandas()

# Prepare Data (Filters applied inside)
X_tv_full, y_tv_full, predictors = prepare_trainval_and_predictors(df_tv_full)
X_test, y_test = prepare_test(df_test_full, predictors)

test_pos = int((y_test == 1).sum())
test_neg = int((y_test == 0).sum())

# Split Pools
tv_data = X_tv_full.copy()
tv_data["burned"] = y_tv_full
pos_pool = tv_data[tv_data["burned"] == 1]
neg_pool = tv_data[tv_data["burned"] == 0]

n_pos_pool = len(pos_pool)
n_neg_pool = len(neg_pool)
actual_ratio_pool = n_neg_pool / max(n_pos_pool, 1)

summary_rows = []

# ============================================================
# SWEEP LOOP
# ============================================================

for i, target_ratio in enumerate(NEG_POS_RATIOS, start=1):
    print("\n" + "=" * 80)
    print(f"=== Sweep {i}/{len(NEG_POS_RATIOS)} — target neg:pos ≈ {target_ratio}:1 ===")

    # 1. Downsample Negatives
    if target_ratio >= actual_ratio_pool:
        n_neg_target = n_neg_pool
        eff_ratio = actual_ratio_pool
        neg_subset = neg_pool
    else:
        n_neg_target = int(round(target_ratio * n_pos_pool))
        n_neg_target = min(n_neg_target, n_neg_pool)
        eff_ratio = n_neg_target / max(n_pos_pool, 1)
        neg_subset = neg_pool.sample(n=n_neg_target, random_state=RANDOM_STATE + i)

    # 2. Recombine & Split
    tv_subset = pd.concat([pos_pool, neg_subset], axis=0)
    tv_subset = tv_subset.sample(frac=1.0, random_state=RANDOM_STATE + 100 + i).reset_index(drop=True)

    if tv_subset["burned"].nunique() < 2:
        print(f"[SKIP] target_ratio={target_ratio}: only one class present.")
        continue

    train_sub, val_sub = train_test_split(
        tv_subset,
        test_size=VAL_SIZE_INNER,
        random_state=RANDOM_STATE,
        stratify=tv_subset["burned"],
    )

    X_train = train_sub[predictors].copy()
    y_train = train_sub["burned"].astype("uint8")
    X_val   = val_sub[predictors].copy()
    y_val   = val_sub["burned"].astype("uint8")

    n_pos_train = int((y_train == 1).sum())
    n_neg_train = int((y_train == 0).sum())
    print(f"Train subset: Pos={n_pos_train:,}, Neg={n_neg_train:,} (Ratio {n_neg_train/n_pos_train:.2f}:1)")

    # 3. Dynamic Alpha Calculation
    # Heuristic: n_neg / (n_pos + n_neg) -> higher alpha for more imbalance
    # This keeps the gradients balanced regardless of sampling ratio.
    current_alpha = n_neg_train / (n_pos_train + n_neg_train)
    print(f"Dynamic FOCAL_ALPHA set to: {current_alpha:.4f}")

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval   = xgb.DMatrix(X_val,   label=y_val,   enable_categorical=True)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  enable_categorical=True)

    # 4. Train with Early Stopping on PR-AUC
    # We use our wrapper to inject the specific alpha for this iteration
    loss_func = get_focal_wrapper(current_alpha, FOCAL_GAMMA)
    
    # Force eval_metric to aucpr
    run_params = tuned_params.copy()
    run_params['eval_metric'] = 'aucpr'
    run_params['disable_default_eval_metric'] = 1

    evals = [(dtrain, "train"), (dval, "validation")]
    evals_result = {}

    print("Training XGBoost (Early Stopping on Val PR-AUC)...")
    booster = xgb.train(
        run_params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=evals,
        obj=loss_func,                 # Custom dynamic objective
        evals_result=evals_result,
        verbose_eval=50,
        early_stopping_rounds=50,      # Stop if PR-AUC doesn't improve
        maximize=True,                 # PR-AUC is higher = better
    )

    best_iter = booster.best_iteration
    best_score_val = booster.best_score # This is the best PR-AUC on Val
    print(f"Best Iteration: {best_iter} | Best Val PR-AUC: {best_score_val:.4f}")

    # 5. Extract Learning Curves (PR-AUC)
    train_aucpr = evals_result['train']['aucpr']
    val_aucpr   = evals_result['validation']['aucpr']

    plt.figure(figsize=(10, 6))
    plt.plot(train_aucpr, label="Train PR-AUC")
    plt.plot(val_aucpr,   label="Val PR-AUC")
    plt.axvline(best_iter, linestyle="--", color='red', label=f"Best Iter ({best_iter})")
    plt.xlabel("Boosting Rounds")
    plt.ylabel("PR-AUC")
    plt.title(f"PR-AUC Learning Curve (Ratio {target_ratio}:1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    curve_path = os.path.join(FIGS_DIR, f"prauc_curve_xgb_negpos{int(target_ratio):03d}.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()

    # 6. Final Test Eval
    # Predict using the best iteration
    test_margin = booster.predict(dtest, iteration_range=(0, best_iter + 1))
    y_test_proba = sigmoid(test_margin)

    # Calculate AUCs
    test_auc_roc = roc_auc_score(y_test, y_test_proba)
    test_auc_pr  = average_precision_score(y_test, y_test_proba)

    # Calculate Point Metrics (F1, IoU) at Best Threshold
    prec, rec, thresholds = precision_recall_curve(y_test, y_test_proba)
    # Handle edge case where last precision is 1.0 and recall is 0.0
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-12)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_test_hat = (y_test_proba >= best_thr).astype("uint8")
    
    test_iou  = jaccard_score(y_test, y_test_hat, average="binary", zero_division=0)
    test_prec = precision_score(y_test, y_test_hat, zero_division=0)
    test_rec  = recall_score(y_test, y_test_hat, zero_division=0)
    test_f1   = f1_score(y_test, y_test_hat, zero_division=0)

    print("\n==== FINAL TEST METRICS ====")
    print(f"PR AUC (Optimization Target): {test_auc_pr:.4f}")
    print(f"ROC AUC                     : {test_auc_roc:.4f}")
    print(f"Best Threshold (Max F1)     : {best_thr:.3f}")
    print(f"Test IoU                    : {test_iou:.4f}")

    # 7. PR Curve Plot
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"PR (AUC = {test_auc_pr:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (Neg:Pos {target_ratio}:1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pr_fig_out = os.path.join(FIGS_DIR, f"pr_curve_xgb_negpos{int(target_ratio):03d}.png")
    plt.savefig(pr_fig_out, dpi=150)
    plt.close()

    # 8. Save Model
    model_path = os.path.join(MODELS_DIR, f"xgb_focal_model_negpos{int(target_ratio):03d}.pkl")
    joblib.dump(booster, model_path)

    summary_rows.append(dict(
        neg_pos_target      = target_ratio,
        eff_neg_pos_ratio   = round(eff_ratio, 3),
        n_pos_train         = n_pos_train,
        n_neg_train         = n_neg_train,
        focal_alpha         = round(current_alpha, 4),
        focal_gamma         = FOCAL_GAMMA,
        best_iteration      = best_iter,
        test_auc_pr         = round(test_auc_pr, 4),
        test_auc_roc        = round(test_auc_roc, 4),
        test_iou            = round(test_iou, 4),
        test_f1             = round(test_f1, 4),
        threshold           = round(best_thr, 4)
    ))

# Save Summary
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUT_DIR, "xgb_focal_negpos_sweep_PRAUC_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary to: {summary_csv}")

print("\n✅ Done. PR-AUC Optimized Sweep Complete.")