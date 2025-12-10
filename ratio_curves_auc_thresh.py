# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-

# # """
# # Option 4 (Focal loss, global fixed test set) — negative-fraction sweep + AUC,
# # with IoU-at-best-threshold learning curves.

# # - Input parquet (already balanced to ~10x negatives):
# #     cems_with_fraction_balanced_10x.parquet

# # - Label: burned = 1 if fraction > 0.5, else 0

# # - Step 1: On the FULL 10x dataset:
# #     * Split into TrainVal vs Test (stratified) once.
# #     * Test is fixed and reused for all experiments.

# # - Step 2: On the TrainVal pool ONLY:
# #     * For each negative fraction f in [1.0, 0.9, ..., 0.1]:
# #         - Keep ALL positives
# #         - Sample f * (all negatives in TrainVal)
# #         - Stratified Train vs Val split
# #         - Train focal-loss model for a fixed NUM_BOOST_ROUNDS trees
# #         - For each iteration:
# #             * Predict on val
# #             * Find best F1 threshold for that iteration
# #             * Compute IoU on val + train at that threshold
# #         - Define best_iteration as the iteration with max val IoU
# #         - Use that iteration & threshold to:
# #             * Compute final test metrics
# #             * Plot ROC and PR curves
# #         - Save:
# #             * IoU-at-best-threshold learning curve (train & val)
# #             * Feature importance PNG
# #             * ROC curve PNG (with AUC)
# #             * Precision–Recall curve PNG (with AUPRC)
# #             * Model file
# #             * Row in summary CSV with IoU/F1 + ROC AUC + PR AUC + best_iteration
# # """

# # import os
# # import inspect
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import lightgbm as lgb
# # import joblib

# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import (
# #     precision_recall_curve,
# #     jaccard_score,
# #     precision_score,
# #     recall_score,
# #     f1_score,
# #     roc_auc_score,
# #     average_precision_score,
# #     roc_curve,
# # )

# # # ----------------- CONFIG -----------------
# # PARQUET_IN    = "/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/cems_with_fraction_balanced_10x.parquet"
# # RANDOM_STATE  = 42

# # TEST_SIZE_GLOBAL = 0.10      # fraction of full data reserved as fixed test set
# # VAL_SIZE_OVERALL = 0.20      # overall fraction of data used as validation

# # TOP_N_IMPORT  = 30

# # FOCAL_ALPHA   = 0.25
# # FOCAL_GAMMA   = 2.0

# # # Fixed number of boosting rounds (trees)
# # NUM_BOOST_ROUNDS = 600  # change to 100 if you want exactly 100 trees

# # # Fractions of negatives in TrainVal pool to keep (all positives kept)
# # NEG_FRAC_STEPS = [
# #     1.0, 0.9, 0.8, 0.7, 0.6,
# #     0.5, 0.4, 0.3, 0.2, 0.1
# # ]

# # LGB_PARAMS = dict(
# #     boosting_type="gbdt",
# #     learning_rate=0.05,
# #     num_leaves=48,
# #     min_data_in_leaf=100,
# #     feature_fraction=0.75,
# #     bagging_fraction=0.75,
# #     bagging_freq=5,
# #     lambda_l2=2.0,
# #     n_jobs=-1,
# #     metric="aucpr",
# # )

# # OUT_ROOT = "/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/neg_ratio_experiments_globaltest"
# # OUT_DIR  = os.path.join(OUT_ROOT, "option4_focal_loss_10x_negative_auc_thresh")
# # os.makedirs(OUT_DIR, exist_ok=True)

# # MODELS_DIR = os.path.join(OUT_DIR, "models")
# # os.makedirs(MODELS_DIR, exist_ok=True)

# # # ----------------- Helpers -----------------
# # def sigmoid(x):
# #     return 1.0 / (1.0 + np.exp(-x))

# # def lgb_has_fobj():
# #     try:
# #         sig = inspect.signature(lgb.train)
# #         return "fobj" in sig.parameters
# #     except Exception:
# #         return False

# # # --- FOCAL LOSS for LightGBM (margin -> grad/hess) ---
# # def focal_loss_lgb(y_pred, dataset):
# #     y_true = dataset.get_label()
# #     p = sigmoid(y_pred)
# #     p = np.clip(p, 1e-7, 1 - 1e-7)
# #     a, g = FOCAL_ALPHA, FOCAL_GAMMA

# #     # Stable approximate focal gradients
# #     grad_pos = a * ((1 - p) ** g) * (g * (-np.log(p)) * (1 - p) - 1) * (p * (1 - p))
# #     grad_neg = (1 - a) * (p ** g) * (g * (-np.log(1 - p)) * p + 1) * (p * (1 - p))
# #     grad = np.where(y_true > 0.5, grad_pos, grad_neg)

# #     # Approximate hessian with logistic hessian
# #     hess = p * (1 - p)
# #     return grad, hess

# # # ----------------- LOAD & PREP -----------------
# # print(f"Loading balanced 10x parquet dataset from: {PARQUET_IN}")
# # df = pd.read_parquet(PARQUET_IN)
# # if "fraction" not in df.columns:
# #     raise ValueError("Expected column 'fraction' in dataset.")

# # df["fraction"] = df["fraction"].astype("float32").clip(0, 1)
# # before = len(df)
# # df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()
# # print(f"Dropped {before - len(df):,} rows with NaNs/±inf; {len(df):,} remain.")

# # # Label: burned = 1 if fraction > 0.5, else 0
# # df["burned"] = (df["fraction"] > 0.5).astype(np.uint8)
# # print("\nClass counts (burned = 1 if fraction>0.5):")
# # print(df["burned"].value_counts(dropna=False))
# # print(df["burned"].value_counts(normalize=True).mul(100))

# # drop_cols = {"fraction", "burned", "bin", "year", "month", "latitude", "longitude"}
# # predictors = [c for c in df.columns if c not in drop_cols]

# # X_full = df[predictors].copy()
# # y_full = df["burned"].astype(np.uint8)

# # # Treat land cover as categorical if present
# # if "b1" in X_full.columns and not pd.api.types.is_categorical_dtype(X_full["b1"]):
# #     X_full["b1"] = X_full["b1"].astype("category")
# #     print("\nTreating 'b1' as pandas 'category'.")

# # # Coerce any non-numeric predictors (except categorical b1) to numeric
# # coerced = 0
# # for c in X_full.columns:
# #     if c == "b1" and pd.api.types.is_categorical_dtype(X_full[c]):
# #         continue
# #     if not np.issubdtype(X_full[c].dtype, np.number):
# #         X_full[c] = pd.to_numeric(X_full[c], errors="coerce")
# #         coerced += 1

# # if coerced:
# #     pre = len(X_full)
# #     num_cols = [
# #         c for c in X_full.columns
# #         if not (c == "b1" and pd.api.types.is_categorical_dtype(X_full["b1"]))
# #     ]
# #     mask = X_full[num_cols].notna().all(axis=1)
# #     if "b1" in X_full.columns and pd.api.types.is_categorical_dtype(X_full["b1"]):
# #         mask &= X_full["b1"].notna()
# #     X_full = X_full.loc[mask].copy()
# #     y_full = y_full.loc[X_full.index]
# #     print(f"Coerced {coerced} column(s); dropped {pre - len(X_full):,} rows post-coercion.")
# # print(f"\nPredictor columns: {len(X_full.columns)}")

# # data = X_full.copy()
# # data["burned"] = y_full

# # # ----------------- FIXED GLOBAL TEST SPLIT -----------------
# # idx_trainval, idx_test = train_test_split(
# #     data.index,
# #     test_size=TEST_SIZE_GLOBAL,
# #     random_state=RANDOM_STATE,
# #     stratify=data["burned"],
# # )
# # trainval = data.loc[idx_trainval].copy()
# # test     = data.loc[idx_test].copy()

# # print("\nGlobal split sizes (true distribution in test):")
# # print(f"  Train/Val pool: {len(trainval):,}")
# # print(f"  Test (fixed)  : {len(test):,}")
# # print("\nTest set class counts:")
# # print(test["burned"].value_counts())
# # print(test["burned"].value_counts(normalize=True).mul(100))

# # X_test = test[predictors].copy()
# # y_test = test["burned"].astype(np.uint8)

# # test_pos = int((y_test == 1).sum())
# # test_neg = int((y_test == 0).sum())
# # print(f"\nTest set positives (1): {test_pos:,}")
# # print(f"Test set negatives (0): {test_neg:,}")

# # # Split TrainVal into pos / neg for downsampling experiments
# # pos_tv = trainval[trainval["burned"] == 1]
# # neg_tv = trainval[trainval["burned"] == 0]
# # n_pos_tv, n_neg_tv = len(pos_tv), len(neg_tv)
# # print("\nTrain/Val pool class counts (before any downsampling):")
# # print(trainval["burned"].value_counts())
# # print(trainval["burned"].value_counts(normalize=True).mul(100))
# # print(f"\nTrain/Val positives: {n_pos_tv:,}")
# # print(f"Train/Val negatives: {n_neg_tv:,}")
# # neg_per_pos_initial = n_neg_tv / max(n_pos_tv, 1)
# # print(f"Initial neg:pos ratio in TrainVal ~ {neg_per_pos_initial:.2f}:1")

# # USE_LGB_FOBJ = lgb_has_fobj()
# # if not USE_LGB_FOBJ:
# #     print("\n[INFO] LightGBM build lacks `fobj` support on train(); falling back to XGBoost for focal loss.")
# #     import xgboost as xgb  # imported only when needed

# # summary_rows = []

# # # ----------------- NEGATIVE FRACTION SWEEP -----------------
# # for step_idx, frac in enumerate(NEG_FRAC_STEPS):
# #     frac_pct = int(round(frac * 100))
# #     print("\n" + "=" * 80)
# #     print(f"=== Negative Fraction Step: {frac:.1f} (approx {frac_pct}% of TrainVal negatives kept) ===")

# #     # Number of negatives to sample in TrainVal
# #     neg_target = int(round(frac * n_neg_tv))
# #     neg_target = max(1, neg_target)
# #     # Effective neg:pos ratio in the TrainVal subset (before Train/Val split)
# #     eff_ratio = neg_target / max(n_pos_tv, 1)
# #     print(f"Target negatives in TrainVal subset: {neg_target:,}")
# #     print(f"Effective neg:pos ratio in TrainVal subset ~ {eff_ratio:.2f}:1")

# #     # Sample negatives and combine with ALL positives from TrainVal
# #     neg_tv_sample = neg_tv.sample(neg_target, random_state=RANDOM_STATE + step_idx)
# #     tv_subset = (
# #         pd.concat([pos_tv, neg_tv_sample], axis=0)
# #         .sample(frac=1.0, random_state=RANDOM_STATE + 100 + step_idx)
# #         .reset_index(drop=True)
# #     )

# #     print("TrainVal subset class counts (for this fraction):")
# #     print(tv_subset["burned"].value_counts())
# #     print(tv_subset["burned"].value_counts(normalize=True).mul(100))

# #     # Train vs Val split within this TrainVal subset
# #     val_size_inner = VAL_SIZE_OVERALL / (1.0 - TEST_SIZE_GLOBAL)
# #     train_sub, val_sub = train_test_split(
# #         tv_subset,
# #         test_size=val_size_inner,
# #         random_state=RANDOM_STATE,
# #         stratify=tv_subset["burned"],
# #     )

# #     print("\nTrain/Val subset split sizes:")
# #     print(f"  Train: {len(train_sub):,}")
# #     print(f"  Val  : {len(val_sub):,}")
# #     print("\nTrain subset class counts:")
# #     print(train_sub["burned"].value_counts())
# #     print("\nVal subset class counts:")
# #     print(val_sub["burned"].value_counts())

# #     X_train = train_sub[predictors].copy()
# #     y_train = train_sub["burned"].astype(np.uint8)
# #     X_val   = val_sub[predictors].copy()
# #     y_val   = val_sub["burned"].astype(np.uint8)

# #     evals_result = {}
# #     backend = "lightgbm"
# #     best_iter = NUM_BOOST_ROUNDS  # will be updated

# #     # ----------------- TRAIN SINGLE MODEL FOR THIS FRACTION -----------------
# #     if USE_LGB_FOBJ:
# #         # ---------- LightGBM path with custom fobj ----------
# #         train_set = lgb.Dataset(X_train, label=y_train)
# #         val_set   = lgb.Dataset(X_val, label=y_val, reference=train_set)
# #         params = LGB_PARAMS.copy()
# #         params["seed"] = RANDOM_STATE
# #         params["objective"] = "binary"  # overridden by fobj

# #         booster = lgb.train(
# #             params,
# #             train_set,
# #             num_boost_round=NUM_BOOST_ROUNDS,
# #             valid_sets=[train_set, val_set],
# #             valid_names=["train", "validation"],
# #             fobj=focal_loss_lgb,
# #             callbacks=[
# #                 lgb.log_evaluation(period=50),
# #                 lgb.record_evaluation(evals_result),
# #             ]
# #         )

# #     else:
# #         # ---------- XGBoost fallback with custom objective ----------
# #         import xgboost as xgb
# #         backend = "xgboost"

# #         def focal_loss_xgb(preds, dtrain):
# #             y = dtrain.get_label()
# #             p = sigmoid(preds)
# #             p = np.clip(p, 1e-7, 1 - 1e-7)
# #             a, g = FOCAL_ALPHA, FOCAL_GAMMA
# #             grad_pos = a * ((1 - p) ** g) * (g * (-np.log(p)) * (1 - p) - 1) * (p * (1 - p))
# #             grad_neg = (1 - a) * (p ** g) * (g * (-np.log(1 - p)) * p + 1) * (p * (1 - p))
# #             grad = np.where(y > 0.5, grad_pos, grad_neg)
# #             hess = p * (1 - p)
# #             return grad, hess

# #         dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
# #         dval   = xgb.DMatrix(X_val,   label=y_val,   enable_categorical=True)

# #         params_xgb = dict(
# #             booster="gbtree",
# #             eta=0.05,
# #             max_depth=0,        # use `max_leaves` with tree_method=hist
# #             max_leaves=48,
# #             subsample=0.75,
# #             colsample_bytree=0.75,
# #             reg_lambda=2.0,
# #             tree_method="hist",   # or "gpu_hist" if you want GPU
# #             objective="reg:logistic",  # overridden by custom obj
# #             eval_metric="aucpr",
# #             seed=RANDOM_STATE,
# #             nthread=-1,
# #         )

# #         evals = [(dtrain, "train"), (dval, "validation")]
# #         booster = xgb.train(
# #             params_xgb,
# #             dtrain,
# #             num_boost_round=NUM_BOOST_ROUNDS,
# #             evals=evals,
# #             obj=focal_loss_xgb,
# #             verbose_eval=50,
# #             evals_result=evals_result,
# #         )

# #     # ----------------- IoU-at-best-threshold LEARNING CURVES -----------------
# #     train_iou_curve = []
# #     val_iou_curve = []
# #     thr_curve = []

# #     print("\nComputing IoU-at-best-threshold curves across iterations...")
# #     for it in range(1, NUM_BOOST_ROUNDS + 1):
# #         if USE_LGB_FOBJ:
# #             # margins -> probs
# #             val_margin = booster.predict(X_val, num_iteration=it)
# #             train_margin = booster.predict(X_train, num_iteration=it)
# #         else:
# #             import xgboost as xgb
# #             dval   = xgb.DMatrix(X_val,   enable_categorical=True)
# #             dtrain = xgb.DMatrix(X_train, enable_categorical=True)
# #             val_margin   = booster.predict(dval,   iteration_range=(0, it))
# #             train_margin = booster.predict(dtrain, iteration_range=(0, it))

# #         val_proba   = sigmoid(val_margin)
# #         train_proba = sigmoid(train_margin)

# #         # Find best threshold on VAL for this iteration (max F1)
# #         prec_val, rec_val, thr_val = precision_recall_curve(y_val, val_proba)
# #         prec_ = prec_val[:-1]
# #         rec_  = rec_val[:-1]
# #         f1_vals = 2 * prec_ * rec_ / (prec_ + rec_ + 1e-12)
# #         best_idx_it = int(np.argmax(f1_vals))
# #         best_thr_it = float(thr_val[best_idx_it])

# #         thr_curve.append(best_thr_it)

# #         # IoU on VAL at its best threshold for this iteration
# #         y_val_hat_it = (val_proba >= best_thr_it).astype(np.uint8)
# #         val_iou_it = jaccard_score(y_val, y_val_hat_it, average="binary", zero_division=0)
# #         val_iou_curve.append(val_iou_it)

# #         # IoU on TRAIN at same threshold
# #         y_train_hat_it = (train_proba >= best_thr_it).astype(np.uint8)
# #         train_iou_it = jaccard_score(y_train, y_train_hat_it, average="binary", zero_division=0)
# #         train_iou_curve.append(train_iou_it)

# #     train_iou_curve = np.array(train_iou_curve)
# #     val_iou_curve   = np.array(val_iou_curve)
# #     thr_curve       = np.array(thr_curve)

# #     # Best iteration = argmax val IoU curve (1-based)
# #     best_iter_idx = int(np.argmax(val_iou_curve))
# #     best_iter = int(best_iter_idx + 1)
# #     best_thr  = float(thr_curve[best_iter_idx])
# #     best_val_iou = float(val_iou_curve[best_iter_idx])

# #     print(f"\nBest iteration based on val IoU-at-best-threshold: {best_iter}")
# #     print(f"  Val IoU at best_iter : {best_val_iou:.3f}")
# #     print(f"  Threshold at best_iter (val F1-optimal): {best_thr:.3f}")

# #     # Plot IoU learning curves
# #     plt.figure(figsize=(10, 6))
# #     plt.plot(train_iou_curve, label="Train IoU (best thr per iter)")
# #     plt.plot(val_iou_curve,   label="Validation IoU (best thr per iter)")
# #     plt.axvline(best_iter_idx, linestyle="--", label=f"best_iter={best_iter}")
# #     plt.xlabel("Boosting Rounds")
# #     plt.ylabel("IoU (Jaccard)")
# #     plt.title(
# #         f"Option 4 (Focal-{backend.upper()}): Train vs Val IoU-at-best-threshold\n"
# #         f"Neg fraction={frac:.1f} (≈{eff_ratio:.2f}:1 in TrainVal subset)"
# #     )
# #     plt.legend()
# #     plt.grid(True)
# #     plt.tight_layout()
# #     iou_fig_out = os.path.join(
# #         OUT_DIR,
# #         f"iou_curve_bestthr_focal_negfrac{frac_pct:03d}.png"
# #     )
# #     plt.savefig(iou_fig_out, dpi=150)
# #     plt.close()
# #     print(f"Saved IoU-at-best-threshold learning curve: {iou_fig_out}")

# #     # ----------------- FINAL TEST METRICS USING best_iter & best_thr -----------------
# #     if USE_LGB_FOBJ:
# #         test_margin = booster.predict(X_test, num_iteration=best_iter)
# #     else:
# #         import xgboost as xgb
# #         dtest = xgb.DMatrix(X_test, enable_categorical=True)
# #         test_margin = booster.predict(dtest, iteration_range=(0, best_iter))

# #     y_test_proba = sigmoid(test_margin)
# #     y_test_hat   = (y_test_proba >= best_thr).astype(np.uint8)

# #     test_iou  = jaccard_score(y_test, y_test_hat, average="binary", zero_division=0)
# #     test_prec = precision_score(y_test, y_test_hat, zero_division=0)
# #     test_rec  = recall_score(y_test, y_test_hat, zero_division=0)
# #     test_f1   = f1_score(y_test, y_test_hat, zero_division=0)

# #     # Threshold-free metrics (AUCs) on fixed test set
# #     fpr, tpr, roc_thr = roc_curve(y_test, y_test_proba)
# #     test_auc_roc = roc_auc_score(y_test, y_test_proba)

# #     prec_curve_test, rec_curve_test, pr_thr_test = precision_recall_curve(y_test, y_test_proba)
# #     test_auc_pr = average_precision_score(y_test, y_test_proba)

# #     print("\n==== FINAL TEST METRICS (focal, fixed global test set) ====")
# #     print(f"Neg fraction (TrainVal)    : {frac:.1f} (≈{eff_ratio:.2f}:1)")
# #     print(f"Best iteration (val IoU)   : {best_iter}")
# #     print(f"Threshold (val best F1)    : {best_thr:.3f}")
# #     print(f"IoU (Jaccard)              : {test_iou:.2f}")
# #     print(f"Precision                  : {test_prec:.2f}")
# #     print(f"Recall                     : {test_rec:.2f}")
# #     print(f"F1 Score                   : {test_f1:.2f}")
# #     print(f"ROC AUC                    : {test_auc_roc:.3f}")
# #     print(f"PR AUC (Average Precision) : {test_auc_pr:.3f}")

# #     # ---- Save ROC curve plot ----
# #     plt.figure(figsize=(6, 5))
# #     plt.plot(fpr, tpr, label=f"ROC (AUC = {test_auc_roc:.3f})")
# #     plt.plot([0, 1], [0, 1], linestyle="--")
# #     plt.xlabel("False Positive Rate")
# #     plt.ylabel("True Positive Rate")
# #     plt.title(
# #         f"Option 4 (Focal, {backend}): ROC Curve\n"
# #         f"Neg fraction={frac:.1f} (≈{eff_ratio:.2f}:1 in TrainVal subset)"
# #     )
# #     plt.legend()
# #     plt.grid(True)
# #     plt.tight_layout()
# #     roc_fig_out = os.path.join(
# #         OUT_DIR,
# #         f"roc_curve_focal_negfrac{frac_pct:03d}.png"
# #     )
# #     plt.savefig(roc_fig_out, dpi=150)
# #     plt.close()
# #     print(f"Saved ROC curve: {roc_fig_out}")

# #     # ---- Save Precision–Recall curve plot ----
# #     plt.figure(figsize=(6, 5))
# #     plt.plot(rec_curve_test, prec_curve_test, label=f"PR (AUC = {test_auc_pr:.3f})")
# #     plt.xlabel("Recall")
# #     plt.ylabel("Precision")
# #     plt.title(
# #         f"Option 4 (Focal, {backend}): Precision–Recall Curve\n"
# #         f"Neg fraction={frac:.1f} (≈{eff_ratio:.2f}:1 in TrainVal subset)"
# #     )
# #     plt.legend()
# #     plt.grid(True)
# #     plt.tight_layout()
# #     pr_fig_out = os.path.join(
# #         OUT_DIR,
# #         f"pr_curve_focal_negfrac{frac_pct:03d}.png"
# #     )
# #     plt.savefig(pr_fig_out, dpi=150)
# #     plt.close()
# #     print(f"Saved Precision–Recall curve: {pr_fig_out}")

# #     # ----------------- FEATURE IMPORTANCE -----------------
# #     if USE_LGB_FOBJ:
# #         gain_imp = booster.feature_importance(importance_type="gain")
# #         feat_names = np.array(X_train.columns)
# #     else:
# #         fmap = booster.get_score(importance_type="gain")
# #         feat_names = np.array(X_train.columns)
# #         gain_imp = np.array(
# #             [fmap.get(f"f{i}", 0.0) for i in range(len(feat_names))],
# #             dtype=float,
# #         )

# #     gain_imp = gain_imp / (gain_imp.sum() + 1e-12)
# #     order = np.argsort(gain_imp)[::-1][:TOP_N_IMPORT]

# #     plt.figure(figsize=(9, max(5, 0.28 * len(order))))
# #     plt.barh(feat_names[order][::-1], gain_imp[order][::-1])
# #     plt.xlabel("Relative Gain Importance")
# #     plt.title(
# #         f"Option 4 (Focal, {backend}): Feature Importance (Top {len(order)})\n"
# #         f"Neg fraction={frac:.1f} (≈{eff_ratio:.2f}:1 in TrainVal subset)"
# #     )
# #     plt.tight_layout()
# #     fi_fig_out = os.path.join(
# #         OUT_DIR,
# #         f"feature_importance_focal_negfrac{frac_pct:03d}.png"
# #     )
# #     plt.savefig(fi_fig_out, dpi=150)
# #     plt.close()
# #     print(f"Saved focal feature importance plot: {fi_fig_out}")

# #     # ----------------- SAVE MODEL -----------------
# #     model_path = os.path.join(
# #         MODELS_DIR,
# #         f"focal_model_negfrac{frac_pct:03d}_{backend}.pkl"
# #     )
# #     joblib.dump(booster, model_path)
# #     print(f"Saved model for neg fraction {frac:.1f} to: {model_path}")

# #     # ----------------- APPEND SUMMARY ROW -----------------
# #     summary_rows.append(
# #         dict(
# #             neg_fraction=frac,
# #             neg_fraction_pct=frac_pct,
# #             eff_neg_pos_ratio=round(eff_ratio, 3),
# #             n_pos_train=int((train_sub["burned"] == 1).sum()),
# #             n_neg_train=int((train_sub["burned"] == 0).sum()),
# #             focal_alpha=FOCAL_ALPHA,
# #             focal_gamma=FOCAL_GAMMA,
# #             threshold=round(best_thr, 3),
# #             val_iou_best=round(best_val_iou, 3),
# #             test_pos=test_pos,
# #             test_neg=test_neg,
# #             test_iou=round(test_iou, 2),
# #             test_precision=round(test_prec, 2),
# #             test_recall=round(test_rec, 2),
# #             test_f1=round(test_f1, 2),
# #             test_auc_roc=round(test_auc_roc, 3),
# #             test_auc_pr=round(test_auc_pr, 3),
# #             best_iteration=int(best_iter),
# #             backend=backend,
# #         )
# #     )

# # # ----------------- SUMMARY CSV (multi-row) -----------------
# # if summary_rows:
# #     summary_df = pd.DataFrame(summary_rows)
# #     summary_csv = os.path.join(
# #         OUT_DIR,
# #         "option4_focal_globaltest_neg_fraction_sweep_metrics_auc_thresh.csv"
# #     )
# #     summary_df.to_csv(summary_csv, index=False)
# #     print(f"\nSaved Option 4 (focal) neg-fraction sweep summary with AUCs + IoU-thresh to: {summary_csv}")
# # else:
# #     print("\nNo neg-fraction runs were executed; summary not saved.")


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# XGBoost Focal Loss — neg:pos sweep (100, 90, ..., 10) using:

# - Train/Val dataset with ~100:1 neg:pos:
#     /explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_trainval_100x

# - 10% fixed global test set with true ~4169:1 neg:pos:
#     /explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_test_true_10pct

# - Tuned XGBoost focal params loaded from:
#     .../option4_focal_loss_10x_negative_auc_thresh/tuned_xgb_focal_params.json

# For each desired neg:pos ratio R in [100, 90, ..., 10]:

#   - Use ALL positives from Train/Val 100x pool.
#   - Sample negatives to target ≈ R:1 neg:pos (cannot exceed pool's actual ratio).
#   - Stratified Train vs Val split (same effective 20% overall Val as before).
#   - Train focal-loss XGBoost with tuned params for NUM_BOOST_ROUNDS trees.
#   - For each iteration:
#         * Predict on Val
#         * Find best F1 threshold for that iteration
#         * Compute IoU on Train + Val at that threshold
#     - Define best_iteration as iteration with max Val IoU.
#     - Use that iteration & threshold to:
#         * Compute final test metrics on the true-ratio test set
#         * Plot ROC and PR curves
#         * Save artifacts + summary CSV.
# """

# import os
# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import xgboost as xgb
# import joblib

# import pyarrow.dataset as ds

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     precision_recall_curve,
#     jaccard_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     average_precision_score,
#     roc_curve,
# )

# # ============================================================
# # CONFIG
# # ============================================================

# RANDOM_STATE = 42

# # These are the datasets you just created
# TRAINVAL_DIR = "/explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_trainval_100x"
# TEST_DIR     = "/explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_test_true_10pct"

# # Where tuned XGB focal params were saved from the tuning script
# OUT_ROOT_OLD  = "/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/neg_ratio_experiments_globaltest"
# PARAMS_DIR    = os.path.join(OUT_ROOT_OLD, "option4_focal_loss_10x_negative_auc_thresh")
# BEST_PARAMS_JSON = os.path.join(PARAMS_DIR, "tuned_xgb_focal_params.json")

# # New output directory for this experiment
# OUT_ROOT = OUT_ROOT_OLD
# OUT_DIR  = os.path.join(OUT_ROOT, "xgb_focal_trueTest_train100_negpos_sweep")
# os.makedirs(OUT_DIR, exist_ok=True)

# MODELS_DIR = os.path.join(OUT_DIR, "models")
# FIGS_DIR   = os.path.join(OUT_DIR, "figures")
# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(FIGS_DIR, exist_ok=True)

# TOP_N_IMPORT = 30

# # Ratios we want to explore (neg:pos in Train/Val subset)
# NEG_POS_RATIOS = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

# # Same semantics as before: 10% test overall, 20% val overall => ~22.22% of TrainVal
# TEST_SIZE_GLOBAL = 0.10   # informational now (already applied upstream)
# VAL_SIZE_OVERALL = 0.20
# VAL_SIZE_INNER   = VAL_SIZE_OVERALL / (1.0 - TEST_SIZE_GLOBAL)

# # ============================================================
# # Helpers
# # ============================================================

# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-x))


# def focal_loss_xgb(preds, dtrain):
#     """
#     Custom focal loss for XGBoost using raw logits.

#     preds: raw scores (logits) from the model
#     dtrain: xgb.DMatrix
#     """
#     y = dtrain.get_label()
#     p = sigmoid(preds)
#     p = np.clip(p, 1e-7, 1 - 1e-7)
#     a, g = FOCAL_ALPHA, FOCAL_GAMMA

#     # Focal loss grad for positives and negatives
#     grad_pos = a * ((1 - p) ** g) * (g * (-np.log(p)) * (1 - p) - 1) * (p * (1 - p))
#     grad_neg = (1 - a) * (p ** g) * (g * (-np.log(1 - p)) * p + 1) * (p * (1 - p))
#     grad = np.where(y > 0.5, grad_pos, grad_neg)

#     # Approximate Hessian by logistic Hessian
#     hess = p * (1 - p)
#     return grad, hess


# def prepare_trainval_and_predictors(df: pd.DataFrame):
#     """
#     From the Train/Val 100x dataframe:

#       - Ensure `burned` exists (0/1 label; recompute from fraction if needed).
#       - Define predictors (drop fraction/burned/coords/etc).
#       - Handle `b1` categorical.
#       - Coerce non-numeric predictors to numeric.
#       - Drop rows with NaNs in predictors (and NaN b1 if categorical).
#     """
#     df = df.copy()

#     if "fraction" in df.columns:
#         df["fraction"] = df["fraction"].astype("float32").clip(0, 1)

#     if "burned" not in df.columns:
#         if "fraction" not in df.columns:
#             raise ValueError("Need either 'burned' or 'fraction' in Train/Val dataset.")
#         df["burned"] = (df["fraction"] > 0.5).astype("uint8")

#     df = df.replace([np.inf, -np.inf], np.nan)

#     drop_cols = {"fraction", "burned", "bin", "year", "month", "latitude", "longitude"}
#     predictors = [c for c in df.columns if c not in drop_cols]

#     X = df[predictors].copy()
#     y = df["burned"].astype("uint8")

#     # Treat land cover as categorical if present
#     if "b1" in X.columns and not pd.api.types.is_categorical_dtype(X["b1"]):
#         X["b1"] = X["b1"].astype("category")
#         print("\nTreating 'b1' as pandas 'category' in Train/Val.")

#     # Coerce non-numeric predictors (except categorical b1) to numeric
#     coerced = 0
#     for c in X.columns:
#         if c == "b1" and pd.api.types.is_categorical_dtype(X[c]):
#             continue
#         if not np.issubdtype(X[c].dtype, np.number):
#             X[c] = pd.to_numeric(X[c], errors="coerce")
#             coerced += 1

#     if coerced:
#         num_cols = [
#             c for c in X.columns
#             if not (c == "b1" and pd.api.types.is_categorical_dtype(X["b1"]))
#         ]
#         mask = X[num_cols].notna().all(axis=1)
#         if "b1" in X.columns and pd.api.types.is_categorical_dtype(X["b1"]):
#             mask &= X["b1"].notna()

#         before = len(X)
#         X = X.loc[mask].copy()
#         y = y.loc[mask].copy()
#         print(f"Dropped {before - len(X):,} Train/Val rows with NaNs after coercion.")

#     return X, y, predictors


# def prepare_test(df: pd.DataFrame, predictors):
#     """
#     Prepare X_test, y_test using the same predictor set as Train/Val.
#     """
#     df = df.copy()

#     if "fraction" in df.columns:
#         df["fraction"] = df["fraction"].astype("float32").clip(0, 1)

#     if "burned" not in df.columns:
#         if "fraction" not in df.columns:
#             raise ValueError("Need either 'burned' or 'fraction' in Test dataset.")
#         df["burned"] = (df["fraction"] > 0.5).astype("uint8")

#     df = df.replace([np.inf, -np.inf], np.nan)

#     missing_preds = [c for c in predictors if c not in df.columns]
#     if missing_preds:
#         raise ValueError(f"Test dataset is missing predictor columns: {missing_preds}")

#     X = df[predictors].copy()
#     y = df["burned"].astype("uint8")

#     # Handle categorical b1 identical to Train/Val
#     if "b1" in X.columns and not pd.api.types.is_categorical_dtype(X["b1"]):
#         X["b1"] = X["b1"].astype("category")
#         print("\nTreating 'b1' as pandas 'category' in Test.")

#     # Coerce any non-numeric predictor (except categorical b1) to numeric
#     coerced = 0
#     for c in X.columns:
#         if c == "b1" and pd.api.types.is_categorical_dtype(X["b1"]):
#             continue
#         if not np.issubdtype(X[c].dtype, np.number):
#             X[c] = pd.to_numeric(X[c], errors="coerce")
#             coerced += 1

#     if coerced:
#         num_cols = [
#             c for c in X.columns
#             if not (c == "b1" and pd.api.types.is_categorical_dtype(X["b1"]))
#         ]
#         mask = X[num_cols].notna().all(axis=1)
#         if "b1" in X.columns and pd.api.types.is_categorical_dtype(X["b1"]):
#             mask &= X["b1"].notna()

#         before = len(X)
#         X = X.loc[mask].copy()
#         y = y.loc[mask].copy()
#         print(f"Dropped {before - len(X):,} Test rows with NaNs after coercion.")

#     return X, y


# # ============================================================
# # LOAD TUNED PARAMS
# # ============================================================

# print(f"Loading tuned XGBoost focal params from:\n  {BEST_PARAMS_JSON}")
# with open(BEST_PARAMS_JSON, "r") as f:
#     tuned_params = json.load(f)

# # Extract focal alpha/gamma and num_boost_rounds
# FOCAL_ALPHA      = tuned_params.get("focal_alpha", 0.25)
# FOCAL_GAMMA      = tuned_params.get("focal_gamma", 2.0)
# NUM_BOOST_ROUNDS = tuned_params.get("num_boost_rounds", 600)

# # Remove non-XGB params
# for k in ["focal_alpha", "focal_gamma", "num_boost_rounds"]:
#     tuned_params.pop(k, None)

# print("\nUsing XGBoost params:")
# print(tuned_params)
# print(f"NUM_BOOST_ROUNDS = {NUM_BOOST_ROUNDS}")
# print(f"FOCAL_ALPHA      = {FOCAL_ALPHA}")
# print(f"FOCAL_GAMMA      = {FOCAL_GAMMA}")

# # ============================================================
# # LOAD TRAINVAL 100x + TEST TRUE-RATIO
# # ============================================================

# print(f"\nLoading Train/Val 100x dataset from:\n  {TRAINVAL_DIR}")
# tv_dataset = ds.dataset(TRAINVAL_DIR, format="parquet")
# tv_table   = tv_dataset.to_table()
# df_tv_full = tv_table.to_pandas()
# print(f"Train/Val 100x raw size: {len(df_tv_full):,} rows")

# print(f"\nLoading true-ratio 10% Test dataset from:\n  {TEST_DIR}")
# test_dataset = ds.dataset(TEST_DIR, format="parquet")
# test_table   = test_dataset.to_table()
# df_test_full = test_table.to_pandas()
# print(f"Test raw size: {len(df_test_full):,} rows")

# # Prepare TrainVal + predictors
# X_tv_full, y_tv_full, predictors = prepare_trainval_and_predictors(df_tv_full)
# print(f"\nTrain/Val 100x after cleaning: {len(X_tv_full):,} rows")
# print(f"Number of predictors: {len(predictors)}")
# print("Train/Val class counts:")
# print(pd.Series(y_tv_full).value_counts())
# print(pd.Series(y_tv_full).value_counts(normalize=True).mul(100))

# # Prepare Test
# X_test, y_test = prepare_test(df_test_full, predictors)
# print(f"\nTest after cleaning: {len(X_test):,} rows")
# print("Test class counts:")
# print(pd.Series(y_test).value_counts())
# print(pd.Series(y_test).value_counts(normalize=True).mul(100))

# test_pos = int((y_test == 1).sum())
# test_neg = int((y_test == 0).sum())
# print(f"\nTest positives (1): {test_pos:,}")
# print(f"Test negatives (0): {test_neg:,}")

# # Split TrainVal into pos / neg pool (starting from ~100:1)
# tv_data = X_tv_full.copy()
# tv_data["burned"] = y_tv_full

# pos_pool = tv_data[tv_data["burned"] == 1]
# neg_pool = tv_data[tv_data["burned"] == 0]

# n_pos_pool = len(pos_pool)
# n_neg_pool = len(neg_pool)
# actual_ratio_pool = n_neg_pool / max(n_pos_pool, 1)

# print("\nTrain/Val 100x pool (starting point) class counts:")
# print(tv_data["burned"].value_counts())
# print(tv_data["burned"].value_counts(normalize=True).mul(100))
# print(f"\nPositives in pool: {n_pos_pool:,}")
# print(f"Negatives in pool: {n_neg_pool:,}")
# print(f"Actual neg:pos ratio in TrainVal pool ≈ {actual_ratio_pool:.2f}:1")

# summary_rows = []

# # ============================================================
# # NEG:POS SWEEP LOOP (100, 90, ..., 10)
# # ============================================================

# for i, target_ratio in enumerate(NEG_POS_RATIOS, start=1):
#     print("\n" + "=" * 80)
#     print(f"=== Sweep {i}/{len(NEG_POS_RATIOS)} — target neg:pos ≈ {target_ratio}:1 ===")

#     # Determine how many negatives we can actually use
#     if target_ratio >= actual_ratio_pool:
#         # Can't upsample negatives, so just use all we have
#         n_neg_target = n_neg_pool
#         eff_ratio = actual_ratio_pool
#         print(f"Target ratio {target_ratio}:1 exceeds pool ratio {actual_ratio_pool:.2f}:1; "
#               f"using all negatives (eff_ratio ≈ {eff_ratio:.2f}:1).")
#         neg_subset = neg_pool
#     else:
#         n_neg_target = int(round(target_ratio * n_pos_pool))
#         n_neg_target = min(n_neg_target, n_neg_pool)
#         eff_ratio = n_neg_target / max(n_pos_pool, 1)
#         print(f"Sampling {n_neg_target:,} negatives for target ratio {target_ratio}:1 "
#               f"(eff_ratio ≈ {eff_ratio:.2f}:1).")
#         neg_subset = neg_pool.sample(
#             n=n_neg_target,
#             random_state=RANDOM_STATE + i
#         )

#     # Combine all positives + sampled negatives
#     tv_subset = pd.concat([pos_pool, neg_subset], axis=0)
#     tv_subset = tv_subset.sample(frac=1.0, random_state=RANDOM_STATE + 100 + i).reset_index(drop=True)

#     print("TrainVal subset class counts (for this neg:pos target):")
#     print(tv_subset["burned"].value_counts())
#     print(tv_subset["burned"].value_counts(normalize=True).mul(100))

#     if tv_subset["burned"].nunique() < 2:
#         print(f"[SKIP] target_ratio={target_ratio}: only one class present.")
#         continue

#     # Train vs Val split within this TrainVal subset
#     train_sub, val_sub = train_test_split(
#         tv_subset,
#         test_size=VAL_SIZE_INNER,
#         random_state=RANDOM_STATE,
#         stratify=tv_subset["burned"],
#     )

#     print("\nTrain/Val subset split sizes:")
#     print(f"  Train: {len(train_sub):,}")
#     print(f"  Val  : {len(val_sub):,}")
#     print("Train subset class counts:")
#     print(train_sub["burned"].value_counts())
#     print("Val subset class counts:")
#     print(val_sub["burned"].value_counts())

#     X_train = train_sub[predictors].copy()
#     y_train = train_sub["burned"].astype("uint8")
#     X_val   = val_sub[predictors].copy()
#     y_val   = val_sub["burned"].astype("uint8")

#     n_pos_train = int((y_train == 1).sum())
#     n_neg_train = int((y_train == 0).sum())
#     n_pos_val   = int((y_val   == 1).sum())
#     n_neg_val   = int((y_val   == 0).sum())

#     print(f"\nTrain subset positives: {n_pos_train:,}, negatives: {n_neg_train:,}")
#     print(f"Val subset positives  : {n_pos_val:,}, negatives: {n_neg_val:,}")

#     # XGBoost DMatrix
#     dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
#     dval   = xgb.DMatrix(X_val,   label=y_val,   enable_categorical=True)
#     dtest  = xgb.DMatrix(X_test,  label=y_test,  enable_categorical=True)

#     # ----------------- TRAIN XGBOOST MODEL -----------------
#     print("\nTraining XGBoost with focal loss...")
#     evals = [(dtrain, "train"), (dval, "validation")]
#     evals_result = {}

#     booster = xgb.train(
#         tuned_params,
#         dtrain,
#         num_boost_round=NUM_BOOST_ROUNDS,
#         evals=evals,
#         obj=focal_loss_xgb,
#         evals_result=evals_result,
#         verbose_eval=50,
#     )

#     # ----------------- IoU-at-best-threshold LEARNING CURVES -----------------
#     train_iou_curve = []
#     val_iou_curve   = []
#     thr_curve       = []

#     print("\nComputing IoU-at-best-threshold curves across iterations...")
#     for it in range(1, NUM_BOOST_ROUNDS + 1):
#         val_margin   = booster.predict(dval,   iteration_range=(0, it))
#         train_margin = booster.predict(dtrain, iteration_range=(0, it))

#         val_proba   = sigmoid(val_margin)
#         train_proba = sigmoid(train_margin)

#         # Find best threshold on VAL for this iteration (max F1)
#         prec_val, rec_val, thr_val = precision_recall_curve(y_val, val_proba)
#         prec_ = prec_val[:-1]
#         rec_  = rec_val[:-1]
#         f1_vals = 2 * prec_ * rec_ / (prec_ + rec_ + 1e-12)
#         best_idx_it = int(np.argmax(f1_vals))
#         best_thr_it = float(thr_val[best_idx_it])

#         thr_curve.append(best_thr_it)

#         # IoU on VAL at its best threshold for this iteration
#         y_val_hat_it = (val_proba >= best_thr_it).astype("uint8")
#         val_iou_it = jaccard_score(y_val, y_val_hat_it, average="binary", zero_division=0)
#         val_iou_curve.append(val_iou_it)

#         # IoU on TRAIN at same threshold
#         y_train_hat_it = (train_proba >= best_thr_it).astype("uint8")
#         train_iou_it = jaccard_score(y_train, y_train_hat_it, average="binary", zero_division=0)
#         train_iou_curve.append(train_iou_it)

#     train_iou_curve = np.array(train_iou_curve)
#     val_iou_curve   = np.array(val_iou_curve)
#     thr_curve       = np.array(thr_curve)

#     # Best iteration = argmax val IoU
#     best_iter_idx = int(np.argmax(val_iou_curve))
#     best_iter     = int(best_iter_idx + 1)
#     best_thr      = float(thr_curve[best_iter_idx])
#     best_val_iou  = float(val_iou_curve[best_iter_idx])

#     print(f"\nBest iteration (Val IoU) : {best_iter}")
#     print(f"Val IoU at best_iter     : {best_val_iou:.3f}")
#     print(f"Threshold at best_iter   : {best_thr:.3f}")

#     # ----------------- PLOT IoU CURVES -----------------
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_iou_curve, label="Train IoU (best thr per iter)")
#     plt.plot(val_iou_curve,   label="Val IoU (best thr per iter)")
#     plt.axvline(best_iter_idx, linestyle="--", label=f"best_iter={best_iter}")
#     plt.xlabel("Boosting Rounds")
#     plt.ylabel("IoU (Jaccard)")
#     plt.title(
#         f"XGB Focal — Train vs Val IoU (neg:pos target {target_ratio}:1)\n"
#         f"Effective neg:pos in subset ≈ {eff_ratio:.2f}:1"
#     )
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     iou_fig_out = os.path.join(
#         FIGS_DIR,
#         f"iou_curve_bestthr_xgb_negpos{int(target_ratio):03d}.png"
#     )
#     plt.savefig(iou_fig_out, dpi=150)
#     plt.close()
#     print(f"Saved IoU learning curve: {iou_fig_out}")

#     # ----------------- FINAL TEST METRICS -----------------
#     print("\nEvaluating on fixed true-ratio Test set...")

#     test_margin = booster.predict(dtest, iteration_range=(0, best_iter))
#     y_test_proba = sigmoid(test_margin)
#     y_test_hat   = (y_test_proba >= best_thr).astype("uint8")

#     test_iou  = jaccard_score(y_test, y_test_hat, average="binary", zero_division=0)
#     test_prec = precision_score(y_test, y_test_hat, zero_division=0)
#     test_rec  = recall_score(y_test, y_test_hat, zero_division=0)
#     test_f1   = f1_score(y_test, y_test_hat, zero_division=0)

#     fpr, tpr, _ = roc_curve(y_test, y_test_proba)
#     test_auc_roc = roc_auc_score(y_test, y_test_proba)

#     prec_curve_test, rec_curve_test, _ = precision_recall_curve(y_test, y_test_proba)
#     test_auc_pr = average_precision_score(y_test, y_test_proba)

#     print("\n==== FINAL TEST METRICS (XGB focal, fixed true-ratio test) ====")
#     print(f"Target neg:pos (TrainVal) : {target_ratio}:1")
#     print(f"Eff neg:pos in TrainVal   : {eff_ratio:.2f}:1")
#     print(f"Best iteration (Val IoU)  : {best_iter}")
#     print(f"Threshold (Val best F1)   : {best_thr:.3f}")
#     print(f"IoU (Jaccard)             : {test_iou:.4f}")
#     print(f"Precision                 : {test_prec:.4f}")
#     print(f"Recall                    : {test_rec:.4f}")
#     print(f"F1 Score                  : {test_f1:.4f}")
#     print(f"ROC AUC                   : {test_auc_roc:.4f}")
#     print(f"PR AUC (Avg Precision)    : {test_auc_pr:.4f}")

#     # ---- ROC curve ----
#     plt.figure(figsize=(6, 5))
#     plt.plot(fpr, tpr, label=f"ROC (AUC = {test_auc_roc:.3f})")
#     plt.plot([0, 1], [0, 1], linestyle="--")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title(
#         f"XGB Focal — ROC (neg:pos target {target_ratio}:1)\n"
#         f"Eff neg:pos ≈ {eff_ratio:.2f}:1 in TrainVal subset"
#     )
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     roc_fig_out = os.path.join(
#         FIGS_DIR,
#         f"roc_curve_xgb_negpos{int(target_ratio):03d}.png"
#     )
#     plt.savefig(roc_fig_out, dpi=150)
#     plt.close()
#     print(f"Saved ROC curve: {roc_fig_out}")

#     # ---- PR curve ----
#     plt.figure(figsize=(6, 5))
#     plt.plot(rec_curve_test, prec_curve_test, label=f"PR (AUC = {test_auc_pr:.3f})")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title(
#         f"XGB Focal — PR Curve (neg:pos target {target_ratio}:1)\n"
#         f"Eff neg:pos ≈ {eff_ratio:.2f}:1 in TrainVal subset"
#     )
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     pr_fig_out = os.path.join(
#         FIGS_DIR,
#         f"pr_curve_xgb_negpos{int(target_ratio):03d}.png"
#     )
#     plt.savefig(pr_fig_out, dpi=150)
#     plt.close()
#     print(f"Saved PR curve: {pr_fig_out}")

#     # ----------------- FEATURE IMPORTANCE -----------------
#     fmap = booster.get_score(importance_type="gain")
#     feat_names = np.array(predictors)
#     gain_imp = np.array(
#         [fmap.get(f"f{i}", 0.0) for i in range(len(feat_names))],
#         dtype=float,
#     )
#     gain_imp = gain_imp / (gain_imp.sum() + 1e-12)
#     order = np.argsort(gain_imp)[::-1][:TOP_N_IMPORT]

#     plt.figure(figsize=(9, max(5, 0.28 * len(order))))
#     plt.barh(feat_names[order][::-1], gain_imp[order][::-1])
#     plt.xlabel("Relative Gain Importance")
#     plt.title(
#         f"XGB Focal — Feature Importance (Top {len(order)})\n"
#         f"Neg:pos target {target_ratio}:1 (eff ≈ {eff_ratio:.2f}:1)"
#     )
#     plt.tight_layout()
#     fi_fig_out = os.path.join(
#         FIGS_DIR,
#         f"feature_importance_xgb_negpos{int(target_ratio):03d}.png"
#     )
#     plt.savefig(fi_fig_out, dpi=150)
#     plt.close()
#     print(f"Saved feature importance plot: {fi_fig_out}")

#     # ----------------- SAVE MODEL -----------------
#     model_path = os.path.join(
#         MODELS_DIR,
#         f"xgb_focal_model_negpos{int(target_ratio):03d}.pkl"
#     )
#     joblib.dump(booster, model_path)
#     print(f"Saved model for neg:pos={target_ratio}:1 to: {model_path}")

#     # ----------------- APPEND SUMMARY ROW -----------------
#     summary_rows.append(
#         dict(
#             neg_pos_target      = target_ratio,
#             eff_neg_pos_ratio   = round(eff_ratio, 3),
#             n_pos_pool          = n_pos_pool,
#             n_neg_pool          = n_neg_pool,
#             n_pos_train         = n_pos_train,
#             n_neg_train         = n_neg_train,
#             n_pos_val           = n_pos_val,
#             n_neg_val           = n_neg_val,
#             n_pos_test          = test_pos,
#             n_neg_test          = test_neg,
#             focal_alpha         = FOCAL_ALPHA,
#             focal_gamma         = FOCAL_GAMMA,
#             threshold           = round(best_thr, 3),
#             val_iou_best        = round(best_val_iou, 4),
#             test_iou            = round(test_iou, 4),
#             test_precision      = round(test_prec, 4),
#             test_recall         = round(test_rec, 4),
#             test_f1             = round(test_f1, 4),
#             test_auc_roc        = round(test_auc_roc, 4),
#             test_auc_pr         = round(test_auc_pr, 4),
#             best_iteration      = int(best_iter),
#         )
#     )

# # ============================================================
# # SAVE SUMMARY CSV
# # ============================================================

# if summary_rows:
#     summary_df = pd.DataFrame(summary_rows)
#     summary_csv = os.path.join(
#         OUT_DIR,
#         "xgb_focal_trueTest_train100_negpos_sweep_metrics_auc_thresh.csv"
#     )
#     summary_df.to_csv(summary_csv, index=False)
#     print(f"\nSaved XGB focal neg:pos sweep summary to:\n  {summary_csv}")
# else:
#     print("\nNo neg:pos runs were executed; summary not saved.")

# print("\n✅ Done. XGBoost focal neg:pos sweeps (100→10) complete.")



# import os
# import json
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# import pyarrow.dataset as ds
# import matplotlib.pyplot as plt
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     precision_recall_curve,
#     jaccard_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_curve,
#     roc_auc_score,
#     average_precision_score,
# )

# # ============================================================
# # CONFIG
# # ============================================================

# RANDOM_STATE = 42

# # These are the datasets you just created
# TRAINVAL_DIR = "/explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_trainval_100x"
# TEST_DIR     = "/explore/nobackup/people/spotter5/clelland_fire_ml/parquet_cems_test_true_10pct"

# # Where tuned XGB focal params were saved from the tuning script
# OUT_ROOT_OLD  = "/explore/nobackup/people/spotter5/clelland_fire_ml/ml_training/neg_ratio_experiments_globaltest"
# PARAMS_DIR    = os.path.join(OUT_ROOT_OLD, "option4_focal_loss_10x_negative_auc_thresh")
# BEST_PARAMS_JSON = os.path.join(PARAMS_DIR, "tuned_xgb_focal_params.json")

# # New output directory for this experiment
# OUT_ROOT = OUT_ROOT_OLD
# OUT_DIR  = os.path.join(OUT_ROOT, "xgb_focal_trueTest_train100_negpos_sweep")
# os.makedirs(OUT_DIR, exist_ok=True)

# MODELS_DIR = os.path.join(OUT_DIR, "models")
# FIGS_DIR   = os.path.join(OUT_DIR, "figures")
# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(FIGS_DIR, exist_ok=True)

# TOP_N_IMPORT = 30

# # Ratios we want to explore (neg:pos in Train/Val subset)
# NEG_POS_RATIOS = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

# # Same semantics as before: 10% test overall, 20% val overall => ~22.22% of TrainVal
# TEST_SIZE_GLOBAL = 0.10   # informational now (already applied upstream)
# VAL_SIZE_OVERALL = 0.20
# VAL_SIZE_INNER   = VAL_SIZE_OVERALL / (1.0 - TEST_SIZE_GLOBAL)

# # ============================================================
# # Helpers
# # ============================================================

# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-x))


# def focal_loss_xgb(preds, dtrain):
#     """
#     Custom focal loss for XGBoost using raw logits.

#     preds: raw scores (logits) from the model
#     dtrain: xgb.DMatrix
#     """
#     y = dtrain.get_label()
#     p = sigmoid(preds)
#     p = np.clip(p, 1e-7, 1 - 1e-7)
#     a, g = FOCAL_ALPHA, FOCAL_GAMMA

#     # Focal loss grad for positives and negatives
#     grad_pos = a * ((1 - p) ** g) * (g * (-np.log(p)) * (1 - p) - 1) * (p * (1 - p))
#     grad_neg = (1 - a) * (p ** g) * (g * (-np.log(1 - p)) * p + 1) * (p * (1 - p))
#     grad = np.where(y > 0.5, grad_pos, grad_neg)

#     # Approximate Hessian by logistic Hessian
#     hess = p * (1 - p)
#     return grad, hess


# def prepare_trainval_and_predictors(df: pd.DataFrame):
#     """
#     From the Train/Val 100x dataframe:

#       - Ensure `burned` exists (0/1 label; recompute from fraction if needed).
#       - Define predictors (drop fraction/burned/coords/etc).
#       - Handle `b1` categorical.
#       - Coerce non-numeric predictors to numeric.
#       - Drop rows with NaNs in predictors (and NaN b1 if categorical).
#     """
#     df = df.copy()

#     if "fraction" in df.columns:
#         df["fraction"] = df["fraction"].astype("float32").clip(0, 1)

#     if "burned" not in df.columns:
#         if "fraction" not in df.columns:
#             raise ValueError("Need either 'burned' or 'fraction' in Train/Val dataset.")
#         df["burned"] = (df["fraction"] > 0.5).astype("uint8")

#     df = df.replace([np.inf, -np.inf], np.nan)

#     drop_cols = {"fraction", "burned", "bin", "year", "month", "latitude", "longitude"}
#     predictors = [c for c in df.columns if c not in drop_cols]

#     X = df[predictors].copy()
#     y = df["burned"].astype("uint8")

#     # Treat land cover as categorical if present
#     if "b1" in X.columns and not pd.api.types.is_categorical_dtype(X["b1"]):
#         X["b1"] = X["b1"].astype("category")
#         print("\nTreating 'b1' as pandas 'category' in Train/Val.")

#     # Coerce non-numeric predictors (except categorical b1) to numeric
#     coerced = 0
#     for c in X.columns:
#         if c == "b1" and pd.api.types.is_categorical_dtype(X[c]):
#             continue
#         if not np.issubdtype(X[c].dtype, np.number):
#             X[c] = pd.to_numeric(X[c], errors="coerce")
#             coerced += 1

#     if coerced:
#         num_cols = [
#             c for c in X.columns
#             if not (c == "b1" and pd.api.types.is_categorical_dtype(X["b1"]))
#         ]
#         mask = X[num_cols].notna().all(axis=1)
#         if "b1" in X.columns and pd.api.types.is_categorical_dtype(X["b1"]):
#             mask &= X["b1"].notna()

#         before = len(X)
#         X = X.loc[mask].copy()
#         y = y.loc[mask].copy()
#         print(f"Dropped {before - len(X):,} Train/Val rows with NaNs after coercion.")

#     return X, y, predictors


# def prepare_test(df: pd.DataFrame, predictors):
#     """
#     Prepare X_test, y_test using the same predictor set as Train/Val.
#     """
#     df = df.copy()

#     if "fraction" in df.columns:
#         df["fraction"] = df["fraction"].astype("float32").clip(0, 1)

#     if "burned" not in df.columns:
#         if "fraction" not in df.columns:
#             raise ValueError("Need either 'burned' or 'fraction' in Test dataset.")
#         df["burned"] = (df["fraction"] > 0.5).astype("uint8")

#     df = df.replace([np.inf, -np.inf], np.nan)

#     missing_preds = [c for c in predictors if c not in df.columns]
#     if missing_preds:
#         raise ValueError(f"Test dataset is missing predictor columns: {missing_preds}")

#     X = df[predictors].copy()
#     y = df["burned"].astype("uint8")

#     # Handle categorical b1 identical to Train/Val
#     if "b1" in X.columns and not pd.api.types.is_categorical_dtype(X["b1"]):
#         X["b1"] = X["b1"].astype("category")
#         print("\nTreating 'b1' as pandas 'category' in Test.")

#     # Coerce any non-numeric predictor (except categorical b1) to numeric
#     coerced = 0
#     for c in X.columns:
#         if c == "b1" and pd.api.types.is_categorical_dtype(X["b1"]):
#             continue
#         if not np.issubdtype(X[c].dtype, np.number):
#             X[c] = pd.to_numeric(X[c], errors="coerce")
#             coerced += 1

#     if coerced:
#         num_cols = [
#             c for c in X.columns
#             if not (c == "b1" and pd.api.types.is_categorical_dtype(X["b1"]))
#         ]
#         mask = X[num_cols].notna().all(axis=1)
#         if "b1" in X.columns and pd.api.types.is_categorical_dtype(X["b1"]):
#             mask &= X["b1"].notna()

#         before = len(X)
#         X = X.loc[mask].copy()
#         y = y.loc[mask].copy()
#         print(f"Dropped {before - len(X):,} Test rows with NaNs after coercion.")

#     return X, y


# # ============================================================
# # LOAD TUNED PARAMS
# # ============================================================

# print(f"Loading tuned XGBoost focal params from:\n  {BEST_PARAMS_JSON}")
# with open(BEST_PARAMS_JSON, "r") as f:
#     tuned_params = json.load(f)

# # Extract focal alpha/gamma and num_boost_rounds
# FOCAL_ALPHA       = tuned_params.get("focal_alpha", 0.25)
# FOCAL_GAMMA       = tuned_params.get("focal_gamma", 2.0)
# NUM_BOOST_ROUNDS = tuned_params.get("num_boost_rounds", 600)

# # Remove non-XGB params
# for k in ["focal_alpha", "focal_gamma", "num_boost_rounds"]:
#     tuned_params.pop(k, None)

# print("\nUsing XGBoost params:")
# print(tuned_params)
# print(f"NUM_BOOST_ROUNDS = {NUM_BOOST_ROUNDS}")
# print(f"FOCAL_ALPHA      = {FOCAL_ALPHA}")
# print(f"FOCAL_GAMMA      = {FOCAL_GAMMA}")

# # ============================================================
# # LOAD TRAINVAL 100x + TEST TRUE-RATIO
# # ============================================================

# print(f"\nLoading Train/Val 100x dataset from:\n  {TRAINVAL_DIR}")
# tv_dataset = ds.dataset(TRAINVAL_DIR, format="parquet")
# tv_table   = tv_dataset.to_table()
# df_tv_full = tv_table.to_pandas()
# print(f"Train/Val 100x raw size: {len(df_tv_full):,} rows")

# print(f"\nLoading true-ratio 10% Test dataset from:\n  {TEST_DIR}")
# test_dataset = ds.dataset(TEST_DIR, format="parquet")
# test_table   = test_dataset.to_table()
# df_test_full = test_table.to_pandas()
# print(f"Test raw size: {len(df_test_full):,} rows")

# # Prepare TrainVal + predictors
# X_tv_full, y_tv_full, predictors = prepare_trainval_and_predictors(df_tv_full)
# print(f"\nTrain/Val 100x after cleaning: {len(X_tv_full):,} rows")
# print(f"Number of predictors: {len(predictors)}")
# print("Train/Val class counts:")
# print(pd.Series(y_tv_full).value_counts())
# print(pd.Series(y_tv_full).value_counts(normalize=True).mul(100))

# # Prepare Test
# X_test, y_test = prepare_test(df_test_full, predictors)
# print(f"\nTest after cleaning: {len(X_test):,} rows")
# print("Test class counts:")
# print(pd.Series(y_test).value_counts())
# print(pd.Series(y_test).value_counts(normalize=True).mul(100))

# test_pos = int((y_test == 1).sum())
# test_neg = int((y_test == 0).sum())
# print(f"\nTest positives (1): {test_pos:,}")
# print(f"Test negatives (0): {test_neg:,}")

# # Split TrainVal into pos / neg pool (starting from ~100:1)
# tv_data = X_tv_full.copy()
# tv_data["burned"] = y_tv_full

# pos_pool = tv_data[tv_data["burned"] == 1]
# neg_pool = tv_data[tv_data["burned"] == 0]

# n_pos_pool = len(pos_pool)
# n_neg_pool = len(neg_pool)
# actual_ratio_pool = n_neg_pool / max(n_pos_pool, 1)

# print("\nTrain/Val 100x pool (starting point) class counts:")
# print(tv_data["burned"].value_counts())
# print(tv_data["burned"].value_counts(normalize=True).mul(100))
# print(f"\nPositives in pool: {n_pos_pool:,}")
# print(f"Negatives in pool: {n_neg_pool:,}")
# print(f"Actual neg:pos ratio in TrainVal pool ≈ {actual_ratio_pool:.2f}:1")

# summary_rows = []

# # ============================================================
# # NEG:POS SWEEP LOOP (100, 90, ..., 10)
# # ============================================================

# for i, target_ratio in enumerate(NEG_POS_RATIOS, start=1):
#     print("\n" + "=" * 80)
#     print(f"=== Sweep {i}/{len(NEG_POS_RATIOS)} — target neg:pos ≈ {target_ratio}:1 ===")

#     # Determine how many negatives we can actually use
#     if target_ratio >= actual_ratio_pool:
#         # Can't upsample negatives, so just use all we have
#         n_neg_target = n_neg_pool
#         eff_ratio = actual_ratio_pool
#         print(f"Target ratio {target_ratio}:1 exceeds pool ratio {actual_ratio_pool:.2f}:1; "
#               f"using all negatives (eff_ratio ≈ {eff_ratio:.2f}:1).")
#         neg_subset = neg_pool
#     else:
#         n_neg_target = int(round(target_ratio * n_pos_pool))
#         n_neg_target = min(n_neg_target, n_neg_pool)
#         eff_ratio = n_neg_target / max(n_pos_pool, 1)
#         print(f"Sampling {n_neg_target:,} negatives for target ratio {target_ratio}:1 "
#               f"(eff_ratio ≈ {eff_ratio:.2f}:1).")
#         neg_subset = neg_pool.sample(
#             n=n_neg_target,
#             random_state=RANDOM_STATE + i
#         )

#     # Combine all positives + sampled negatives
#     tv_subset = pd.concat([pos_pool, neg_subset], axis=0)
#     tv_subset = tv_subset.sample(frac=1.0, random_state=RANDOM_STATE + 100 + i).reset_index(drop=True)

#     print("TrainVal subset class counts (for this neg:pos target):")
#     print(tv_subset["burned"].value_counts())
#     print(tv_subset["burned"].value_counts(normalize=True).mul(100))

#     if tv_subset["burned"].nunique() < 2:
#         print(f"[SKIP] target_ratio={target_ratio}: only one class present.")
#         continue

#     # Train vs Val split within this TrainVal subset
#     train_sub, val_sub = train_test_split(
#         tv_subset,
#         test_size=VAL_SIZE_INNER,
#         random_state=RANDOM_STATE,
#         stratify=tv_subset["burned"],
#     )

#     print("\nTrain/Val subset split sizes:")
#     print(f"  Train: {len(train_sub):,}")
#     print(f"  Val  : {len(val_sub):,}")
#     print("Train subset class counts:")
#     print(train_sub["burned"].value_counts())
#     print("Val subset class counts:")
#     print(val_sub["burned"].value_counts())

#     X_train = train_sub[predictors].copy()
#     y_train = train_sub["burned"].astype("uint8")
#     X_val   = val_sub[predictors].copy()
#     y_val   = val_sub["burned"].astype("uint8")

#     n_pos_train = int((y_train == 1).sum())
#     n_neg_train = int((y_train == 0).sum())
#     n_pos_val   = int((y_val   == 1).sum())
#     n_neg_val   = int((y_val   == 0).sum())

#     print(f"\nTrain subset positives: {n_pos_train:,}, negatives: {n_neg_train:,}")
#     print(f"Val subset positives  : {n_pos_val:,}, negatives: {n_neg_val:,}")

#     # XGBoost DMatrix
#     dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
#     dval   = xgb.DMatrix(X_val,   label=y_val,   enable_categorical=True)
#     dtest  = xgb.DMatrix(X_test,  label=y_test,  enable_categorical=True)

#     # ----------------- TRAIN XGBOOST MODEL -----------------
#     print("\nTraining XGBoost with focal loss...")
#     evals = [(dtrain, "train"), (dval, "validation")]
#     evals_result = {}

#     booster = xgb.train(
#         tuned_params,
#         dtrain,
#         num_boost_round=NUM_BOOST_ROUNDS,
#         evals=evals,
#         obj=focal_loss_xgb,
#         evals_result=evals_result,
#         verbose_eval=50,
#     )

#     # ----------------- IoU-at-best-threshold LEARNING CURVES -----------------
#     train_iou_curve = []
#     val_iou_curve   = []
#     thr_curve       = []

#     print("\nComputing IoU-at-best-threshold curves across iterations...")
#     for it in range(1, NUM_BOOST_ROUNDS + 1):
#         val_margin   = booster.predict(dval,   iteration_range=(0, it))
#         train_margin = booster.predict(dtrain, iteration_range=(0, it))

#         val_proba   = sigmoid(val_margin)
#         train_proba = sigmoid(train_margin)

#         # Find best threshold on VAL for this iteration (max F1)
#         prec_val, rec_val, thr_val = precision_recall_curve(y_val, val_proba)
#         prec_ = prec_val[:-1]
#         rec_  = rec_val[:-1]
#         f1_vals = 2 * prec_ * rec_ / (prec_ + rec_ + 1e-12)
#         best_idx_it = int(np.argmax(f1_vals))
#         best_thr_it = float(thr_val[best_idx_it])

#         thr_curve.append(best_thr_it)

#         # IoU on VAL at its best threshold for this iteration
#         y_val_hat_it = (val_proba >= best_thr_it).astype("uint8")
#         val_iou_it = jaccard_score(y_val, y_val_hat_it, average="binary", zero_division=0)
#         val_iou_curve.append(val_iou_it)

#         # IoU on TRAIN at same threshold
#         y_train_hat_it = (train_proba >= best_thr_it).astype("uint8")
#         train_iou_it = jaccard_score(y_train, y_train_hat_it, average="binary", zero_division=0)
#         train_iou_curve.append(train_iou_it)

#     train_iou_curve = np.array(train_iou_curve)
#     val_iou_curve   = np.array(val_iou_curve)
#     thr_curve       = np.array(thr_curve)

#     # Best iteration = argmax val IoU
#     best_iter_idx = int(np.argmax(val_iou_curve))
#     best_iter     = int(best_iter_idx + 1)
#     best_thr      = float(thr_curve[best_iter_idx])
#     best_val_iou  = float(val_iou_curve[best_iter_idx])

#     print(f"\nBest iteration (Val IoU) : {best_iter}")
#     print(f"Val IoU at best_iter      : {best_val_iou:.3f}")
#     print(f"Threshold at best_iter    : {best_thr:.3f}")

#     # ----------------- PLOT IoU CURVES -----------------
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_iou_curve, label="Train IoU (best thr per iter)")
#     plt.plot(val_iou_curve,   label="Val IoU (best thr per iter)")
#     plt.axvline(best_iter_idx, linestyle="--", label=f"best_iter={best_iter}")
#     plt.xlabel("Boosting Rounds")
#     plt.ylabel("IoU (Jaccard)")
#     plt.title(
#         f"XGB Focal — Train vs Val IoU (neg:pos target {target_ratio}:1)\n"
#         f"Effective neg:pos in subset ≈ {eff_ratio:.2f}:1"
#     )
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     iou_fig_out = os.path.join(
#         FIGS_DIR,
#         f"iou_curve_bestthr_xgb_negpos{int(target_ratio):03d}.png"
#     )
#     plt.savefig(iou_fig_out, dpi=150)
#     plt.close()
#     print(f"Saved IoU learning curve: {iou_fig_out}")

#     # ----------------- FINAL TEST METRICS -----------------
#     print("\nEvaluating on fixed true-ratio Test set...")

#     test_margin = booster.predict(dtest, iteration_range=(0, best_iter))
#     y_test_proba = sigmoid(test_margin)
#     y_test_hat   = (y_test_proba >= best_thr).astype("uint8")

#     test_iou  = jaccard_score(y_test, y_test_hat, average="binary", zero_division=0)
#     test_prec = precision_score(y_test, y_test_hat, zero_division=0)
#     test_rec  = recall_score(y_test, y_test_hat, zero_division=0)
#     test_f1   = f1_score(y_test, y_test_hat, zero_division=0)

#     fpr, tpr, _ = roc_curve(y_test, y_test_proba)
#     test_auc_roc = roc_auc_score(y_test, y_test_proba)

#     prec_curve_test, rec_curve_test, _ = precision_recall_curve(y_test, y_test_proba)
#     test_auc_pr = average_precision_score(y_test, y_test_proba)

#     print("\n==== FINAL TEST METRICS (XGB focal, fixed true-ratio test) ====")
#     print(f"Target neg:pos (TrainVal) : {target_ratio}:1")
#     print(f"Eff neg:pos in TrainVal   : {eff_ratio:.2f}:1")
#     print(f"Best iteration (Val IoU)  : {best_iter}")
#     print(f"Threshold (Val best F1)   : {best_thr:.3f}")
#     print(f"IoU (Jaccard)             : {test_iou:.4f}")
#     print(f"Precision                 : {test_prec:.4f}")
#     print(f"Recall                    : {test_rec:.4f}")
#     print(f"F1 Score                  : {test_f1:.4f}")
#     print(f"ROC AUC                   : {test_auc_roc:.4f}")
#     print(f"PR AUC (Avg Precision)    : {test_auc_pr:.4f}")

#     # ---- ROC curve ----
#     plt.figure(figsize=(6, 5))
#     plt.plot(fpr, tpr, label=f"ROC (AUC = {test_auc_roc:.3f})")
#     plt.plot([0, 1], [0, 1], linestyle="--")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title(
#         f"XGB Focal — ROC (neg:pos target {target_ratio}:1)\n"
#         f"Eff neg:pos ≈ {eff_ratio:.2f}:1 in TrainVal subset"
#     )
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     roc_fig_out = os.path.join(
#         FIGS_DIR,
#         f"roc_curve_xgb_negpos{int(target_ratio):03d}.png"
#     )
#     plt.savefig(roc_fig_out, dpi=150)
#     plt.close()
#     print(f"Saved ROC curve: {roc_fig_out}")

#     # ---- PR curve ----
#     plt.figure(figsize=(6, 5))
#     plt.plot(rec_curve_test, prec_curve_test, label=f"PR (AUC = {test_auc_pr:.3f})")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title(
#         f"XGB Focal — PR Curve (neg:pos target {target_ratio}:1)\n"
#         f"Eff neg:pos ≈ {eff_ratio:.2f}:1 in TrainVal subset"
#     )
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     pr_fig_out = os.path.join(
#         FIGS_DIR,
#         f"pr_curve_xgb_negpos{int(target_ratio):03d}.png"
#     )
#     plt.savefig(pr_fig_out, dpi=150)
#     plt.close()
#     print(f"Saved PR curve: {pr_fig_out}")

#     # ----------------- FEATURE IMPORTANCE (FIXED) -----------------
#     fmap = booster.get_score(importance_type="gain")
#     feat_names = np.array(predictors)

#     # Use the actual column name from feat_names to look up in fmap
#     gain_imp = np.array(
#         [fmap.get(name, 0.0) for name in feat_names],
#         dtype=float,
#     )
#     gain_imp = gain_imp / (gain_imp.sum() + 1e-12)
#     order = np.argsort(gain_imp)[::-1][:TOP_N_IMPORT]

#     plt.figure(figsize=(9, max(5, 0.28 * len(order))))
#     plt.barh(feat_names[order][::-1], gain_imp[order][::-1])
#     plt.xlabel("Relative Gain Importance")
#     plt.title(
#         f"XGB Focal — Feature Importance (Top {len(order)})\n"
#         f"Neg:pos target {target_ratio}:1 (eff ≈ {eff_ratio:.2f}:1)"
#     )
#     plt.tight_layout()
#     fi_fig_out = os.path.join(
#         FIGS_DIR,
#         f"feature_importance_xgb_negpos{int(target_ratio):03d}.png"
#     )
#     plt.savefig(fi_fig_out, dpi=150)
#     plt.close()
#     print(f"Saved feature importance plot: {fi_fig_out}")

#     # ----------------- SAVE MODEL -----------------
#     model_path = os.path.join(
#         MODELS_DIR,
#         f"xgb_focal_model_negpos{int(target_ratio):03d}.pkl"
#     )
#     joblib.dump(booster, model_path)
#     print(f"Saved model for neg:pos={target_ratio}:1 to: {model_path}")

#     # ----------------- APPEND SUMMARY ROW -----------------
#     summary_rows.append(
#         dict(
#             neg_pos_target      = target_ratio,
#             eff_neg_pos_ratio   = round(eff_ratio, 3),
#             n_pos_pool          = n_pos_pool,
#             n_neg_pool          = n_neg_pool,
#             n_pos_train         = n_pos_train,
#             n_neg_train         = n_neg_train,
#             n_pos_val           = n_pos_val,
#             n_neg_val           = n_neg_val,
#             n_pos_test          = test_pos,
#             n_neg_test          = test_neg,
#             focal_alpha         = FOCAL_ALPHA,
#             focal_gamma         = FOCAL_GAMMA,
#             threshold           = round(best_thr, 3),
#             val_iou_best        = round(best_val_iou, 4),
#             test_iou            = round(test_iou, 4),
#             test_precision      = round(test_prec, 4),
#             test_recall         = round(test_rec, 4),
#             test_f1             = round(test_f1, 4),
#             test_auc_roc        = round(test_auc_roc, 4),
#             test_auc_pr         = round(test_auc_pr, 4),
#             best_iteration      = int(best_iter),
#         )
#     )

# # ============================================================
# # SAVE SUMMARY CSV
# # ============================================================

# if summary_rows:
#     summary_df = pd.DataFrame(summary_rows)
#     summary_csv = os.path.join(
#         OUT_DIR,
#         "xgb_focal_trueTest_train100_negpos_sweep_metrics_auc_thresh.csv"
#     )
#     summary_df.to_csv(summary_csv, index=False)
#     print(f"\nSaved XGB focal neg:pos sweep summary to:\n  {summary_csv}")
# else:
#     print("\nNo neg:pos runs were executed; summary not saved.")

# print("\n✅ Done. XGBoost focal neg:pos sweeps (100→10) complete.")



#----------------new try
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

# NEW Output directory
OUT_ROOT = OUT_ROOT_OLD
OUT_DIR  = os.path.join(OUT_ROOT, "xgb_focal_trueTest_train100_negpos_sweep_PRAUC_FIX")
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

def prepare_trainval_and_predictors(df: pd.DataFrame):
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

# Prepare Data
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

