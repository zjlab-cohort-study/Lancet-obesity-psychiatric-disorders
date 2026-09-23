"""Step 01 — 80/20 train-test split + missing-data imputation fitted within the training set.

Outputs:
  data/train_imputed.parquet, data/test_imputed.parquet  (imputed + standardized design + outcomes)
  data/feature_levels.json                               (feature lists for model 1/2/3)
  results/imputation_summary.csv                         (missingness before/after)
  data/imputer_scaler.pkl                                (fitted transforms)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

from common import (RAW_CSV, DATA_DIR, RESULTS_DIR, OUTCOMES, RANDOM_STATE, TEST_SIZE,
                    IMPUTE_BLOCK, BIOMARKERS, CONT_COLS, build_design, level_columns, save_json)

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

df = pd.read_csv(RAW_CSV)
log(f"loaded {df.shape}")

keep = (["id", "obesity"] + ["sex", "age", "smoking", "alcohol", "physical_activity",
       "sleep_duration", "family_income", "TDI", "race", "college", "employment"]
       + [f"{o}_label" for o in OUTCOMES] + [f"{o}_followup_time" for o in OUTCOMES] + BIOMARKERS)
df = df[keep].copy()
log(f"kept columns: {df.shape}")

# ---- stratified 80/20 split on the joint outcome pattern (same split for all 3 outcomes)
strata = df[[f"{o}_label" for o in OUTCOMES]].astype(str).agg("-".join, axis=1)
train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strata)
log(f"train {len(train)}, test {len(test)}")

# ---- missingness snapshot (before imputation)
miss_before = train[IMPUTE_BLOCK].isna().mean().rename("train_missing_frac")
miss_rows_before = pd.DataFrame({
    "n_missing_cells_train": [int(train[IMPUTE_BLOCK].isna().sum().sum())],
    "frac_cells_missing_train": [train[IMPUTE_BLOCK].isna().sum().sum() / (len(train) * len(IMPUTE_BLOCK))],
    "n_rows_any_missing_train": [int(train[IMPUTE_BLOCK].isna().any(axis=1).sum())],
})

# ---- MICE-style iterative imputation, FIT ON TRAIN ONLY, applied to train & test
log("fitting IterativeImputer (BayesianRidge, max_iter=15) on training set ...")
imp = IterativeImputer(random_state=RANDOM_STATE, max_iter=15, skip_complete=True,
                       sample_posterior=False, min_value=-np.inf, max_value=np.inf)
imp.fit(train[IMPUTE_BLOCK])
train_imp = train.copy()
test_imp = test.copy()
train_imp[IMPUTE_BLOCK] = imp.transform(train[IMPUTE_BLOCK])
test_imp[IMPUTE_BLOCK] = imp.transform(test[IMPUTE_BLOCK])
assert not train_imp[IMPUTE_BLOCK].isna().any().any() and not test_imp[IMPUTE_BLOCK].isna().any().any()
log("imputation done")

# ---- design matrices (one-hot for multi-level categoricals; reference = first level)
Xtr = build_design(train_imp.reset_index(drop=True))
Xte = build_design(test_imp.reset_index(drop=True))

# ---- standardize continuous columns + biomarkers using TRAIN statistics
scale_cols = CONT_COLS + BIOMARKERS
sc = StandardScaler().fit(Xtr[scale_cols])
Xtr[scale_cols] = sc.transform(Xtr[scale_cols])
Xte[scale_cols] = sc.transform(Xte[scale_cols])

# ---- assemble final frames
meta_cols = ["id", "obesity"] + [f"{o}_label" for o in OUTCOMES] + [f"{o}_followup_time" for o in OUTCOMES]
train_out = pd.concat([train_imp.reset_index(drop=True)[meta_cols], Xtr], axis=1)
test_out = pd.concat([test_imp.reset_index(drop=True)[meta_cols], Xte], axis=1)
train_out.to_parquet(f"{DATA_DIR}/train_imputed.parquet", index=False)
test_out.to_parquet(f"{DATA_DIR}/test_imputed.parquet", index=False)
log(f"saved parquet: train {train_out.shape}, test {test_out.shape}")

save_json({str(l): level_columns(Xtr, l) for l in (1, 2, 3)}, f"{DATA_DIR}/feature_levels.json")
with open(f"{DATA_DIR}/imputer_scaler.pkl", "wb") as fh:
    pickle.dump({"imputer": imp, "scaler": sc, "scale_cols": scale_cols}, fh)

# ---- imputation summary
summ = miss_before.to_frame().join(
    pd.Series(train_imp[IMPUTE_BLOCK].isna().mean(), name="train_missing_after")).fillna(0.0)
summ["test_missing_frac"] = test[IMPUTE_BLOCK].isna().mean()
pd.concat([summ.reset_index().rename(columns={"index": "variable"}),
           miss_rows_before], axis=1).fillna(method="ffill").to_csv(
    f"{RESULTS_DIR}/imputation_summary.csv", index=False)

for o in OUTCOMES:
    log(f"{o}: train events {int(train_out[f'{o}_label'].sum())}/{len(train_out)}"
        f" ({train_out[f'{o}_label'].mean()*100:.2f}%), test events {int(test_out[f'{o}_label'].sum())}/{len(test_out)}"
        f" ({test_out[f'{o}_label'].mean()*100:.2f}%)")
log("DONE 01_prepare_data")
