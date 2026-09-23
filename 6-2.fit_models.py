"""Step 02 — fit Cox PH / elastic-net Cox / RSF / XGBoost at 3 predictor levels for one outcome.

Usage:  python 6-2.fit_models.py <outcome> [--smoke 20000] [--reuse] [--only rsf]
Per-model artifacts are written immediately (checkpoint/resume with --reuse):
  data/risk_train_{tag}.npy, data/risk_test_{tag}.npy, data/S_test_{tag}.npy,
  results/meta_{tag}.json, models/*, results/coef_{tag}.csv
Final:  data/pred_test_{outcome}.parquet, data/pred_train_{outcome}.parquet,
        results/fit_meta_{outcome}.json
"""
import argparse, json, os, pickle, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from common import (DATA_DIR, MODEL_DIR, RESULTS_DIR, ALGOS, LEVELS, GRID, RANDOM_STATE,
                    make_y, breslow_baseline_hazard, eval_step_functions,
                    survival_from_hazard, save_json)
np.random.seed(RANDOM_STATE)
from sklearn.model_selection import KFold, train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter
from xgboost import XGBRegressor

ap = argparse.ArgumentParser()
ap.add_argument("outcome")
ap.add_argument("--smoke", type=int, default=0)
ap.add_argument("--reuse", action="store_true", help="skip models whose artifacts already exist")
ap.add_argument("--only", default="", help="comma-separated subset of algorithms")
args = ap.parse_args()
outcome = args.outcome
todo_algos = [a for a in ALGOS if not args.only or a in args.only.split(",")]

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

train = pd.read_parquet(f"{DATA_DIR}/train_imputed.parquet")
test = pd.read_parquet(f"{DATA_DIR}/test_imputed.parquet")
levels = {int(k): v for k, v in json.load(open(f"{DATA_DIR}/feature_levels.json")).items()}

if args.smoke:
    train = train.sample(args.smoke, random_state=42).reset_index(drop=True)
    test = test.iloc[:max(4000, args.smoke // 4)].reset_index(drop=True)  # 6-3 --smoke slices the same rows
    log(f"SMOKE MODE: train {len(train)}, test {len(test)}")

y_tr, y_te = make_y(train, outcome), make_y(test, outcome)

def surv_chunked(predict_fn, X, grid, chunk=4096):
    """Evaluate list-of-StepFunction survival predictions in chunks to bound memory."""
    outs = []
    for i in range(0, X.shape[0], chunk):
        funcs = predict_fn(X[i:i + chunk])
        outs.append(eval_step_functions(funcs, grid).astype(np.float32))
        del funcs
    return np.vstack(outs)

def checkpoint(tag, risk_tr, risk_te, S_te, m, model=None, save_pickle=False):
    np.asarray(risk_tr, np.float32).tofile(f"{DATA_DIR}/risk_train_{tag}.npy")
    np.asarray(risk_te, np.float32).tofile(f"{DATA_DIR}/risk_test_{tag}.npy")
    np.asarray(S_te, np.float32).tofile(f"{DATA_DIR}/S_test_{tag}.npy")
    m = {**m,
         "c_train": float(concordance_index_censored(y_tr["event"], y_tr["time"], risk_tr)[0]),
         "c_test": float(concordance_index_censored(y_te["event"], y_te["time"], risk_te)[0])}
    save_json(m, f"{RESULTS_DIR}/meta_{tag}.json")
    if save_pickle and model is not None:
        try:
            with open(f"{MODEL_DIR}/{tag}.pkl", "wb") as fh:
                pickle.dump(model, fh, protocol=5)
        except Exception as e:
            log(f"pickle save failed for {tag}: {e}")
    log(f"{tag}: C_train={m['c_train']:.4f} C_test={m['c_test']:.4f}")
    return m

def has_checkpoint(tag):
    exp = {"risk_train": len(train), "risk_test": len(test), "S_test": len(test) * len(GRID)}
    for k, n in exp.items():
        p = f"{DATA_DIR}/{k}_{tag}.npy"
        if not os.path.exists(p) or os.path.getsize(p) != n * 4:  # float32
            return False
    return os.path.exists(f"{RESULTS_DIR}/meta_{tag}.json")

# ================================================================= Cox PH
def fit_cox(level):
    cols = levels[level]
    d = train[cols + [f"{outcome}_followup_time", f"{outcome}_label"]].copy()
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(d, duration_col=f"{outcome}_followup_time", event_col=f"{outcome}_label")
    risk_tr = cph.predict_log_partial_hazard(train[cols]).to_numpy().ravel()
    risk_te = cph.predict_log_partial_hazard(test[cols]).to_numpy().ravel()
    S_te = cph.predict_survival_function(test[cols], times=GRID).to_numpy().T
    s = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    s.to_csv(f"{RESULTS_DIR}/coef_{outcome}_cox_m{level}.csv")
    return cph, risk_tr, risk_te, S_te, {"log_likelihood": float(cph.log_likelihood_),
                                         "n_features": len(cols)}, True


# ================================================================= Elastic-net Cox
def fit_encox(level):
    cols = levels[level]
    X, y = train[cols].to_numpy(float), y_tr
    path = CoxnetSurvivalAnalysis(l1_ratio=0.5, n_alphas=50, alpha_min_ratio=0.01,
                                  max_iter=200000, tol=1e-7, fit_baseline_model=True)
    path.fit(X, y)
    alphas = path.alphas_
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    scores = np.full((5, len(alphas)), np.nan)
    for k, (tr, va) in enumerate(cv.split(X)):
        m = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=alphas, max_iter=200000, tol=1e-7)
        m.fit(X[tr], y[tr])
        for j, a in enumerate(alphas):
            try:
                r = m.predict(X[va], alpha=a)
                scores[k, j] = concordance_index_censored(y[va]["event"], y[va]["time"], r)[0]
            except Exception:
                pass
    mean_scores = np.nanmean(scores, axis=0)
    best_alpha = float(alphas[np.argmax(mean_scores)])
    risk_tr = path.predict(X, alpha=best_alpha)
    risk_te = path.predict(test[cols].to_numpy(float), alpha=best_alpha)
    X_te = test[cols].to_numpy(float)
    S_te = surv_chunked(lambda xx: path.predict_survival_function(xx, alpha=best_alpha), X_te, GRID)
    ja = int(np.argmax(np.asarray(alphas) == best_alpha))
    pd.Series(path.coef_[:, ja], index=cols, name="coef").to_csv(
        f"{RESULTS_DIR}/coef_{outcome}_encox_m{level}.csv")
    return path, risk_tr, risk_te, S_te, {"best_alpha": best_alpha,
        "cv_c_max": float(np.nanmax(mean_scores)), "n_features": len(cols),
        "n_coef_nonzero": int((np.abs(path.coef_[:, ja]) > 1e-8).sum())}, True


# ================================================================= Random survival forest
RSF_TIME_BIN = 0.1   # discretize the time axis to bound memory: sksurv trees store a per-leaf
                     # CHF over all unique event times
def fit_rsf(level):
    from sksurv.util import Surv
    cols = levels[level]
    y_disc = Surv.from_arrays(event=y_tr["event"], time=np.round(y_tr["time"] / RSF_TIME_BIN) * RSF_TIME_BIN)
    # min_samples_leaf=200 chosen by sensitivity analysis: small leaves overfit badly once the
    # weak biomarkers enter (large train-test C gap; model-3 test C below model-2)
    rsf = RandomSurvivalForest(n_estimators=400, max_features="sqrt", min_samples_leaf=200,
                               max_samples=0.5, n_jobs=-1, random_state=RANDOM_STATE)
    rsf.fit(train[cols].to_numpy(float), y_disc)
    risk_tr = rsf.predict(train[cols].to_numpy(float))
    risk_te = rsf.predict(test[cols].to_numpy(float))
    X_te = test[cols].to_numpy(float)
    S_te = surv_chunked(rsf.predict_survival_function, X_te, GRID, chunk=2048)
    return rsf, risk_tr, risk_te, S_te, {"n_estimators": 400, "max_features": "sqrt",
                                         "min_samples_leaf": 200, "max_samples": 0.5,
                                         "n_features": len(cols),
                                         "time_discretization_y": RSF_TIME_BIN,
                                         "n_unique_times_used": int(np.unique(y_disc["time"]).size)}, True


# ================================================================= XGBoost (survival:cox + Breslow baseline)
def fit_xgb(level):
    cols = levels[level]
    X = train[cols].to_numpy(float)
    ylbl = np.where(y_tr["event"], y_tr["time"], -y_tr["time"])  # negative = censored
    X_fit, X_va, y_fit, y_va = train_test_split(X, ylbl, test_size=0.15,
                                                random_state=RANDOM_STATE, stratify=y_tr["event"])
    model = XGBRegressor(
        n_estimators=4000, learning_rate=0.05, max_depth=6, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="survival:cox", eval_metric="cox-nloglik",
        tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
        early_stopping_rounds=100)
    model.fit(X_fit, y_fit, eval_set=[(X_va, y_va)], verbose=False)
    f_tr = model.predict(X, iteration_range=(0, model.best_iteration + 1))
    f_te = model.predict(test[cols].to_numpy(float), iteration_range=(0, model.best_iteration + 1))
    # absolute-risk calibration: Cox PH on the single risk score (standard recalibration of ML
    # risk scores); a hand-rolled Breslow on exp(f) was mis-calibrated in pilot runs
    S_te = cox_recalibrate_survival(f_tr, f_te)
    model.save_model(f"{MODEL_DIR}/{outcome}_xgb_m{level}.json")
    return model, f_tr, f_te, S_te, {"best_iteration": int(model.best_iteration),
                                     "n_features": len(cols),
                                     "survival_calibration": "cox-on-score"}, False


FITTERS = {"cox": fit_cox, "encox": fit_encox, "rsf": fit_rsf, "xgb": fit_xgb}

def cox_recalibrate_survival(f_tr, f_te):
    """Survival function for a pre-specified risk score via Cox PH on the score (train-fit).
    Score enters as normal scores (rank transform): monotone, bounded, overflow-proof."""
    from scipy.stats import norm, rankdata
    z_tr = norm.ppf((rankdata(f_tr, method="average") - 0.5) / len(f_tr))
    pct_te = np.searchsorted(np.sort(f_tr), f_te, side="right") / len(f_tr)
    z_te = norm.ppf(np.clip(pct_te, 1e-6, 1 - 1e-6))
    d = pd.DataFrame({"f": z_tr, "T": y_tr["time"], "E": y_tr["event"]})
    cph = CoxPHFitter(penalizer=0.0).fit(d, duration_col="T", event_col="E")
    return cph.predict_survival_function(pd.DataFrame({"f": z_te}), times=GRID).to_numpy().T

meta = {}
for algo in todo_algos:
    for level in LEVELS:
        tag = f"{outcome}_{algo}_m{level}"
        if args.reuse and has_checkpoint(tag):
            meta[tag] = json.load(open(f"{RESULTS_DIR}/meta_{tag}.json"))
            log(f"{tag}: reuse existing checkpoint (C_test={meta[tag]['c_test']:.4f})")
            continue
        log(f"=== fitting {algo} level {level} ===")
        ts = time.time()
        model, risk_tr, risk_te, S_te, m, do_pickle = FITTERS[algo](level)
        meta[tag] = checkpoint(tag, risk_tr, risk_te, S_te, {**m, "fit_seconds": round(time.time() - ts, 1)},
                               model=model, save_pickle=do_pickle)

# ---------------------------------------------------------------- assemble prediction parquet
res_test = {"id": test["id"].to_numpy()}
res_train = {"id": train["id"].to_numpy()}
for tag in [f"{outcome}_{a}_m{l}" for a in ALGOS for l in LEVELS]:
    if tag in meta:
        res_test[f"risk_{tag}"] = np.fromfile(f"{DATA_DIR}/risk_test_{tag}.npy", np.float32)
        res_train[f"risk_{tag}"] = np.fromfile(f"{DATA_DIR}/risk_train_{tag}.npy", np.float32)
pd.DataFrame(res_test).to_parquet(f"{DATA_DIR}/pred_test_{outcome}.parquet", index=False)
pd.DataFrame(res_train).to_parquet(f"{DATA_DIR}/pred_train_{outcome}.parquet", index=False)
save_json(meta, f"{RESULTS_DIR}/fit_meta_{outcome}.json")
log("DONE 02_fit_models")
