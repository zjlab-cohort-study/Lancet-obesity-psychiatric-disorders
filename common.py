"""Shared config & helpers for the obesity -> psychiatric disorder prediction pipeline."""
import json
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
RAW_CSV = "input.csv"
OUT = "output_dir"
DATA_DIR = f"{OUT}/data"
MODEL_DIR = f"{OUT}/models"
RESULTS_DIR = f"{OUT}/results"
FIG_DIR = f"{OUT}/figures"
LOG_DIR = f"{OUT}/logs"
for d in (DATA_DIR, MODEL_DIR, RESULTS_DIR, FIG_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
OUTCOMES = ["SUD"] # psychiatric disorder outcomes to predict
ALGOS = ["cox", "encox", "rsf", "xgb"]
ALGO_LABELS = {"cox": "Cox PH", "encox": "Elastic-net Cox", "rsf": "Random survival forest", "xgb": "XGBoost"}
LEVELS = [1, 2, 3]
LEVEL_NAMES = {1: "Model 1: covariates", 2: "Model 2: covariates + obesity", 3: "Model 3: covariates + obesity + biomarkers"}

OBESITY = "obesity"
CAT_COLS = ["smoking", "alcohol", "physical_activity","family_income", "employment"]  # multi-level -> one-hot
BIN_COLS = ["sex", "race", "college"]                           # binary 0/1 as-is
CONT_COLS = ["age", "sleep_duration",  "TDI"]               # numeric

# imputation block: complete columns act as predictors, biomarkers get imputed
BIOMARKERS =  [
    "biomarker_1",
    "biomarker_2",
    "biomarker_3",  
]
IMPUTE_BLOCK = BIOMARKERS + CONT_COLS + BIN_COLS + CAT_COLS + [OBESITY]

# evaluation horizons (years)
HORIZONS = [5, 10]
# grid for IBC / prediction-error curves; horizons are included
GRID = np.round(np.arange(0.25, 10.001, 0.25), 4)
assert all(h in GRID for h in HORIZONS)

ALGO_COLORS = {"cox": "#1f77b4", "encox": "#ff7f0e", "rsf": "#2ca02c", "xgb": "#d62728"}
LEVEL_COLORS = {1: "#999999", 2: "#1f77b4", 3: "#d62728"}
LEVEL_LSDASH = {1: ":", 2: "--", 3: "-"}


# ---------------------------------------------------------------- feature engineering
def dummy_name(col: str, val) -> str:
    return f"{col}_{int(val)}"


def build_design(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full design matrix (imputed block must already be imputed). Returns feature frame."""
    X = pd.DataFrame(index=df.index)
    for c in BIN_COLS + CONT_COLS:
        X[c] = df[c].astype(float)
    for c in CAT_COLS + [OBESITY]:
        vals = sorted(df[c].dropna().unique())
        for v in vals[1:]:  # drop first level as reference
            X[dummy_name(c, v)] = (df[c] == v).astype(float)
    for c in BIOMARKERS:
        X[c] = df[c].astype(float)
    return X


def level_columns(X: pd.DataFrame, level: int) -> list:
    cats = [c for c in X.columns if c.startswith(tuple(f"{col}_" for col in CAT_COLS))]
    obes = [c for c in X.columns if c.startswith(f"{OBESITY}_")]
    base = BIN_COLS + CONT_COLS + cats
    cols = {1: base, 2: base + obes, 3: base + obes + BIOMARKERS}[level]
    missing = [c for c in cols if c not in X.columns]
    assert not missing, f"missing features: {missing}"
    return cols


def make_y(df: pd.DataFrame, outcome: str):
    from sksurv.util import Surv
    ev = df[f"{outcome}_label"].astype(bool).to_numpy()
    tt = df[f"{outcome}_followup_time"].astype(float).to_numpy()
    return Surv.from_arrays(event=ev, time=tt)


# ---------------------------------------------------------------- survival helpers
def km_probability_at(time, event, t_star) -> float:
    """P(T <= t_star) via Kaplan-Meier."""
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter().fit(time, event_observed=event)
    return float(1.0 - kmf.predict(t_star))


def censoring_km_train(y_train):
    """KM of the censoring distribution (event = censored) on train."""
    from lifelines import KaplanMeierFitter
    ctime = y_train["time"]
    cev = ~y_train["event"]
    kmf = KaplanMeierFitter().fit(ctime, event_observed=cev)
    return kmf


def breslow_baseline_hazard(time, event, risk_weight):
    """Breslow baseline cumulative hazard given risk weights exp(f)."""
    d = pd.DataFrame({"t": time, "e": event.astype(int), "w": risk_weight})
    d = d.sort_values("t", ascending=False)
    denom = d.groupby("t", sort=False)["w"].sum().cumsum()  # sum of w over risk set at each t
    deaths = d[d.e == 1].groupby("t")["e"].sum()
    dh = (deaths / denom.reindex(deaths.index, fill_value=np.inf)).sort_index()
    H0 = dh.cumsum()
    return H0


def eval_step_functions(funcs, times) -> np.ndarray:
    """Evaluate a list of sksurv StepFunction objects at `times` -> (n_samples, len(times))."""
    return np.vstack([np.asarray(f(np.asarray(times, dtype=float)), dtype=float) for f in funcs])


def survival_from_hazard(H0_at_grid: np.ndarray, risk: np.ndarray) -> np.ndarray:
    """S(t|x) = exp(-H0(t) * exp(f(x))) for centered log-hazard f."""
    return np.exp(-np.outer(np.exp(risk - risk.mean()), H0_at_grid))


# ---------------------------------------------------------------- misc IO
def save_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)
