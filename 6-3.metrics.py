"""Step 03 — evaluation for one outcome.

Reads:  data/{train,test}_imputed.parquet, data/pred_test_{outcome}.parquet,
        data/S_test_{outcome}_{algo}_m{level}.npy, results/fit_meta_{outcome}.json
Writes: results/metrics_{outcome}.csv          per-model: C, Uno C, AUC@5/10, BS@5/10, IBC,
                                                     calibration (O/E, slope, ICI), N
        results/bootstrap_{outcome}.npz        bootstrap reps (for CIs and deltas)
        results/increments_{outcome}.csv       Δ(model-2 − model-1, model-3 − model-1, 3−2)
        results/lrt_{outcome}.csv              partial-likelihood ratio tests (Cox PH)
        results/nri_idi_{outcome}.csv          continuous NRI/IDI at 10 y (IPCW)
        results/dca_{outcome}.csv              decision-curve net benefit at 10 y
        results/calibration_{outcome}.csv      calibration tables (5 y & 10 y)
"""
import argparse, json, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from scipy.stats import chi2
from lifelines import KaplanMeierFitter
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            cumulative_dynamic_auc, brier_score, integrated_brier_score)
from common import (DATA_DIR, RESULTS_DIR, ALGOS, LEVELS, GRID, HORIZONS, OUTCOMES,
                    censoring_km_train, make_y)

ap = argparse.ArgumentParser()
ap.add_argument("outcome")
ap.add_argument("--boot", type=int, default=300)
ap.add_argument("--smoke", type=int, default=0)
args = ap.parse_args()
outcome, B = args.outcome, args.boot

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

train = pd.read_parquet(f"{DATA_DIR}/train_imputed.parquet")
test = pd.read_parquet(f"{DATA_DIR}/test_imputed.parquet")
if args.smoke:
    test = test.iloc[:args.smoke].reset_index(drop=True)
preds = pd.read_parquet(f"{DATA_DIR}/pred_test_{outcome}.parquet").set_index("id").loc[test["id"]].reset_index(drop=True)
meta = json.load(open(f"{RESULTS_DIR}/fit_meta_{outcome}.json"))

y_tr = make_y(train, outcome)
y_te = make_y(test, outcome)
T_te, E_te = y_te["time"], y_te["event"]
n = len(test)

TAGS = [f"{outcome}_{a}_m{l}" for a in ALGOS for l in LEVELS]
risk = {t: preds[f"risk_{t}"].to_numpy(float) for t in TAGS}
S = {t: np.fromfile(f"{DATA_DIR}/S_test_{t}.npy", dtype=np.float32).reshape(n, len(GRID)).astype(float) for t in TAGS}
S = {t: np.clip(v, 1e-12, 1.0) for t, v in S.items()}

idx = {h: int(np.where(np.isclose(GRID, h))[0][0]) for h in HORIZONS}
p_event = {t: 1.0 - v for t, v in S.items()}  # predicted event probability on GRID

# ---------------------------------------------------------------- IPCW machinery (censoring dist from TRAIN)
G_km = censoring_km_train(y_tr)
def G_at(tt):
    return np.clip(np.asarray(G_km.predict(tt), float), 1e-12, 1.0)

G_T = G_at(T_te)                      # G(T_i-) evaluated at each individual time
G_h = {h: float(G_at(h)) for h in HORIZONS}

# Graf IPCW Brier weights (model independent): BS(t) = mean( S^2 * A_t + (1-S)^2 * B_t )
WA = np.zeros((n, len(GRID)))
WB = np.zeros((n, len(GRID)))
for j, t in enumerate(GRID):
    case = (T_te <= t) & E_te
    ctrl = T_te > t
    WA[case, j] = 1.0 / G_T[case]
    WB[ctrl, j] = 1.0 / G_at(t)

def bs_curve(Sm):  # (n, T) -> BS at each grid time
    return (Sm**2 * WA + (1 - Sm)**2 * WB).mean(axis=0)

def ibc(Sm):
    bc = bs_curve(Sm)
    return float(np.trapezoid(bc, GRID) / (GRID[-1] - GRID[0]))

def auc_t(r, t):
    """Incident/dynamic AUC at t (IPCW, controls weighted by 1/G(t))."""
    case = (T_te <= t) & E_te
    ctrl = T_te > t
    rc, ro = r[case], r[ctrl]
    w = 1.0 / G_T[case]
    order = np.argsort(ro)
    ro_s = ro[order]
    pos = np.searchsorted(ro_s, rc, side="right")      # strictly less
    tie = np.searchsorted(ro_s, rc, side="left")       # less-or-equal
    num = np.sum(w * (pos + 0.5 * (tie - pos)))
    return float(num / (np.sum(w) * len(ro_s)))

# ---- validate manual implementations against scikit-survival on the full test set
Sm = S[TAGS[0]]; rm = risk[TAGS[0]]
sk_bs = brier_score(y_tr, y_te, Sm[:, idx[10]], [10.0])[1][0]
man_bs = bs_curve(Sm)[idx[10]]
sk_auc = cumulative_dynamic_auc(y_tr, y_te, rm, [10.0])[0][0]
man_auc = auc_t(rm, 10.0)
log(f"validation: BS sksurv={sk_bs:.6f} manual={man_bs:.6f} | AUC sksurv={sk_auc:.6f} manual={man_auc:.6f}")
assert abs(sk_bs - man_bs) < 1e-3 and abs(sk_auc - man_auc) < 1e-3, "manual metric mismatch"

# ---------------------------------------------------------------- calibration helpers
def km_obs_prob(mask, t_star):
    if mask.sum() < 5:
        return np.nan
    kmf = KaplanMeierFitter().fit(T_te[mask], event_observed=E_te[mask])
    return float(1.0 - kmf.predict(t_star))

def calibration(p, t_star, n_groups=10, ici_bins=50):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    grp = pd.qcut(pd.Series(p).rank(method="first"), n_groups, labels=False).to_numpy()
    rows = []
    for g in range(n_groups):
        m = grp == g
        rows.append({"group": g + 1, "n": int(m.sum()), "pred_mean": float(p[m].mean()),
                     "obs_km": km_obs_prob(m, t_star)})
    cal = pd.DataFrame(rows)
    cal["oe"] = cal["obs_km"] / cal["pred_mean"]
    valid = cal.dropna()
    slope = float(np.polyfit(valid["pred_mean"], valid["obs_km"], 1, w=valid["n"])[0]) if len(valid) > 2 else np.nan
    intercept = float(np.polyfit(valid["pred_mean"], valid["obs_km"], 1, w=valid["n"])[1]) if len(valid) > 2 else np.nan
    # ICI: smooth calibration via quantile bins + interpolation
    gb = pd.qcut(pd.Series(p).rank(method="first"), ici_bins, labels=False).to_numpy()
    obs_b, ctr_b = np.full(ici_bins, np.nan), np.full(ici_bins, np.nan)
    for g in range(ici_bins):
        m = gb == g
        obs_b[g] = km_obs_prob(m, t_star)
        ctr_b[g] = p[m].mean()
    ok = ~np.isnan(obs_b)
    smoothed = np.interp(p, ctr_b[ok], obs_b[ok])
    ici = float(np.mean(np.abs(smoothed - p)))
    return cal, slope, intercept, ici

# ---------------------------------------------------------------- per-model metrics
rows, cal_tables = [], []
for tag in TAGS:
    algo, lvl = tag.split("_")[1], tag.split("_")[-1]
    r, Sm = risk[tag], S[tag]
    c_h = concordance_index_censored(E_te, T_te, r)[0]
    c_uno = concordance_index_ipcw(y_tr, y_te, r, tau=float(GRID[-1]))[0]
    aucs = {h: (cumulative_dynamic_auc(y_tr, y_te, r, [float(h)])[0][0]) for h in HORIZONS}
    bs = {h: float(bs_curve(Sm)[idx[h]]) for h in HORIZONS}
    row = {"outcome": outcome, "algo": algo, "level": int(lvl[1]),
           "n_test": n, "events_test": int(E_te.sum()),
           "C_harrell": c_h, "C_uno": c_uno,
           "AUC_5y": aucs[5], "AUC_10y": aucs[10],
           "Brier_5y": bs[5], "Brier_10y": bs[10], "IBC_0.25-10y": ibc(Sm)}
    for h in HORIZONS:
        cal, slope, intercept, ici = calibration(p_event[tag][:, idx[h]], float(h))
        cal.insert(0, "model", tag); cal.insert(1, "horizon_y", h)
        cal_tables.append(cal)
        ok = cal.dropna(subset=["obs_km"])
        row[f"cal_OE_{h}y"] = float((ok["n"] * ok["obs_km"]).sum() / (ok["n"] * ok["pred_mean"]).sum())
        row[f"cal_slope_{h}y"] = slope
        row[f"cal_ICI_{h}y"] = ici
    rows.append(row)
    log(f"{tag}: C={c_h:.4f} AUC10={aucs[10]:.4f} BS10={bs[5]:.4f}/{bs[10]:.4f} IBC={row['IBC_0.25-10y']:.4f}")
metrics = pd.DataFrame(rows)
metrics.to_csv(f"{RESULTS_DIR}/metrics_{outcome}.csv", index=False)
pd.concat(cal_tables).to_csv(f"{RESULTS_DIR}/calibration_{outcome}.csv", index=False)

# prediction-error curves (IPCW Brier vs time) for figures
pec = [pd.DataFrame({"model": t, "time": GRID, "brier": bs_curve(S[t])}) for t in TAGS]
pd.concat(pec).to_csv(f"{RESULTS_DIR}/pec_curves_{outcome}.csv", index=False)

# ---------------------------------------------------------------- bootstrap (paired, for CIs and deltas)
log(f"bootstrap B={B} (parallel) ...")
import multiprocessing as mp

_BG = {"T": T_te, "E": E_te, "G_T": G_T, "WA": WA, "WB": WB, "risk": risk, "S": S,
       "TAGS": TAGS, "GRID": GRID, "idx10": idx[10], "n": n, "H": HORIZONS}

def _auc_ipcw(r, case, ctrl, w_case):
    rc, ro = r[case], r[ctrl]
    ro_s = np.sort(ro)
    pos = np.searchsorted(ro_s, rc, side="right")
    tie = np.searchsorted(ro_s, rc, side="left")
    return np.sum(w_case * (pos + 0.5 * (tie - pos))) / (np.sum(w_case) * len(ro_s))

def _boot_rep(b):
    rng = np.random.default_rng(10007 + b)
    ii = rng.integers(0, _BG["n"], _BG["n"])
    Tb, Eb = _BG["T"][ii], _BG["E"][ii]
    WAb, WBb = _BG["WA"][ii], _BG["WB"][ii]
    G_Tb = _BG["G_T"][ii]
    cases = {h: ((Tb <= h) & Eb, Tb > h) for h in _BG["H"]}
    out = {}
    for tag in _BG["TAGS"]:
        r = _BG["risk"][tag][ii]; Sm = _BG["S"][tag][ii]
        c = concordance_index_censored(Eb, Tb, r)[0]
        aucs = {}
        for h, (ca, ct) in cases.items():
            aucs[h] = _auc_ipcw(r, ca, ct, 1.0 / G_Tb[ca])
        bc = (Sm**2 * WAb + (1 - Sm)**2 * WBb).mean(axis=0)
        out[tag] = (float(c), float(aucs[5]), float(aucs[10]), float(bc[_BG["idx10"]]),
                    float(np.trapezoid(bc, _BG["GRID"]) / (_BG["GRID"][-1] - _BG["GRID"][0])))
    return b, out

boot = {t: {"C": np.empty(B), "AUC5": np.empty(B), "AUC10": np.empty(B),
            "BS10": np.empty(B), "IBC": np.empty(B)} for t in TAGS}
with mp.get_context("fork").Pool(min(32, mp.cpu_count())) as pool:
    done = 0
    for b, res in pool.imap_unordered(_boot_rep, range(B), chunksize=2):
        for t, vals in res.items():
            (boot[t]["C"][b], boot[t]["AUC5"][b], boot[t]["AUC10"][b],
             boot[t]["BS10"][b], boot[t]["IBC"][b]) = vals
        done += 1
        if done % 50 == 0:
            log(f"  boot {done}/{B}")
np.savez_compressed(f"{RESULTS_DIR}/bootstrap_{outcome}.npz",
                    **{f"{t}_{k}": v for t, d in boot.items() for k, v in d.items()})

# CIs for each model metric
def ci(x):
    return np.percentile(x, 2.5), np.percentile(x, 97.5)

for i, row in metrics.iterrows():
    tag = f"{outcome}_{row['algo']}_m{row['level']}"
    for met in ("C", "AUC5", "AUC10", "BS10", "IBC"):
        lo, hi = ci(boot[tag][met])
        col = {"C": "C_harrell", "AUC5": "AUC_5y", "AUC10": "AUC_10y",
               "BS10": "Brier_10y", "IBC": "IBC_0.25-10y"}[met]
        metrics.loc[i, f"{col}_lo"] = lo; metrics.loc[i, f"{col}_hi"] = hi
metrics.to_csv(f"{RESULTS_DIR}/metrics_{outcome}.csv", index=False)

# deltas with paired bootstrap CI + p-value
PAIRS = [(2, 1), (3, 1), (3, 2)]
drows = []
METMAP = {"C_harrell": "C", "AUC_5y": "AUC5", "AUC_10y": "AUC10",
          "Brier_10y": "BS10", "IBC_0.25-10y": "IBC"}
for algo in ALGOS:
    for (hi_l, lo_l) in PAIRS:
        th, tl = f"{outcome}_{algo}_m{hi_l}", f"{outcome}_{algo}_m{lo_l}"
        base = metrics[(metrics.algo == algo) & (metrics.level == lo_l)].iloc[0]
        cur = metrics[(metrics.algo == algo) & (metrics.level == hi_l)].iloc[0]
        for col, bk in METMAP.items():
            d = cur[col] - base[col]
            db = boot[th][bk] - boot[tl][bk]
            lo_, hi_ = ci(db)
            p = 2 * min((db <= 0).mean(), (db >= 0).mean())
            drows.append({"outcome": outcome, "algo": algo, "comparison": f"model{hi_l} - model{lo_l}",
                          "metric": col, "base": base[col], "new": cur[col], "delta": d,
                          "delta_lo": lo_, "delta_hi": hi_, "p_bootstrap": max(p, 1.0 / B)})
pd.DataFrame(drows).to_csv(f"{RESULTS_DIR}/increments_{outcome}.csv", index=False)

# ---------------------------------------------------------------- LRT (Cox PH, training-set partial likelihood)
lrt = []
for hi_l, lo_l in PAIRS:
    mh = meta[f"{outcome}_cox_m{hi_l}"]; ml = meta[f"{outcome}_cox_m{lo_l}"]
    stat = 2 * (mh["log_likelihood"] - ml["log_likelihood"])
    dfree = mh["n_features"] - ml["n_features"]
    lrt.append({"outcome": outcome, "comparison": f"model{hi_l} vs model{lo_l}",
                "logLik_model": mh["log_likelihood"], "logLik_base": ml["log_likelihood"],
                "chi2": stat, "df": dfree, "p": chi2.sf(stat, dfree)})
pd.DataFrame(lrt).to_csv(f"{RESULTS_DIR}/lrt_{outcome}.csv", index=False)

# ---------------------------------------------------------------- continuous NRI / IDI at 10 y (IPCW)
def nri_idi(p1, p2, t_star=10.0):
    """Continuous NRI/IDI (IPCW). Event weights 1/G(T_i); nonevent weights uniform 1/G(t*)."""
    case = (T_te <= t_star) & E_te
    ctrl = T_te > t_star
    wc = 1.0 / G_T[case]
    up_c, dn_c = p2[case] > p1[case], p2[case] < p1[case]
    up_n, dn_n = p2[ctrl] > p1[ctrl], p2[ctrl] < p1[ctrl]
    nri_ev = np.sum(wc * up_c) / np.sum(wc) - np.sum(wc * dn_c) / np.sum(wc)
    # uniform nonevent weights -> plain means
    nri_nev = dn_n.mean() - up_n.mean()
    idi = (np.sum(wc * (p2[case] - p1[case])) / np.sum(wc)) - (p2[ctrl] - p1[ctrl]).mean()
    return float(nri_ev + nri_nev), float(nri_ev), float(nri_nev), float(idi)

nrows = []
for algo in ALGOS:
    for hi_l, lo_l in PAIRS:
        p1 = p_event[f"{outcome}_{algo}_m{lo_l}"][:, idx[10]]
        p2 = p_event[f"{outcome}_{algo}_m{hi_l}"][:, idx[10]]
        nn, ev, nev, idi = nri_idi(p1, p2)
        nrows.append({"outcome": outcome, "algo": algo, "comparison": f"model{hi_l} - model{lo_l}",
                      "NRI_continuous": nn, "NRI_event": ev, "NRI_nonevent": nev, "IDI": idi})
pd.DataFrame(nrows).to_csv(f"{RESULTS_DIR}/nri_idi_{outcome}.csv", index=False)

# ---------------------------------------------------------------- decision-curve analysis at 10 y
ths = np.round(np.arange(0.01, 0.301, 0.005), 4)
dca_rows = []
def nb_series(tag, label):
    p = p_event[tag][:, idx[10]]
    for pt in ths:
        pos = p >= pt
        if pos.sum() < 10:
            nb = np.nan
        else:
            # FP/n must come from the predicted-positive group (pos), not the negative group:
            # TP/n = P(pos & event) = pos.mean()*tp ; FP/n = P(pos & no event) = pos.mean()*(1-tp)
            tp = km_obs_prob(pos, 10.0)
            nb = tp * pos.mean() - (1.0 - tp) * pos.mean() * (pt / (1 - pt))
        dca_rows.append({"model": label, "threshold": pt, "net_benefit": nb})

for algo in ALGOS:
    for l in LEVELS:
        nb_series(f"{outcome}_{algo}_m{l}", f"{algo}_m{l}")
km_all = 1.0 - KaplanMeierFitter().fit(T_te, event_observed=E_te).predict(10.0)
for pt in ths:
    dca_rows.append({"model": "treat_all", "threshold": pt, "net_benefit": km_all - (1 - km_all) * (pt / (1 - pt))})
    dca_rows.append({"model": "treat_none", "threshold": pt, "net_benefit": 0.0})
pd.DataFrame(dca_rows).to_csv(f"{RESULTS_DIR}/dca_{outcome}.csv", index=False)
log("DONE 03_metrics")
