"""
Step 04 — figures (PDF) & final report for all outcomes.

Also computes the time-dependent (cumulative/dynamic, IPCW) ROC curves here from the saved
risk-score checkpoints (no refit / no bootstrap; each curve's trapezoid AUC is cross-checked
against the AUC_5y/AUC_10y columns produced by scikit-survival in 03_metrics).

Figures (per outcome, all PDF in figures/):
  fig_{outcome}_C_index.pdf          forest: C-index (test) with bootstrap 95% CI
  fig_{outcome}_calibration_10y.pdf  12-panel calibration at 10 y
  fig_{outcome}_pec.pdf              prediction-error (IPCW Brier) curves vs time
  fig_{outcome}_roc_5y / _10y.pdf    time-dependent ROC curves, 12 panels
  fig_{outcome}_dca_10y.pdf          decision curves at 10 y
  fig_{outcome}_delta_C.pdf          incremental C-index vs model-1 with 95% CI
  fig_{outcome}_shap_beeswarm.pdf    SHAP beeswarm, XGBoost model-3 (random test-set subsample)
  fig_{outcome}_shap_importance.pdf  mean |SHAP| by feature group, XGBoost model-3
Writes: results/roc_curves_{outcome}.csv, results/shap_importance_{outcome}.csv,
        data/shap_{outcome}_xgb_m3.npy, REPORT.md, tables_combined.csv
"""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (OUT, RAW_CSV, DATA_DIR, MODEL_DIR, RESULTS_DIR, FIG_DIR, ALGOS, LEVELS,
                    GRID, HORIZONS, OUTCOMES, ALGO_LABELS, ALGO_COLORS, LEVEL_COLORS,
                    LEVEL_NAMES, LEVEL_LSDASH, BIOMARKERS, OBESITY, make_y, censoring_km_train)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})

def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}.pdf")
    plt.close(fig)

# ---------------------------------------------------------------- KM curves by XGBoost risk tertiles
def km_section(o, test):
    """KM curves by XGBoost model-3 predicted-risk tertiles + log-rank + numbers at risk."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    OUTCN_ = {"anxiety": "Anxiety", "depression": "Depression", "SUD": "SUD"}[o]
    risk = np.fromfile(f"{DATA_DIR}/risk_test_{o}_xgb_m3.npy", np.float32)
    T = test[f"{o}_followup_time"].to_numpy(float)
    E = test[f"{o}_label"].to_numpy(bool)
    grp = pd.qcut(pd.Series(risk).rank(method="first"), 3, labels=["Low", "Middle", "High"]).to_numpy()
    colors = {"Low": "#4daf4a", "Middle": "#ff7f00", "High": "#d62728"}
    marks = [0, 5, 10, 15]

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 5.8), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1.15], "hspace": 0.05})
    for g in ("Low", "Middle", "High"):
        mask = grp == g
        kmf = KaplanMeierFitter().fit(T[mask], E[mask], label=f"{g} risk, n={mask.sum():,}")
        kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.15, color=colors[g], lw=1.6)
        for t in marks:  # numbers at risk
            n = int((T[mask] >= t).sum())
            axr.text(t, 0.62 - 0.31 * ("Low", "Middle", "High").index(g), f"{n:,}",
                     ha="center", va="center", fontsize=7.5, color=colors[g])
    p = multivariate_logrank_test(T, grp, E).p_value
    ax.text(0.985, 0.97, f"log-rank P = {p:.1e}" if p >= 1e-300 else "log-rank P < 1e-300",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.set_ylabel("Event-free probability")
    ax.set_ylim(None, 1.003)
    ax.legend(fontsize=8, loc="lower left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axr.set_yticks([]); axr.set_xticks(marks)
    axr.text(-0.4, 0.62, "Numbers\nat risk", ha="right", va="center", fontsize=7.5)
    for s in axr.spines.values():
        s.set_visible(False)
    axr.set_xlabel("Time since baseline (years)")
    axr.set_xlim(-0.5, 16.2)
    ax.set_title(f"{OUTCN_}: KM curves by XGBoost model-3 predicted-risk tertiles (test set)")
    savefig(fig, f"fig_{o}_km_xgb_m3")

def tag(o, a, l):
    return f"{o}_{a}_m{l}"

# ---------------------------------------------------------------- time-dependent ROC
def roc_td(risk, T, E, t_star, G_km, n_pts=1000):
    """Weighted cumulative/dynamic ROC at t* (cases T<=t* & event, w=1/G(T); controls T>t*)."""
    case = (T <= t_star) & E
    ctrl = T > t_star
    rc, ro = risk[case], risk[ctrl]
    w = 1.0 / np.clip(np.asarray(G_km.predict(T[case]), float), 1e-12, 1.0)
    thr = np.unique(np.concatenate([rc, ro, [-np.inf, np.inf]]))
    s_c = np.sort(rc)
    w_asc = np.cumsum(w[np.argsort(rc, kind="stable")])
    k_c = np.searchsorted(s_c, thr, side="left")
    W_tot = w_asc[-1]
    tpr = (W_tot - np.where(k_c > 0, w_asc[np.clip(k_c - 1, 0, None)], 0.0)) / W_tot
    fpr = 1.0 - np.searchsorted(np.sort(ro), thr, side="left") / len(ro)
    order = np.argsort(fpr, kind="stable")
    fpr, tpr = fpr[order], tpr[order]
    auc = float(np.trapezoid(tpr, fpr))
    ii = np.unique(np.linspace(0, len(fpr) - 1, n_pts).astype(int))  # decimate, keep endpoints
    return fpr[ii], tpr[ii], auc

def compute_roc_curves(outcome, G_km, T, E, test_ids):
    """ROC curves for all 12 models x 2 horizons; saves CSV and returns the long data frame."""
    preds = pd.read_parquet(f"{DATA_DIR}/pred_test_{outcome}.parquet") \
              .set_index("id").loc[test_ids].reset_index(drop=True)
    met = pd.read_csv(f"{RESULTS_DIR}/metrics_{outcome}.csv")
    rows = []
    for h in HORIZONS:
        for a in ALGOS:
            for l in LEVELS:
                t = tag(outcome, a, l)
                fpr, tpr, auc = roc_td(preds[f"risk_{t}"].to_numpy(float), T, E, float(h), G_km)
                ref = float(met[(met.algo == a) & (met.level == l)][f"AUC_{h}y"].iloc[0])
                assert abs(auc - ref) < 5e-3, f"{t}@{h}y trapezoid {auc:.4f} vs sksurv {ref:.4f}"
                rows.append(pd.DataFrame({"model": t, "horizon_y": h, "fpr": fpr, "tpr": tpr}))
    curves = pd.concat(rows)
    curves.to_csv(f"{RESULTS_DIR}/roc_curves_{outcome}.csv", index=False)
    return curves

# ---------------------------------------------------------------- decision-curve analysis (5 & 10 y)
def compute_dca(outcome, T, E, h):
    """Net-benefit curves at horizon h from the cached survival matrices; returns long df."""
    from lifelines import KaplanMeierFitter
    n = len(T)

    def km_event_prob(mask, t_star):
        if mask.sum() < 10:
            return np.nan
        return float(1 - KaplanMeierFitter().fit(T[mask], E[mask]).predict(t_star))

    ths = np.round(np.arange(0.005, 0.1501, 0.005) if h == 5 else np.arange(0.01, 0.3001, 0.005), 4)
    jh = int(np.where(np.isclose(GRID, h))[0][0])
    rows = []
    for a in ALGOS:
        for l in LEVELS:
            t = tag(outcome, a, l)
            S = np.fromfile(f"{DATA_DIR}/S_test_{t}.npy", np.float32).reshape(n, len(GRID))
            p = 1.0 - S[:, jh].astype(float)
            for pt in ths:
                pos = p >= pt
                if pos.sum() < 10:
                    nb = np.nan
                else:
                    # FP/n must come from the predicted-positive group (pos), not the negative group:
                    # TP/n = P(pos & event) = pos.mean()*tp ; FP/n = P(pos & no event) = pos.mean()*(1-tp)
                    tp = km_event_prob(pos, float(h))
                    nb = tp * pos.mean() - (1.0 - tp) * pos.mean() * (pt / (1 - pt))
                rows.append({"model": f"{a}_m{l}", "horizon_y": h, "threshold": pt, "net_benefit": nb})
    prev = km_event_prob(np.ones(n, bool), float(h))
    for pt in ths:
        rows.append({"model": "treat_all", "horizon_y": h, "threshold": pt,
                     "net_benefit": prev - (1 - prev) * (pt / (1 - pt))})
        rows.append({"model": "treat_none", "horizon_y": h, "threshold": pt, "net_benefit": 0.0})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- SHAP (XGBoost model-3)
GROUP_COLORS = {"covariate": "#999999", "obesity": "#1f77b4", "biomarker": "#d62728"}
GROUP_NAMES = {"covariate": "Covariates", "obesity": "Obesity", "biomarker": "Biomarkers"}

def feature_group(name):
    if name.startswith(f"{OBESITY}_"):
        return "obesity"
    if name in BIOMARKERS:
        return "biomarker"
    return "covariate"

SHAP_N_SAMPLES = 2000   # rows of the test set used for TreeSHAP / SHAP figures (few hundred–few thousand suffices)
SHAP_SEED = 42

def shap_section(o, test, levels3, scaler_pack, n_samples=SHAP_N_SAMPLES, seed=SHAP_SEED):
    """TreeSHAP for the saved XGBoost model-3 on a random test-set subsample;
    returns (mean|SHAP| table, raw-value matrix, shap matrix)."""
    from xgboost import XGBRegressor, DMatrix
    model = XGBRegressor()
    model.load_model(f"{MODEL_DIR}/{o}_xgb_m3.json")
    bi = getattr(model, "best_iteration", None)
    kw = {"iteration_range": (0, int(bi) + 1)} if bi is not None else {}
    X = test[levels3].to_numpy(float)
    # subsample rows before pred_contribs (the expensive step); a random subsample of
    # a few hundred–few thousand gives the typical beeswarm / importance figures
    rng = np.random.default_rng(seed)
    ii = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X = X[ii]
    contrib = model.get_booster().predict(DMatrix(X), pred_contribs=True, **kw)
    shap_vals = np.asarray(contrib[:, :-1], dtype=np.float32)   # last col = bias / base value
    np.save(f"{DATA_DIR}/shap_{o}_xgb_m3.npy", shap_vals)
    print(f"{o}: TreeSHAP on {len(ii)}/{len(test)} test samples (random subsample, seed {seed})")

    # raw (un-standardized) feature values for display
    test_sub = test.iloc[ii]
    sc, scale_cols = scaler_pack["scaler"], scaler_pack["scale_cols"]
    X_raw = test_sub[levels3].copy()
    X_raw[scale_cols] = sc.inverse_transform(test_sub[scale_cols])

    imp = pd.DataFrame({"feature": levels3,
                        "group": [feature_group(c) for c in levels3],
                        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
                        "mean_shap": shap_vals.mean(axis=0)}) \
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.to_csv(f"{RESULTS_DIR}/shap_importance_{o}.csv", index=False)
    return imp, X_raw[levels3].to_numpy(float), shap_vals

def plot_shap_figures(o, imp, X_raw, shap_vals, levels3, top=20, n_pts=2000, seed=42):
    import shap
    OUTCN_ = {"anxiety": "Anxiety", "depression": "Depression", "SUD": "SUD"}[o]
    n = len(shap_vals)
    # (a) importance bar, all features, colored by predictor group
    d = imp.sort_values("mean_abs_shap")  # smallest at bottom
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(d.feature, d.mean_abs_shap, color=[GROUP_COLORS[g] for g in d.group], height=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("mean |SHAP value|  (log relative hazard, log scale)")
    ax.set_title(f"{OUTCN_}: SHAP importance — XGBoost model 3 (test-set subsample, n = {n})")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GROUP_COLORS[g], label=GROUP_NAMES[g]) for g in GROUP_COLORS],
              loc="lower right", fontsize=8)
    savefig(fig, f"fig_{o}_shap_importance")

    # (b) classic SHAP beeswarm (shap library): red = high feature value, blue = low;
    # a few hundred-to-a-few-thousand points give the standard, smoothly-swarmed look
    rng = np.random.default_rng(seed)
    ii = rng.choice(n, size=min(n_pts, n), replace=False)
    explanation = shap.Explanation(values=shap_vals[ii], data=X_raw[ii], feature_names=levels3)
    shap.plots.beeswarm(explanation, max_display=top, show=False, plot_size=(9.5, 7.2),
                        group_remaining_features=False)
    fig = plt.gcf()
    fig.axes[0].set_title(
        f"{OUTCN_}: SHAP summary — XGBoost model 3 (top {top}, test-set subsample, n = {len(ii)})",
        fontsize=9)
    fig.axes[0].set_xlabel("SHAP value (impact on log relative hazard)")
    savefig(fig, f"fig_{o}_shap_beeswarm")

# ---------------------------------------------------------------- load results
tables, incs, lrts, nris, metas = {}, {}, {}, {}, {}
for o in OUTCOMES:
    tables[o] = pd.read_csv(f"{RESULTS_DIR}/metrics_{o}.csv")
    incs[o] = pd.read_csv(f"{RESULTS_DIR}/increments_{o}.csv")
    lrts[o] = pd.read_csv(f"{RESULTS_DIR}/lrt_{o}.csv")
    nris[o] = pd.read_csv(f"{RESULTS_DIR}/nri_idi_{o}.csv")
    metas[o] = json.load(open(f"{RESULTS_DIR}/fit_meta_{o}.json"))
pd.concat(tables.values()).to_csv(f"{RESULTS_DIR}/tables_combined.csv", index=False)

OUTCN = {"anxiety": "Anxiety", "depression": "Depression", "SUD": "SUD"}
scaler_pack = pickle.load(open(f"{DATA_DIR}/imputer_scaler.pkl", "rb"))
levels3 = json.load(open(f"{DATA_DIR}/feature_levels.json"))["3"]
shap_top5, shap_ob_rank, shap_n = {}, {}, {}

for o in OUTCOMES:
    m = tables[o]

    # ---- Fig 1: C-index by algorithm (x) and predictor level (y), with bootstrap 95% CI
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = np.arange(len(ALGOS))
    off = {1: -0.24, 2: 0.0, 3: 0.24}
    for ai, a in enumerate(ALGOS):
        pts = []
        for l in LEVELS:
            r = m[(m.algo == a) & (m.level == l)].iloc[0]
            x = xs[ai] + off[l]
            ax.errorbar(x, r["C_harrell"],
                        yerr=[[r["C_harrell"] - r["C_harrell_lo"]], [r["C_harrell_hi"] - r["C_harrell"]]],
                        fmt="o", color=LEVEL_COLORS[l], capsize=2.5, ms=5.5)
            pts.append((x, r["C_harrell"]))
            ax.text(x, r["C_harrell_hi"] + 0.0015, f'{r["C_harrell"]:.3f}', ha="center", fontsize=7, color=LEVEL_COLORS[l])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="0.75", lw=0.7, zorder=0)
    ax.set_xticks(xs); ax.set_xticklabels([ALGO_LABELS[a] for a in ALGOS], fontsize=9)
    ax.set_ylabel("Harrell C-index (test set)")
    lo = m["C_harrell_lo"].min() - 0.008; hi = m["C_harrell_hi"].max() + 0.008
    ax.set_ylim(lo, hi)
    from matplotlib.lines import Line2D as _L2D
    ax.legend(handles=[_L2D([], [], color=LEVEL_COLORS[l], marker="o", ls="", label=LEVEL_NAMES[l]) for l in LEVELS],
              fontsize=8, loc="lower right")
    ax.set_title(f"{OUTCN[o]}: C-index by algorithm and predictor level (95% bootstrap CI)")
    savefig(fig, f"fig_{o}_C_index")

    # ---- Fig 2: calibration at 5 y & 10 y (12 panels each)
    cal = pd.read_csv(f"{RESULTS_DIR}/calibration_{o}.csv")
    for h in HORIZONS:
        ch = cal[cal.horizon_y == h]
        fig, axes = plt.subplots(3, 4, figsize=(11, 7.5), sharex=True, sharey=True)
        for ai, a in enumerate(ALGOS):
            for l in LEVELS:
                axi = axes[l - 1, ai]
                d = ch[ch.model == tag(o, a, l)]
                axi.plot(d.pred_mean * 100, d.obs_km * 100, "o-", color=ALGO_COLORS[a], ms=3.5, lw=1)
                axi.plot([0, 100], [0, 100], "k--", lw=0.7)
                rm = m[(m.algo == a) & (m.level == l)].iloc[0]
                axi.set_title(f"{ALGO_LABELS[a]} M{l} (slope {rm[f'cal_slope_{h}y']:.2f})", fontsize=8)
                if l == 3: axi.set_xlabel(f"Predicted {h}-y risk (%)")
                if ai == 0: axi.set_ylabel(f"Observed {h}-y risk (%)" if l == 2 else "")
        lim = max(ch.pred_mean.max(), np.nanmax(ch.obs_km)) * 100 * 1.05
        for axi in axes.ravel():
            axi.set_xlim(0, lim); axi.set_ylim(0, lim)
        fig.suptitle(f"{OUTCN[o]}: calibration at {h} years (test set, deciles)")
        savefig(fig, f"fig_{o}_calibration_{h}y")

    # ---- Fig 3: prediction-error curves (IPCW Brier vs time)
    pec = pd.read_csv(f"{RESULTS_DIR}/pec_curves_{o}.csv")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ai, a in enumerate(ALGOS):
        axi = axes[ai]
        for l in LEVELS:
            d = pec[pec.model == tag(o, a, l)]
            axi.plot(d.time, d.brier, color=LEVEL_COLORS[l], ls=LEVEL_LSDASH[l], lw=1.4, label=f"M{l}")
        axi.set_title(ALGO_LABELS[a], fontsize=9)
        axi.set_xlabel("Time (years)")
        if ai == 0:
            axi.set_ylabel("IPCW Brier score"); axi.legend(fontsize=7)
    fig.suptitle(f"{OUTCN[o]}: prediction error over time (test set)")
    savefig(fig, f"fig_{o}_pec")

    # ---- Fig 3b: time-dependent ROC curves (cumulative/dynamic, IPCW) at 5 y & 10 y
    test = pd.read_parquet(f"{DATA_DIR}/test_imputed.parquet")
    y_te = make_y(test, o)
    G_km = censoring_km_train(make_y(pd.read_parquet(f"{DATA_DIR}/train_imputed.parquet"), o))
    roc = compute_roc_curves(o, G_km, y_te["time"], y_te["event"], test["id"])
    for h in HORIZONS:
        rh = roc[roc.horizon_y == h]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharex=True, sharey=True)
        for ai, a in enumerate(ALGOS):
            axi = axes[ai]
            axi.plot([0, 1], [0, 1], "k--", lw=0.7)
            for l in LEVELS:
                d = rh[rh.model == tag(o, a, l)]
                rm = m[(m.algo == a) & (m.level == l)].iloc[0]
                axi.plot(d.fpr, d.tpr, color=LEVEL_COLORS[l], lw=1.4,
                         label=f"M{l}: {rm[f'AUC_{h}y']:.3f} ({rm[f'AUC_{h}y_lo']:.3f}–{rm[f'AUC_{h}y_hi']:.3f})")
            axi.set_title(ALGO_LABELS[a], fontsize=9)
            axi.set_xlabel("1 − specificity")
            axi.set_xlim(0, 1); axi.set_ylim(0, 1.005)
            axi.legend(fontsize=6.5, loc="lower right", handlelength=1.4, labelspacing=0.3)
        axes[0].set_ylabel("Sensitivity")
        fig.suptitle(f"{OUTCN[o]}: time-dependent ROC at {h} years (test set)")
        savefig(fig, f"fig_{o}_roc_{h}y")

    # ---- Fig 4: DCA at 5 y & 10 y (computed from cached survival matrices)
    y_te_ = make_y(test, o)
    for h in HORIZONS:
        dca = compute_dca(o, y_te_["time"], y_te_["event"], h)
        dca.to_csv(f"{RESULTS_DIR}/dca_curves_{o}_{h}y.csv", index=False)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
        for ai, a in enumerate(ALGOS):
            axi = axes[ai]
            for l in LEVELS:
                d = dca[(dca.model == f"{a}_m{l}")].dropna(subset=["net_benefit"])
                axi.plot(d.threshold * 100, d.net_benefit * 100, color=LEVEL_COLORS[l], ls=LEVEL_LSDASH[l],
                         lw=1.4, label=f"M{l}")
            for nm, cc in [("treat_all", "black"), ("treat_none", "grey")]:
                d = dca[dca.model == nm]
                axi.plot(d.threshold * 100, d.net_benefit * 100, color=cc, lw=0.8, ls="-." if nm == "treat_all" else ":",
                         label="Treat all" if nm == "treat_all" else "Treat none")
            axi.set_title(ALGO_LABELS[a], fontsize=9)
            axi.set_xlabel("Threshold probability (%)")
            axi.legend(fontsize=7, loc="upper right")
            axi.set_ylim(-5, 8)
        axes[0].set_ylabel("Net benefit (%)")
        fig.suptitle(f"{OUTCN[o]}: decision curve analysis at {h} years (test set)")
        savefig(fig, f"fig_{o}_dca_{h}y")

    # ---- Fig 5: incremental C-index
    fig, ax = plt.subplots(figsize=(7, 4))
    inc = incs[o]
    width = 0.18
    for li, (lv, comp) in enumerate([(2, "model2 - model1"), (3, "model3 - model1")]):
        for ai, a in enumerate(ALGOS):
            d = inc[(inc.algo == a) & (inc.comparison == comp) & (inc.metric == "C_harrell")].iloc[0]
            x = ai + (li - 0.5) * width * 2.2
            ax.errorbar(x, d["delta"] * 100, yerr=[[ (d["delta"] - d.delta_lo) * 100], [(d.delta_hi - d["delta"]) * 100]],
                        fmt="o", color=LEVEL_COLORS[lv], capsize=2.5, ms=5)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xticks(range(len(ALGOS))); ax.set_xticklabels([ALGO_LABELS[a] for a in ALGOS], fontsize=8)
    ax.set_ylabel("Δ C-index vs model-1 (×100)")
    ax.set_title(f"{OUTCN[o]}: incremental discrimination (test set)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=LEVEL_COLORS[2], marker="o", ls="", label="Model 2 − Model 1"),
                       Line2D([], [], color=LEVEL_COLORS[3], marker="o", ls="", label="Model 3 − Model 1")])
    savefig(fig, f"fig_{o}_delta_C")

    # ---- Fig 6: SHAP (XGBoost model-3, exact TreeSHAP via pred_contribs)
    try:
        imp_shap, X_raw, shap_vals = shap_section(o, test, levels3, scaler_pack)
        shap_n[o] = len(shap_vals)
        plot_shap_figures(o, imp_shap, X_raw, shap_vals, levels3)
        shap_top5[o] = ", ".join(f"{r.feature} ({r.mean_abs_shap:.4f})" for _, r in imp_shap.head(5).iterrows())
        ob = imp_shap[imp_shap.feature.str.startswith(f"{OBESITY}_")]
        shap_ob_rank[o] = int(ob.index.min()) + 1
        print(f"{o}: SHAP done — top5 = {list(imp_shap.feature[:5])}, obesity best rank = {shap_ob_rank[o]}")
    except Exception as e:
        print(f"SHAP skipped for {o}: {e}")

    # ---- Fig 7: KM curves by XGBoost model-3 predicted-risk tertiles
    try:
        km_section(o, test)
    except Exception as e:
        print(f"KM skipped for {o}: {e}")

# ================================================================ REPORT
def fmt(x, nd=3):
    return "—" if pd.isna(x) else f"{x:.{nd}f}"

FIGLINKS = [("C-index (by algorithm)", "fig_{o}_C_index"),
            ("Calibration (10 y)", "fig_{o}_calibration_10y"),
            ("Calibration (5 y)", "fig_{o}_calibration_5y"),
            ("Time-dependent ROC (10 y)", "fig_{o}_roc_10y"),
            ("Time-dependent ROC (5 y)", "fig_{o}_roc_5y"),
            ("KM curves (XGBoost M3 risk tertiles)", "fig_{o}_km_xgb_m3"),
            ("Prediction-error curves", "fig_{o}_pec"),
            ("Decision curves (10 y)", "fig_{o}_dca_10y"),
            ("Decision curves (5 y)", "fig_{o}_dca_5y"),
            ("Incremental C-index", "fig_{o}_delta_C"),
            ("SHAP importance (XGBoost M3)", "fig_{o}_shap_importance"),
            ("SHAP beeswarm (XGBoost M3)", "fig_{o}_shap_beeswarm")]

L = []
L.append("# Incremental value of obesity and blood biomarkers for psychiatric-risk prediction\n")
L.append(f"Data: `{RAW_CSV}` (n = {len(pd.read_parquet(f'{DATA_DIR}/test_imputed.parquet')) + len(pd.read_parquet(f'{DATA_DIR}/train_imputed.parquet')):,})\n")
L.append("## Methods\n")
L.append("- **Design**: 80/20 train-test split (stratified on the joint outcome event pattern, "
         "random_state=42). Missing data imputed MICE-style (`IterativeImputer`, BayesianRidge, "
         "max_iter=15); imputer and standardizer fitted on the training set only, then applied "
         "to the test set.")
L.append(f"- **Predictor levels**: Model 1 = baseline covariates; Model 2 = + obesity "
         f"({OBESITY}, one-hot); Model 3 = + {len(BIOMARKERS)} blood biomarkers. Multi-level "
         "categoricals one-hot encoded (first level as reference); continuous variables "
         "standardized.")
L.append("- **Algorithms**: Cox PH; elastic-net Cox (l1_ratio=0.5, alpha by 5-fold CV on the "
         "C-index); random survival forest (400 trees, max_features=sqrt, min_samples_leaf=200 "
         "by sensitivity analysis, 50% subsample per tree, time axis discretized at 0.1 y); "
         "XGBoost (survival:cox, early stopping; absolute risk recalibrated by a Cox model on "
         "the risk score using a normal-scores transform).")
L.append("- **Evaluation (test set)**: Harrell C, Uno C, time-dependent AUC and ROC curves "
         "(5/10 y, cumulative/dynamic, IPCW), Brier score (5/10 y), integrated Brier score "
         "IBC(0.25-10 y), calibration (deciles, O/E, slope, ICI), clinical utility (net "
         "benefit / DCA). Increments = delta metric with paired bootstrap 95% CI (B=300) plus "
         "NRI/IDI (continuous, IPCW); likelihood-ratio tests for Cox PH on the training set.\n")

for o in OUTCOMES:
    m = tables[o]; inc = incs[o]; lrt = lrts[o]; nr = nris[o]
    L.append(f"\n## {OUTCN[o]}\n")
    ev = int(m["events_test"].iloc[0]); nn = int(m["n_test"].iloc[0])
    L.append(f"Test set n = {nn:,}, events {ev:,} ({ev/nn*100:.2f}%).")
    L.append("### Main metrics\n")
    hdr = ["Algorithm", "Level", "C-index (95% CI)", "AUC 5y", "AUC 10y", "Brier 5y", "Brier 10y",
           "IBC 0.25-10y", "O/E 10y", "Cal. slope 10y", "ICI 10y"]
    L.append("| " + " | ".join(hdr) + " |"); L.append("|" + "---|" * len(hdr))
    for a in ALGOS:
        for l in LEVELS:
            r = m[(m.algo == a) & (m.level == l)].iloc[0]
            L.append(f"| {ALGO_LABELS[a]} | M{l} | {r.C_harrell:.4f} ({r.C_harrell_lo:.4f}-{r.C_harrell_hi:.4f}) | "
                     f"{fmt(r.AUC_5y)} | {fmt(r.AUC_10y)} | {fmt(r.Brier_5y, 4)} | {fmt(r.Brier_10y, 4)} | "
                     f"{fmt(r['IBC_0.25-10y'], 4)} | {fmt(r['cal_OE_10y'], 2)} | {fmt(r['cal_slope_10y'], 2)} | {fmt(r['cal_ICI_10y'], 4)} |")
    L.append("\n### Increments (vs model 1)\n")
    hdr2 = ["Algorithm", "Comparison", "ΔC-index (95% CI), ×100", "p", "ΔAUC10 ×100", "ΔBrier10 ×100", "ΔIBC ×1000", "NRI", "IDI"]
    L.append("| " + " | ".join(hdr2) + " |"); L.append("|" + "---|" * len(hdr2))
    for a in ALGOS:
        for comp in ["model2 - model1", "model3 - model1", "model3 - model2"]:
            def g(metric):
                d = inc[(inc.algo == a) & (inc.comparison == comp) & (inc.metric == metric)]
                return d.iloc[0] if len(d) else None
            dc, da, db, di = g("C_harrell"), g("AUC_10y"), g("Brier_10y"), g("IBC_0.25-10y")
            nr_ = nr[(nr.algo == a) & (nr.comparison == comp)]
            nri, idi = (fmt(nr_.NRI_continuous.iloc[0], 3), fmt(nr_.IDI.iloc[0], 4)) if len(nr_) else ("—", "—")
            L.append(f"| {ALGO_LABELS[a]} | {comp.replace('model', 'M').replace(' - ', '-')} | "
                     f"{dc['delta']*100:.2f} ({dc.delta_lo*100:.2f}-{dc.delta_hi*100:.2f}) | {fmt(dc.p_bootstrap, 3)} | "
                     f"{da['delta']*100:.2f} | {db['delta']*100:.2f} | {di['delta']*1000:.2f} | {nri} | {idi} |")
    lr = lrt[lrt.comparison != "model3 vs model2"]
    L.append("\nCox PH likelihood-ratio tests (training set): " + "; ".join(
        f"{r.comparison}: chi2={r.chi2:.0f}, df={r.df:.0f}, p={r.p:.2e}" for _, r in lr.iterrows()))
    L.append("\n### Figures (PDF)\n")
    if o in shap_top5:
        L.append(f"**SHAP (XGBoost model 3, random test-set subsample n = {shap_n.get(o, '—')}, exact TreeSHAP)**: "
                 f"top-5 features by mean|SHAP|: {shap_top5[o]}; best rank of the obesity indicators: "
                 f"#{shap_ob_rank[o]} (SHAP values are log relative hazard; positive = higher risk).\n")
    for label, pat in FIGLINKS:
        L.append(f"- [{label}](figures/{pat.format(o=o)}.pdf)")

L.append("\n## Output files\n")
L.append("- `results/tables_combined.csv` — main metrics for all outcomes × algorithms × levels")
L.append("- `results/increments_*.csv` — increment metrics (ΔC, ΔAUC, ΔBrier, ΔIBC with bootstrap CI and p)")
L.append("- `results/lrt_*.csv` — Cox likelihood-ratio tests; `results/nri_idi_*.csv` — NRI/IDI; "
         "`results/dca_*.csv` — net-benefit curves; `results/calibration_*.csv` — calibration tables; "
         "`results/roc_curves_*.csv` — time-dependent ROC data; `results/coef_*.csv` — Cox / elastic-net coefficients")
L.append("- `figures/` — all figures (vector PDF); `models/` — fitted models; `data/` — imputed "
         "train/test sets and test-set predictions")
L.append("- Scripts: `6-1.prepare_data.py` → `6-2.fit_models.py` (per outcome) → "
         "`6-3.metrics.py` (per outcome) → `6-4.figures_report.py` (ROC curves, figures, report)")
with open(f"{OUT}/REPORT.md", "w") as fh:
    fh.write("\n".join(L))
print("DONE 04 — figures (PDF) & REPORT.md")
