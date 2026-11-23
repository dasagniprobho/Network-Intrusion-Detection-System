"""
Streamlit demo of NIDS_model_final.ipynb (CIC-IDS2017 network intrusion detection, leakage-safe revision 2).

Run:  streamlit run app.py
The pipeline code lives in nids_core.py (a faithful port of the notebook cells); this file is only the UI.
"""
import glob
import importlib.util
import io
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import nids_core as nc

st.set_page_config(page_title="NIDS demo - CIC-IDS2017", page_icon="🛡️", layout="wide")
HAS_XGB = importlib.util.find_spec("xgboost") is not None
SS = st.session_state
MIN_OOF_GAIN = 0.003


def ss(key, default=None):
    return SS.get(key, default)


# --------------------------------------------------------------------------------------
# Small plotting helpers
# --------------------------------------------------------------------------------------
def plot_label_bar(counts: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(counts))))
    counts.sort_values().plot.barh(ax=ax, logx=True)
    ax.set_xlabel("flows (log scale)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_confusion(res: dict, classes: list, title: str):
    cm = res["confusion"]
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.heatmap(norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1, cbar=False, xticklabels=classes,
                yticklabels=classes, annot_kws={"size": 8}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title + "  (colour = recall, numbers = flows)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def show_pyplot(fig):
    st.pyplot(fig)
    plt.close(fig)


def need(condition: bool, message: str) -> bool:
    if not condition:
        st.info(message)
    return condition


def need_xgb() -> bool:
    if not HAS_XGB:
        st.error("`xgboost` is not installed. Run `pip install xgboost` and restart the app to train models.")
    return HAS_XGB


# --------------------------------------------------------------------------------------
# Data loading + preparation (cleaning, engineering, the one train/test split)
# --------------------------------------------------------------------------------------
def reset_downstream():
    for k in ["fs", "models", "port_res", "val_leak", "val_iso", "val_loso", "bundle_bytes", "uploaded_bundle", "live_out"]:
        SS.pop(k, None)


def load_and_prepare(source: str, rows: int, seed: int, files, folder: str, cap: int):
    status = st.status("Loading and preparing data ...", expanded=True)
    with status:
        t0 = time.time()
        if source == "Synthetic demo data":
            st.write(f"Generating {rows:,} synthetic CIC-IDS2017-shaped flows ...")
            raw = nc.make_synthetic(rows, seed)
        elif source == "Upload CIC-IDS2017 CSVs":
            if not files:
                status.update(label="No files selected", state="error")
                return
            parts = []
            for f in files:
                st.write(f"Reading {f.name} ...")
                parts.append(nc.read_cic_csv(f, f.name, cap or None))
            raw = nc.combine_parts(parts)
        else:
            paths = sorted(glob.glob(os.path.join(folder.strip(), "*.csv")))
            if not paths:
                status.update(label=f"No CSV files found in '{folder}'", state="error")
                return
            parts = []
            for p in paths:
                st.write(f"Reading {os.path.basename(p)} ...")
                parts.append(nc.read_cic_csv(p, p, cap or None))
            raw = nc.combine_parts(parts)

        reset_downstream()
        raw_labels = raw["Label"].str.replace(r"[^\x00-\x7F]+", "-", regex=True).str.strip()
        SS.raw_info = {
            "shape": raw.shape, "preview": raw.head(8), "label_counts": raw_labels.value_counts(),
            "session_table": pd.crosstab(raw["session"], raw_labels),
            "top_ports": raw["Destination Port"].astype("int64").value_counts().head(15) if "Destination Port" in raw else None,
            "source": source,
        }
        st.write("Cleaning (inf -> NaN, constant columns, invalid rows, exact duplicates) ...")
        clean, rep = nc.clean_raw(raw)
        SS.integrity = nc.integrity_checks(clean)          # as in the notebook: checked right after cleaning, before engineering
        del raw
        st.write("Feature engineering (sentinel flags, Active/Idle flags, port groups, label grouping) ...")
        df = nc.engineer(clean)
        del clean
        st.write("Stratified 80/20 split, made once, before any supervised feature analysis ...")
        train_idx, test_idx = nc.make_split(df)
        SS.rep, SS.df, SS.train_idx, SS.test_idx = rep, df, train_idx, test_idx
        SS.models = {}
        status.update(label=f"Ready: {len(df):,} clean flows ({time.time() - t0:.1f}s)", state="complete")


# --------------------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------------------
NN_ARCHS = {"128-64, dropout 0.3 (fast)": ((128, 64), 0.3), "256-128, dropout 0.3 (notebook default)": ((256, 128), 0.3),
            "512-256-128, dropout 0.2": ((512, 256, 128), 0.2)}


def get_cfgs():
    xgb_cfg = dict(wpow=ss("x_wpow", 0.5), max_depth=ss("x_depth", 8), mcw=ss("x_mcw", 1), lr=ss("x_lr", 0.1), max_rounds=ss("x_rounds", 400))
    hidden, drop = NN_ARCHS[ss("n_arch", "128-64, dropout 0.3 (fast)")]
    nn_cfg = dict(wpow=ss("n_wpow", 0.1), hidden=hidden, dropout=drop, epochs=ss("n_epochs", 25))
    return xgb_cfg, nn_cfg


def run_training(setup: str, use_nn: bool, tune: bool, status=None):
    fs, df = SS.fs, SS.df
    num_cols, cat_cols = nc.setup_columns(setup, fs["numeric"], fs["categorical"])
    cols = list(dict.fromkeys(num_cols + cat_cols))
    X, y = df[cols], df["Label_grouped"]
    classes = sorted(y.unique())
    xgb_cfg, nn_cfg = get_cfgs()
    log = (lambda m: status.write(m)) if status is not None else (lambda m: None)
    t = nc.train_setup(X.loc[SS.train_idx], y.loc[SS.train_idx], X.loc[SS.test_idx], y.loc[SS.test_idx], num_cols, cat_cols, classes,
                       pool_cap=int(ss("pool_cap", 20_000)), xgb_cfg=xgb_cfg, nn_cfg=nn_cfg, use_nn=use_nn,
                       n_seeds=int(ss("n_seeds", 1)), ens_w=float(ss("ens_w", 0.5)), tune=tune, cv_folds=int(ss("cv_folds", 3)), log=log)
    # headline choice: thresholds must earn their place on out-of-fold data (as in the notebook)
    ti = t["tune_info"]
    base = "Ensemble" if use_nn else "XGBoost"
    headline = base
    if tune and ti is not None:
        base_oof = ti["ens_scores"][ti["ens_w"]] if use_nn else ti["oof_macro_f1_xgb"]
        if ti["oof_macro_f1_thresholds"] - base_oof >= MIN_OOF_GAIN:
            headline = "Ensemble + class thresholds" if use_nn else "XGBoost + class thresholds"
    if "thresholds" not in headline:
        t["class_scales"] = np.ones(len(classes), dtype=np.float32)
    t["setup"], t["headline"] = setup, headline
    return t


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ NIDS demo")
    st.caption("Network intrusion detection on CIC-IDS2017 - the notebook's pipeline, step by step.")
    source = st.radio("Data source", ["Synthetic demo data", "Upload CIC-IDS2017 CSVs", "Folder on this machine"], key="source")
    files, folder, rows, seed, cap = None, "", 60_000, 42, 20_000
    if source == "Synthetic demo data":
        rows = st.slider("Flows to generate", 20_000, 300_000, 60_000, step=10_000)
        seed = st.number_input("Random seed", 0, 10_000, 42)
        st.caption("Synthetic data has the real column names, 15 labels and 8 sessions, so every step runs without the "
                   "dataset. It is **not real traffic**: scores here are not CIC-IDS2017 results.")
    else:
        if source == "Upload CIC-IDS2017 CSVs":
            files = st.file_uploader("The 8 original CSV files", type=["csv"], accept_multiple_files=True)
        else:
            folder = st.text_input("Folder containing the 8 CSV files", value="Network_Intrusion_Dataset")
        cap = st.number_input("Max flows kept per label per file (0 = keep all)", 0, 2_000_000, 20_000, step=5_000,
                              help="The full dataset is ~2.8M rows. Capping keeps memory modest; rare classes stay whole. "
                                   "It changes class balance versus the notebook's full run.")
    if st.button("Load & prepare data", type="primary"):
        load_and_prepare(source, rows, int(seed), files, folder, int(cap))
    if "df" in SS:
        st.success(f"{len(SS.df):,} flows ready")
    if not HAS_XGB:
        st.warning("xgboost missing - training tabs disabled")

# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
tabs = st.tabs(["Overview", "1 · Data", "2 · Cleaning", "3 · Features", "4 · Train & evaluate",
                "5 · Port test & baseline", "6 · Validation (Rev 2)", "7 · Live detection"])
t_over, t_data, t_clean, t_feat, t_train, t_port, t_val, t_live = tabs

# ---------------------------------- Overview ------------------------------------------
with t_over:
    st.header("Network intrusion detection on CIC-IDS2017")
    st.write(
        "This app runs the notebook's pipeline interactively. Each tab is one stage; the stages feed each other, "
        "so go left to right. Start by pressing **Load & prepare data** in the sidebar (the synthetic option needs no download).")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontsize=11];
      raw [label="8 raw CSVs\\n(15 labels, 8 sessions)"]; clean [label="Cleaning\\ninf/NaN, constants,\\nbad rows, duplicates"];
      eng [label="Engineering\\nsentinel flags, Active/Idle,\\nport groups, 7 classes"];
      split [label="ONE stratified 80/20 split\\n(before feature analysis)"]; fs [label="Feature selection\\nSpearman 0.95 -> VIF 10\\n(training rows only)"];
      pre [label="Preprocessing\\nimpute -> signed log1p -> RobustScaler\\n+ one-hot"]; mdl [label="Models\\nXGBoost + NumPy MLP\\n+ ensemble + thresholds"];
      ev [label="Evaluation\\nMacro-F1, port test, session-held-out"]; live [label="Live detection\\nverdict, family, confidence,\\nSHAP top-3"];
      raw -> clean -> eng -> split -> fs -> pre -> mdl -> ev -> live; }
    """)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What the notebook does")
        st.markdown(
            "- **Cleans** the 2.83M-row dataset (inf/NaN, constant columns, negative durations, exact duplicates).\n"
            "- **Engineers** `forward/backward_unknown` (the `-1` window sentinel), Active/Idle presence flags and a "
            "`Destination_Port_Group`, and merges 15 labels into **7 classes**.\n"
            "- **Splits once**, then runs all supervised feature analysis on training rows only (Rev 2 leakage fix).\n"
            "- **Preprocesses inside every CV fold** (imputer, log, scaler, encoder fitted on the fold's training part).\n"
            "- Trains **XGBoost**, a regularised **NumPy MLP**, an **ensemble** and per-class **thresholds**.\n"
            "- Tests the **port shortcut** (A with port vs B without vs G port-group only) and a session-held-out score.")
    with c2:
        st.subheader("What this demo simplifies")
        st.markdown(
            "- Works on a **per-label capped sample** (or synthetic data) instead of all 2.83M rows.\n"
            "- The full hyper-parameter grid is replaced by **sliders** (defaults are the notebook's starting values); "
            "fold-safe OOF tuning of the ensemble weight and class multipliers is available as a checkbox.\n"
            "- VIF is computed from the inverse correlation matrix (same numbers as per-column regressions, much faster).\n"
            "- Fewer seeds and smaller pools by default, so a run takes a minute or two.")
    st.caption("Class taxonomy: BENIGN | DoS_DDoS (DDoS, Hulk, GoldenEye, slowloris, Slowhttptest) | Reconnaissance (PortScan) | "
               "Brute_Force (FTP/SSH-Patator, web brute force) | Web_Attack (XSS, SQLi) | Bot | Other_Rare (Heartbleed, Infiltration)")

# ---------------------------------- 1 Data --------------------------------------------
with t_data:
    st.header("1 · Raw data")
    if need("raw_info" in SS, "Load data from the sidebar first."):
        info = SS.raw_info
        a, b, c = st.columns(3)
        a.metric("Raw rows", f"{info['shape'][0]:,}")
        b.metric("Raw columns", info["shape"][1])
        c.metric("Source", info["source"])
        st.subheader("First rows")
        st.dataframe(info["preview"])
        l, r = st.columns(2)
        with l:
            show_pyplot(plot_label_bar(info["label_counts"], "Label distribution (raw)"))
        with r:
            st.dataframe(info["label_counts"].to_frame("flows"))
        st.subheader("Which classes occur in which capture session?")
        st.dataframe(info["session_table"])
        st.caption("Most attack classes live in a single session, so session and attack type are confounded. "
                   "That is why Rev 2 adds the session-held-out evaluation (tab 6).")
        if info["top_ports"] is not None:
            st.subheader("Top destination ports")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(info["top_ports"].index.astype(str), info["top_ports"].values)
            ax.set_ylabel("flows")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            show_pyplot(fig)

# ---------------------------------- 2 Cleaning ----------------------------------------
with t_clean:
    st.header("2 · Cleaning")
    if need("rep" in SS, "Load data from the sidebar first."):
        rep = SS.rep
        cols = st.columns(5)
        cols[0].metric("Rows before", f"{rep['rows_start']:,}")
        cols[1].metric("Rows after", f"{rep['rows_end']:,}", delta=f"-{rep['rows_start'] - rep['rows_end']:,}")
        cols[2].metric("inf values", f"{rep['inf_values']:,}")
        cols[3].metric("Invalid rows dropped", f"{rep['bad_rows_union']:,}")
        cols[4].metric("Exact duplicates", f"{rep['duplicates_before']:,}")
        st.markdown(
            f"- `inf` becomes `NaN`; rows with `NaN` in *Flow Bytes/s* / *Flow Packets/s* ({rep['nan_rows']:,}) or a negative value in a column "
            f"that cannot be negative ({rep['negative_rows']:,}) are dropped (union {rep['bad_rows_union']:,}).\n"
            f"- **{len(rep['constant_columns'])} constant columns** removed" + (", plus the duplicated `Fwd Header Length.1`"
                                                                              + (" (identical to `Fwd Header Length`)" if rep.get("dup_header_identical") else "")
                                                                              if rep["dup_header_col"] else "") + ".\n"
            "- Exact duplicate rows are removed after the bad rows (session column ignored).")
        with st.expander("Constant columns removed"):
            st.write(rep["constant_columns"])
        with st.expander("NaN / negative counts per column (before cleaning)"):
            st.dataframe(pd.DataFrame({"NaN after inf->NaN": pd.Series(rep["nan_by_column"]), "negative": pd.Series(rep["negative_by_column"])}).fillna(0).astype(int))
        st.subheader("Rows per label")
        st.dataframe(rep["label_table"])
        st.subheader("Dataset-integrity checks on the cleaned data")
        chk = SS.integrity
        st.dataframe(chk)
        if chk["passed"].all():
            st.success("All integrity checks pass.")
        else:
            st.error("An integrity check failed.")

# ---------------------------------- 3 Features ----------------------------------------
with t_feat:
    st.header("3 · Feature engineering, split and selection")
    if need("df" in SS, "Load data from the sidebar first."):
        df = SS.df
        st.subheader("Engineered features")
        st.markdown(
            "- `forward_unknown` / `backward_unknown`: set where `Init_Win_bytes_*` was the sentinel **-1** (flag created *before* -1 becomes NaN).\n"
            "- `active_*_present` / `idle_*_present`: 1 where the Active/Idle statistic is > 0 (most flows have none).\n"
            "- `Destination_Port_Group`: service name for common ports, else well-known / registered / dynamic.")
        eng_cols = [c for c in ["Destination Port", "Destination_Port_Group", "Init_Win_bytes_forward", "forward_unknown", "backward_unknown",
                                "active_mean_present", "idle_mean_present", "Label", "Label_grouped"] if c in df.columns]
        st.dataframe(df[eng_cols].sample(8, random_state=1))
        l, r = st.columns(2)
        with l:
            show_pyplot(plot_label_bar(df["Label_grouped"].value_counts(), "Grouped classes (7)"))
        with r:
            st.dataframe(pd.crosstab(df["Label"], df["Label_grouped"]))

        st.subheader("The single train/test split")
        a, b = st.columns(2)
        a.metric("Training rows", f"{len(SS.train_idx):,}")
        b.metric("Test rows (untouched until final evaluation)", f"{len(SS.test_idx):,}")
        st.caption("Stratified on the original 15-class label so even Heartbleed appears on both sides. Everything below uses training rows only.")

        st.subheader("Spearman -> VIF feature selection (training rows only)")
        c1, c2, c3 = st.columns(3)
        sp_thr = c1.slider("Spearman threshold", 0.80, 0.99, 0.95, step=0.01)
        vif_thr = c2.slider("VIF threshold", 3.0, 30.0, 10.0, step=1.0)
        sp_n = c3.number_input("Spearman sample rows", 10_000, 200_000, 100_000, step=10_000)
        if st.button("Run feature selection"):
            with st.spinner("Spearman correlation and iterative VIF removal ..."):
                df_tr = df.loc[SS.train_idx]
                sp = nc.spearman_prune(df_tr, threshold=sp_thr, sample=int(sp_n))
                vf = nc.vif_prune(df_tr, sp["remaining"], threshold=vif_thr)
                numeric, categorical = nc.feature_lists(vf["selected"], df.columns)
                SS.fs = {"sp": sp, "vf": vf, "numeric": numeric, "categorical": categorical}
                SS.pop("models", None)
                SS.models = {}
        if "fs" in SS:
            fs = SS.fs
            m = st.columns(4)
            m[0].metric("Numeric candidates", len(fs["sp"]["corr_features"]))
            m[1].metric("After Spearman", len(fs["sp"]["remaining"]))
            m[2].metric("After VIF", len(fs["vf"]["selected"]))
            m[3].metric("Model inputs (incl. flags + categorical)", len(fs["numeric"]) + len(fs["categorical"]))
            fig, ax = plt.subplots(figsize=(7, 5.5))
            sns.heatmap(fs["sp"]["corr"], cmap="coolwarm", center=0, vmin=-1, vmax=1, xticklabels=False, yticklabels=False, ax=ax)
            ax.set_title("Spearman correlation (training sample)")
            fig.tight_layout()
            show_pyplot(fig)
            l, r = st.columns(2)
            with l:
                st.markdown("**Removed by Spearman** (|rho| above threshold; the one with higher mean correlation goes)")
                st.dataframe(pd.DataFrame(fs["sp"]["removed"], columns=["removed", "|rho|", "paired with"]))
            with r:
                st.markdown("**Removed by VIF** (largest first, iteratively)")
                st.dataframe(pd.DataFrame(fs["vf"]["removed"], columns=["removed", "VIF"]))
            with st.expander("Final numeric features and VIFs"):
                st.dataframe(fs["vf"]["final"])
            st.write("**Categorical inputs:**", fs["categorical"], "| **Raw Destination Port is excluded** from setups B and G.")

# ---------------------------------- 4 Train -------------------------------------------
with t_train:
    st.header("4 · Preprocessing, models and evaluation")
    ready = need("fs" in SS, "Run feature selection in tab 3 first.")
    if ready and need_xgb():
        st.markdown("Preprocessing is fitted on the **training pool only**, then applied to the test set: median imputation → signed log1p → "
                    "RobustScaler for numbers, most-frequent imputation → one-hot for categories. Class weights are `(n_max / n_class) ** power`; "
                    "boosting rounds come from early stopping on an inner 10% split.")
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Feature setup", [nc.SETUP_B, nc.SETUP_G, nc.SETUP_A], format_func=lambda s: nc.SETUP_TEXT[s], key="tr_setup",
                         help="A = with port, B = no port information, G = port group only. B is the notebook's strict headline setup.")
            st.radio("Models", ["XGBoost only", "XGBoost + neural net (ensemble)"], key="tr_models")
            st.number_input("Training pool: max flows per class", 1_000, 200_000, 20_000, step=1_000, key="pool_cap",
                            help="The notebook uses 100,000. Rare classes are always kept whole.")
            st.slider("Seeds averaged", 1, 3, 1, key="n_seeds")
        with c2:
            with st.expander("XGBoost settings", expanded=False):
                st.slider("Class-weight power", 0.0, 1.0, 0.5, step=0.05, key="x_wpow")
                st.slider("max_depth", 3, 12, 8, key="x_depth")
                st.slider("min_child_weight", 1, 10, 1, key="x_mcw")
                st.select_slider("learning rate", [0.05, 0.1, 0.2, 0.3], value=0.1, key="x_lr")
                st.slider("max boosting rounds (early stopping applies)", 50, 600, 400, step=50, key="x_rounds")
            with st.expander("Neural-net and ensemble settings", expanded=False):
                st.selectbox("Architecture", list(NN_ARCHS), key="n_arch")
                st.slider("NN class-weight power", 0.0, 0.5, 0.1, step=0.05, key="n_wpow")
                st.slider("NN max epochs", 5, 60, 25, key="n_epochs")
                st.slider("XGBoost share in the ensemble (when not tuned)", 0.0, 1.0, 0.5, step=0.25, key="ens_w")
            st.checkbox("Tune ensemble weight + per-class thresholds on fold-safe out-of-fold predictions", value=False, key="tr_tune",
                        help="Preprocessing is re-fitted inside every CV fold (Rev 2). Slower: runs k-fold training first.")
            st.slider("CV folds (if tuning)", 3, 5, 3, key="cv_folds")

        if st.button("Train and evaluate", type="primary", key="btn_train"):
            setup, use_nn, tune = ss("tr_setup", nc.SETUP_B), ss("tr_models") != "XGBoost only", bool(ss("tr_tune", False))
            status = st.status(f"Training {setup} ...", expanded=True)
            with status:
                t0 = time.time()
                SS.models[setup] = run_training(setup, use_nn, tune, status)
                status.update(label=f"Done in {time.time() - t0:.0f}s", state="complete")

        trained = list(SS.get("models", {}))
        if trained:
            pick = st.selectbox("Show results for", trained, format_func=lambda s: nc.SETUP_TEXT[s], key="show_setup")
            t = SS.models[pick]
            res = t["results"]
            classes = t["class_names"]
            st.subheader(f"Test-set results ({pick})  -  headline model: {t['headline']}")
            comp = pd.DataFrame({k: {"Macro-F1": v["macro_f1"], "accuracy (reference only)": v["accuracy"],
                                     **nc.binary_view(SS.df.loc[SS.test_idx, "Label_grouped"].to_numpy(), v["y_pred"])} for k, v in res.items()}).T
            st.dataframe(comp.round(4))
            st.caption(f"Pool: {t['pool_size']:,} flows | XGBoost rounds: {t['n_rounds']} | ensemble XGBoost share: {t['xgb_weight']:.2f}. "
                       "Models are ranked on Macro-F1 because accuracy is dominated by BENIGN.")
            variant = st.selectbox("Variant to inspect", list(res), index=list(res).index(t["headline"]), key="show_variant")
            l, r = st.columns([1, 1])
            with l:
                st.markdown("**Per-class precision / recall / F1**")
                st.dataframe(res[variant]["per_class"].round(4))
            with r:
                show_pyplot(plot_confusion(res[variant], classes, f"{variant} | {pick}"))
            if t["tune_info"]:
                with st.expander("Out-of-fold tuning details"):
                    ti = t["tune_info"]
                    st.write({k: v for k, v in ti.items() if k not in ("cv_rounds",)})

# ---------------------------------- 5 Port test & baseline ----------------------------
with t_port:
    st.header("5 · Port-shortcut test and the logistic baseline")
    st.markdown("Can the model lean on the destination port as a shortcut (for example port 80 → DoS)? "
                "The same model and settings are trained on three column sets: **A** with the raw port, **B** with no port information, "
                "**G** with the port group only. A large drop from A to B means the port was carrying the score. "
                "The progression table then walks from the logistic baseline (setup G columns) to the tree model.")
    if need("fs" in SS, "Run feature selection in tab 3 first.") and need_xgb():
        with_nn = st.checkbox("Also train the neural net and the ensemble (slower)", value=False, key="port_nn")
        if st.button("Run A vs B vs G", key="btn_port"):
            status = st.status("Training three feature setups ...", expanded=True)
            with status:
                out = {}
                for s in (nc.SETUP_A, nc.SETUP_B, nc.SETUP_G):
                    status.write(f"Setup {s}")
                    out[s] = run_training(s, with_nn, False, None)
                SS.port_res = out
                SS.port_res_nn = with_nn
                status.update(label="Done", state="complete")
        if "port_res" in SS:
            pr = SS.port_res
            variants = list(pr[nc.SETUP_B]["results"])
            classes = pr[nc.SETUP_B]["class_names"]
            rows_ = []
            for v in variants:
                for s in (nc.SETUP_A, nc.SETUP_B, nc.SETUP_G):
                    r_ = pr[s]["results"][v]
                    rows_.append({"model": v, "setup": s, "Macro-F1": r_["macro_f1"], **{f"recall {c}": r_["per_class"].loc[c, "recall"] for c in classes}})
            tbl = pd.DataFrame(rows_).set_index(["model", "setup"])
            st.subheader("Macro-F1 and per-class recall")
            st.dataframe(tbl.round(4))
            delta = pd.DataFrame([tbl.loc[(v, nc.SETUP_B)] - tbl.loc[(v, nc.SETUP_A)] for v in variants], index=pd.Index(variants, name="model (B minus A)"))
            st.subheader("Change from removing the port (B - A)")
            st.dataframe(delta.round(4))
            gap = float(delta["Macro-F1"].iloc[0])
            if gap < -0.10:
                st.error(f"Macro-F1 drops by {abs(gap):.3f} without the port: the port is acting as a shortcut.")
            else:
                st.success(f"Macro-F1 changes by {gap:+.3f} without the port: no collapse (the notebook calls a drop larger than 0.10 a collapse).")
            fig, ax = plt.subplots(figsize=(7, 3))
            xs = np.arange(len(variants))
            for k, s in enumerate((nc.SETUP_A, nc.SETUP_B, nc.SETUP_G)):
                ax.bar(xs + 0.27 * (k - 1), [tbl.loc[(v, s), "Macro-F1"] for v in variants], width=0.27, label=nc.SETUP_TEXT[s].split(" - ")[0])
            ax.set_xticks(xs)
            ax.set_xticklabels(variants)
            ax.set_ylabel("Macro-F1")
            ax.legend(title="setup")
            fig.tight_layout()
            show_pyplot(fig)

            st.subheader("Progression: logistic baseline → tree ensemble (setup G columns)")
            yte = SS.df.loc[SS.test_idx, "Label_grouped"]
            with st.spinner("Fitting the logistic baseline (class_weight='balanced', saga) ..."):
                lr_res = nc.logreg_baseline(pr[nc.SETUP_G], yte)
            prog = {"Logistic regression (baseline)": lr_res, **{v: pr[nc.SETUP_G]["results"][v] for v in variants}}
            rows_p = []
            for name, r_ in prog.items():
                pc = r_["per_class"]
                rows_p.append({"model": name, "Macro-F1": r_["macro_f1"], "accuracy": r_["accuracy"],
                               **{f"F1 {c}": pc.loc[c, "f1"] for c in ("Web_Attack", "Bot") if c in pc.index}})
            ptab = pd.DataFrame(rows_p).set_index("model")
            ptab["Macro-F1 gain vs first row"] = ptab["Macro-F1"] - ptab["Macro-F1"].iloc[0]
            st.dataframe(ptab.round(4))
            fig, ax = plt.subplots(figsize=(7, 2.4))
            ax.barh(ptab.index[::-1], ptab["Macro-F1"].iloc[::-1])
            for i, v in enumerate(ptab["Macro-F1"].iloc[::-1]):
                ax.text(v + 0.01, i, f"{v:.3f}", va="center")
            ax.set_xlim(0, 1.1)
            ax.set_xlabel("Macro-F1 (test set)")
            fig.tight_layout()
            show_pyplot(fig)

# ---------------------------------- 6 Validation --------------------------------------
with t_val:
    st.header("6 · Validation added in revision 2")
    if need("fs" in SS, "Run feature selection in tab 3 first."):
        df, fs = SS.df, SS.fs
        st.subheader("Feature-leakage checks")
        if st.button("Run leakage checks", key="btn_leak"):
            with st.spinner("Checking ..."):
                num_c, cat_c = nc.setup_columns(nc.SETUP_G, fs["numeric"], fs["categorical"])
                cols = num_c + cat_c
                SS.val_leak = nc.leakage_checks(df.loc[SS.train_idx, cols], df.loc[SS.train_idx, "Label_grouped"],
                                                df.loc[SS.test_idx, cols], df.loc[SS.test_idx, "Label_grouped"], cols, num_c,
                                                df.loc[SS.train_idx, "session"], df.loc[SS.test_idx, "session"])
        if "val_leak" in SS:
            lk = SS.val_leak
            st.markdown(f"1. Identifier / label-derived columns among the inputs: **{lk['forbidden_columns_used'] or 'none'}**")
            st.markdown("2. Strongest single features, attack vs benign (AUC, direction ignored, training rows). "
                        f"Features above 0.99: **{lk['suspicious_auc_gt_099'] or 'none'}**")
            st.dataframe(lk["single_feature_auc"].head(8).round(4).to_frame("AUC"))
            st.markdown(f"3. Test rows whose exact feature vector also occurs in training: **{lk['overlap_share']:.2%}**. "
                        "Near-duplicate flows inflate random-split scores, which is why the session-held-out check below is stricter.")
            st.dataframe(lk["overlap_by_class"].to_frame("share of test rows also in train"))
            st.markdown(f"4. Largest class-share gap train vs test: **{lk['max_class_share_gap']:.5f}**  |  "
                        f"5. Sessions in train / test: **{lk['sessions_train']} / {lk['sessions_test']}** (a random split puts every session on both sides)")

        st.subheader("Anomaly detector for ultra-rare flows (Isolation Forest)")
        st.caption("Fitted on BENIGN pool rows only; the alert cut-off is set on held-out BENIGN rows so about the chosen share of BENIGN is flagged.")
        models = SS.get("models", {})
        if models:
            fpr = st.slider("BENIGN false-positive target", 0.01, 0.15, 0.05, step=0.01, key="iso_fpr")
            iso_setup = st.selectbox("Use preprocessed features of setup", list(models), format_func=lambda s: nc.SETUP_TEXT[s], key="iso_setup")
            if st.button("Fit Isolation Forest", key="btn_iso"):
                t = models[iso_setup]
                yte = SS.df.loc[SS.test_idx, "Label_grouped"].to_numpy()
                with st.spinner("Fitting ..."):
                    sc, cut = nc.run_isolation_forest(t["mat_pool"], t["mat_test"], t["y_pool_codes"], t["class_names"].index("BENIGN"), fpr_target=float(fpr))
                    SS.val_iso = {"res": nc.record_anomaly(sc, cut, yte), "yte": yte, "setup": iso_setup}
            if "val_iso" in SS:
                ir = SS.val_iso["res"]
                k = st.columns(4)
                k[0].metric("Attack recall", f"{ir['attack_recall']:.3f}")
                k[1].metric("Attack precision", f"{ir['attack_precision']:.3f}")
                k[2].metric("ROC AUC", f"{ir['roc_auc']:.3f}")
                k[3].metric("BENIGN flagged", f"{ir['benign_flagged']:,} / {ir['benign_total']:,}")
                share = pd.Series(ir["flagged"]).groupby(SS.val_iso["yte"]).mean().round(4)
                st.dataframe(share.to_frame("share flagged as anomalous"))
        else:
            st.info("Train a model in tab 4 first (the detector reuses its preprocessed matrices).")

        st.subheader("Session-held-out evaluation (leave-one-session-out)")
        st.markdown("Each capture session is held out in turn; the model trains on the other sessions and is scored on the held-out one. "
                    "For a class that exists in only one session, the score measures **detection of an unseen attack**, not classification.")
        if need_xgb():
            cap_s = st.number_input("Training flows per class in each fold", 500, 50_000, 3_000, step=500, key="loso_cap")
            if st.button("Run leave-one-session-out (setup B, XGBoost)", key="btn_loso"):
                num_b, cat_b = nc.setup_columns(nc.SETUP_B, fs["numeric"], fs["categorical"])
                xgb_cfg, _ = get_cfgs()
                n_rounds = SS.get("models", {}).get(nc.SETUP_B, {}).get("n_rounds", 100)
                status = st.status("Training 8 folds ...", expanded=True)
                with status:
                    dfm = df[["session", "Label_grouped"] + list(dict.fromkeys(num_b + cat_b))]
                    tab, pred = nc.leave_one_session_out(dfm, num_b, cat_b, xgb_cfg, int(n_rounds), int(cap_s), log=lambda m: status.write(m))
                    SS.val_loso = {"tab": tab, "pred": pred, "n_rounds": n_rounds}
                    status.update(label="Done", state="complete")
            if "val_loso" in SS:
                tab, pred = SS.val_loso["tab"], SS.val_loso["pred"]
                st.dataframe(tab.round(4))
                y_s = df["Label_grouped"].to_numpy()
                sess_tab = pd.crosstab(df["session"], df["Label_grouped"])
                spc = (sess_tab > 0).sum(axis=0)
                classes = sorted(np.unique(y_s))
                rs = SS.get("models", {}).get(nc.SETUP_B, {}).get("results", {}).get("XGBoost")
                rows_c = []
                for c in classes:
                    mc = y_s == c
                    rows_c.append({"class": c, "flows": int(mc.sum()), "sessions containing class": int(spc.get(c, 0)),
                                   "session-held-out: exact-class recall": float((pred[mc] == c).mean()),
                                   "session-held-out: flagged as attack": float((pred[mc] != "BENIGN").mean()) if c != "BENIGN" else np.nan,
                                   "random split: recall (XGBoost B)": float(rs["per_class"].loc[c, "recall"]) if rs is not None else np.nan})
                st.markdown("**Pooled over all held-out sessions, next to the random split**")
                st.dataframe(pd.DataFrame(rows_c).set_index("class").round(4))
                multi = [c for c in classes if spc.get(c, 0) >= 2]
                from sklearn.metrics import precision_recall_fscore_support as _prfs
                _, _, f_, _ = _prfs(y_s, pred.astype(str), labels=multi, zero_division=0)
                yb, pb = y_s != "BENIGN", pred != "BENIGN"
                k = st.columns(3)
                k[0].metric("Macro-F1 (classes in ≥2 sessions)", f"{np.mean(f_):.3f}")
                k[1].metric("Attack detection rate", f"{pb[yb].mean():.3f}")
                k[2].metric("BENIGN false-alarm rate", f"{pb[~yb].mean():.3f}")
                st.caption("Classes seen in only one session cannot be classified when that session is held out, so they are excluded from the Macro-F1 above. "
                           "Compare with the random-split scores in tab 4: the gap is the optimism of random splitting.")

# ---------------------------------- 7 Live --------------------------------------------
with t_live:
    st.header("7 · Live detection: verdict, family, confidence, top-3 features")
    if need_xgb():
        src = st.radio("Model", ["Model trained in this session", "Upload the notebook's ids_demo_bundle.joblib"], horizontal=True, key="live_src")
        bundle = None
        if src == "Model trained in this session":
            tm = SS.get("models", {})
            if need(bool(tm), "Train a model in tab 4 first."):
                live_setup = st.selectbox("Setup", list(tm), format_func=lambda s: nc.SETUP_TEXT[s], key="live_setup")
                bundle = nc.bundle_from_training(tm[live_setup])
                st.caption(f"Headline model: {bundle['headline']} | inputs: {len(bundle['input_columns'])} columns")
                if st.button("Prepare bundle for download", key="btn_bundle"):
                    buf = io.BytesIO()
                    joblib.dump(bundle, buf, compress=3)
                    SS.bundle_bytes = buf.getvalue()
                if "bundle_bytes" in SS:
                    st.download_button("Download ids_demo_bundle.joblib", SS.bundle_bytes, file_name="ids_demo_bundle.joblib")
        else:
            st.warning("Pickle files can run code when loaded. Only upload a bundle you exported yourself from the notebook.")
            up = st.file_uploader("ids_demo_bundle.joblib", type=["joblib", "pkl"], key="bundle_up")
            if up is not None:
                key = (up.name, up.size)
                if SS.get("uploaded_bundle_key") != key:
                    try:
                        SS.uploaded_bundle = nc.load_notebook_bundle(up)
                        SS.uploaded_bundle_key = key
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Could not load the bundle: {e}")
                bundle = SS.get("uploaded_bundle")

        if bundle is not None and need("df" in SS, "Load data from the sidebar to get flows to score."):
            df = SS.df
            miss = [c for c in bundle["input_columns"] if c not in df.columns]
            if miss:
                st.error(f"The loaded data lacks {len(miss)} columns this model needs (for example {miss[:4]}).")
            else:
                mode = st.radio("Flows to score", ["Random held-out test flows", "Edit one flow (what-if)", "Upload a flow CSV"], horizontal=True, key="live_mode")
                batch, truth = None, None
                classes_all = ["(any)"] + sorted(df["Label_grouped"].unique())
                if mode == "Random held-out test flows":
                    a, b, c = st.columns(3)
                    cls = a.selectbox("True class", classes_all, key="live_cls")
                    n = b.slider("How many flows", 1, 30, 8, key="live_n")
                    sd = c.number_input("Draw", 0, 1000, 0, key="live_seed")
                    pool = df.loc[SS.test_idx]
                    if cls != "(any)":
                        pool = pool[pool["Label_grouped"] == cls]
                    batch_ix = pool.sample(min(n, len(pool)), random_state=int(sd)).index
                    batch, truth = df.loc[batch_ix], df.loc[batch_ix, "Label_grouped"]
                elif mode == "Edit one flow (what-if)":
                    a, b = st.columns(2)
                    cls = a.selectbox("Start from a test flow of class", classes_all[1:], key="edit_cls")
                    sd = b.number_input("Which flow", 0, 1000, 0, key="edit_seed")
                    pool = df.loc[SS.test_idx]
                    pool = pool[pool["Label_grouped"] == cls]
                    ix = pool.sample(1, random_state=int(sd)).index[0]
                    row = df.loc[[ix]].copy()
                    numeric_inputs = [c for c in bundle["input_columns"] if c not in bundle.get("cat_columns", [])]
                    row[numeric_inputs] = row[numeric_inputs].astype("float64")      # flags are int8 in df; allow any edited value
                    default_feats = [c for c in ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s", "SYN Flag Count"] if c in numeric_inputs][:3]
                    chosen = st.multiselect("Features to change", numeric_inputs, default=default_feats, key="edit_feats")
                    cols = st.columns(3)
                    for i, f in enumerate(chosen):
                        cur = row.at[ix, f]
                        row.at[ix, f] = cols[i % 3].number_input(f, value=float(0 if pd.isna(cur) else cur), key=f"edit_{f}_{ix}", format="%.4f")
                    for cc in [c for c in bundle.get("cat_columns", []) if c in row.columns]:
                        opts = sorted(df[cc].dropna().astype(str).unique())
                        cur = str(row.at[ix, cc])
                        row[cc] = st.selectbox(cc, opts, index=opts.index(cur) if cur in opts else 0, key=f"edit_cat_{cc}_{ix}")
                    batch, truth = row, row["Label_grouped"]
                else:
                    up = st.file_uploader("CSV of raw flows (CIC-IDS2017 columns; Label optional)", type=["csv"], key="flow_up")
                    if up is not None:
                        raw_in = pd.read_csv(up, encoding="latin1", low_memory=False, nrows=5000)
                        if "Label" in raw_in.columns:
                            lab = raw_in["Label"].astype(str).str.replace(r"[^\x00-\x7F]+", "-", regex=True).str.strip().map(nc.ATTACK_MAPPING)
                        else:
                            lab = None
                        batch = nc.prepare_for_inference(raw_in.drop(columns=[c for c in ["Label"] if c in raw_in.columns]))
                        truth = lab.set_axis(batch.index) if lab is not None else None

                if batch is not None and len(batch):
                    try:
                        out = nc.predict_flows(batch, bundle)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Prediction failed: {e}")
                        out = None
                    if out is not None:
                        shown = out[[c for c in out.columns if not c.startswith("p(")]].copy()
                        if truth is not None:
                            shown.insert(0, "true_class", truth.values)
                            acc = (shown["true_class"] == shown["attack_family"]).mean()
                            st.caption(f"Exact-class agreement on this batch: {acc:.0%} ({len(shown)} flows)")
                        st.dataframe(shown)
                        pick = st.selectbox("Explain flow", list(range(len(out))), format_func=lambda i: f"#{i}  ->  {out['attack_family'].iloc[i]} ({out['confidence'].iloc[i]:.2f})", key="live_pick")
                        l, r = st.columns(2)
                        with l:
                            pcols = [c for c in out.columns if c.startswith("p(")]
                            st.markdown("**Class probabilities**")
                            st.bar_chart(pd.Series(out[pcols].iloc[pick].to_numpy(dtype=float), index=[c[2:-1] for c in pcols]))
                        with r:
                            shap_vals, names, pred_idx = out.attrs.get("shap", (None, None, None))
                            if shap_vals is not None:
                                sv = pd.Series(shap_vals[pick], index=names)
                                top = sv.reindex(sv.abs().sort_values(ascending=False).index[:10])[::-1]
                                fig, ax = plt.subplots(figsize=(5.5, 3.6))
                                ax.barh(top.index, top.values, color=["#c0392b" if v > 0 else "#2874a6" for v in top.values])
                                ax.set_title(f"TreeSHAP for '{out['attack_family'].iloc[pick]}' (XGBoost part)")
                                ax.axvline(0, color="grey", lw=0.8)
                                fig.tight_layout()
                                show_pyplot(fig)
                                st.caption("Red pushes towards the predicted class, blue away from it. If the ensemble weights the neural net, the explanation covers the XGBoost part only.")
