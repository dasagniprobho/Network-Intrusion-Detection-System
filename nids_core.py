"""
nids_core.py - the NIDS notebook's pipeline as importable functions (no Streamlit code in here).

Mirrors NIDS_model_final.ipynb (CIC-IDS2017, leakage-safe revision 2):
  clean_raw          -> cleaning cells (inf->NaN, constant cols, bad rows, duplicates)
  engineer           -> sentinel flags, Active/Idle flags, Destination_Port_Group, Label_grouped
  select_features    -> Spearman (0.95) then VIF (10), training rows only
  build_preproc      -> imputer -> signed log1p -> RobustScaler | imputer -> one-hot
  fit_xgb / fit_nn   -> class-weighted XGBoost and the regularised NumPy MLP
  cv_oof_probs       -> out-of-fold probabilities with fold-safe preprocessing
  predict_flows      -> verdict, family, confidence, top-3 TreeSHAP features
  make_synthetic     -> CIC-IDS2017-shaped synthetic data so the app runs without the dataset
"""
from __future__ import annotations

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

SEED = 42

# --------------------------------------------------------------------------------------
# Constants copied from the notebook
# --------------------------------------------------------------------------------------
ATTACK_MAPPING = {
    "BENIGN": "BENIGN",
    "DDoS": "DoS_DDoS", "DoS Hulk": "DoS_DDoS", "DoS GoldenEye": "DoS_DDoS",
    "DoS slowloris": "DoS_DDoS", "DoS Slowhttptest": "DoS_DDoS",
    "PortScan": "Reconnaissance",
    "FTP-Patator": "Brute_Force", "SSH-Patator": "Brute_Force", "Web Attack - Brute Force": "Brute_Force",
    "Web Attack - XSS": "Web_Attack", "Web Attack - Sql Injection": "Web_Attack",
    "Bot": "Bot",
    "Heartbleed": "Other_Rare", "Infiltration": "Other_Rare",
}
NEG_COLS = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Min", "Fwd Header Length", "Bwd Header Length", "min_seg_size_forward"]
TCP_COLS = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]
ACTIVE_IDLE_COLS = ["Active Mean", "Active Std", "Active Max", "Active Min",
                    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]
ENGINEERED_NUMERIC = ["forward_unknown", "backward_unknown",
                      "active_mean_present", "active_std_present", "active_max_present", "active_min_present",
                      "idle_mean_present", "idle_std_present", "idle_max_present", "idle_min_present"]
PORT_COL = "Destination Port"
SETUP_A, SETUP_B, SETUP_G = "A_with_port", "B_no_port", "G_port_group_only"
SETUP_TEXT = {
    SETUP_A: "A - raw Destination Port + port group",
    SETUP_B: "B - no port information at all (strictest)",
    SETUP_G: "G - port group only, no raw port",
}
XGB_BASE_PARAMS = dict(objective="multi:softprob", tree_method="hist", eval_metric="mlogloss",
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
NN_INPUT_CLIP = 10.0
THR_GRID = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
ENS_W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


# --------------------------------------------------------------------------------------
# Loading real CIC-IDS2017 CSVs
# --------------------------------------------------------------------------------------
def _session_name(name: str) -> str:
    base = os.path.basename(str(name))
    base = base.replace(".pcap_ISCX.csv", "")
    return re.sub(r"\.csv$", "", base, flags=re.I)


def cap_per_label(df: pd.DataFrame, cap: int | None, seed: int = SEED) -> pd.DataFrame:
    """Keep at most `cap` rows per original label (rare classes stay whole). Used only to bound memory."""
    if not cap:
        return df
    parts = [g.sample(min(len(g), cap), random_state=seed) if len(g) > cap else g for _, g in df.groupby("Label", sort=False)]
    return pd.concat(parts).sort_index()


def read_cic_csv(src, name: str, cap_per_label_rows: int | None = None) -> pd.DataFrame:
    """Read one original CIC-IDS2017 CSV the way the notebook does (latin1, stripped headers, session column)."""
    d = pd.read_csv(src, encoding="latin1", low_memory=False)
    d.columns = d.columns.str.strip()
    d["session"] = _session_name(name)
    if "Label" not in d.columns:
        raise ValueError(f"{name}: no 'Label' column - is this an original CIC-IDS2017 file?")
    d["Label"] = d["Label"].astype(str).str.replace(r"[^\x00-\x7F]+", "-", regex=True).str.strip()
    return cap_per_label(d, cap_per_label_rows)


def combine_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    df["Label"] = df["Label"].str.replace(r"[^\x00-\x7F]+", "-", regex=True).str.strip()
    num = df.select_dtypes("number").columns
    df[num] = df[num].astype("float32")
    return df


# --------------------------------------------------------------------------------------
# Cleaning (notebook: "Data Quality" + row-drop cell + integrity checks)
# --------------------------------------------------------------------------------------
def clean_raw(df: pd.DataFrame):
    """Returns (clean_df, report). Same order as the notebook: inf->NaN, drop constant cols and
    'Fwd Header Length.1', drop NaN/negative rows in the listed columns, drop exact duplicates."""
    rep = {}
    df = df.copy()
    df["Label"] = df["Label"].astype(str).str.replace(r"[^\x00-\x7F]+", "-", regex=True).str.strip()   # idempotent label normalisation
    n0 = len(df)
    num = df.select_dtypes("number").columns
    rep["rows_start"] = n0
    rep["inf_values"] = int(np.isinf(df[num].to_numpy(dtype="float64")).sum())
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    na = df[num].isna().sum()
    rep["nan_by_column"] = na[na > 0].to_dict()
    rep["duplicates_before"] = int(df.drop(columns="session").duplicated().sum())
    nu = df[num].nunique()
    const_cols = nu[nu <= 1].index.tolist()
    rep["constant_columns"] = const_cols
    neg = (df[num] < 0).sum()
    rep["negative_by_column"] = neg[neg > 0].to_dict()
    rep["dup_header_col"] = "Fwd Header Length.1" in df.columns
    rep["dup_header_identical"] = bool(rep["dup_header_col"] and df["Fwd Header Length"].equals(df["Fwd Header Length.1"]))
    df = df.drop(columns=const_cols + ["Fwd Header Length.1"], errors="ignore")

    neg_cols = [c for c in NEG_COLS if c in df.columns]
    nan_mask = df[[c for c in ["Flow Bytes/s", "Flow Packets/s"] if c in df.columns]].isna().any(axis=1)
    neg_mask = (df[neg_cols] < 0).any(axis=1)
    rep["nan_rows"], rep["negative_rows"], rep["bad_rows_union"] = int(nan_mask.sum()), int(neg_mask.sum()), int((nan_mask | neg_mask).sum())
    before_labels = df["Label"].value_counts()
    df = df[~(nan_mask | neg_mask)]
    rep["rows_after_bad_row_drop"] = len(df)
    feature_cols = [c for c in df.columns if c != "session"]
    df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    rep["rows_end"] = len(df)
    after = df["Label"].value_counts()
    loss = pd.DataFrame({"before_bad_rows": before_labels, "after_cleaning": after}).fillna(0).astype(int)
    loss["removed_in_bad_row+dup_steps"] = loss["before_bad_rows"] - loss["after_cleaning"]
    rep["label_table"] = loss.sort_values("after_cleaning", ascending=False)
    return df, rep


def integrity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Subset of the [REV2-ADDED] integrity checks that make sense for any input (real or synthetic)."""
    num = df.select_dtypes("number").columns
    neg_cols = [c for c in NEG_COLS if c in df.columns]
    lc = df["Label"].value_counts()
    checks = [
        ("no missing label", df["Label"].notna().all(), ""),
        ("no NaN in numeric columns", int(df[num].isna().sum().sum()) == 0, ""),
        ("no +/-inf in numeric columns", int(np.isinf(df[num].to_numpy(dtype="float64")).sum()) == 0, ""),
        ("no exact duplicate rows (session ignored)", int(df.drop(columns="session").duplicated().sum()) == 0, ""),
        ("no negatives in columns that cannot be negative", int((df[neg_cols] < 0).sum().sum()) == 0, ""),
        ("row index is 0..n-1", df.index.equals(pd.RangeIndex(len(df))), ""),
        ("every class has >= 2 rows (stratified split possible)", bool((lc >= 2).all()), f"smallest: {lc.idxmin()} = {int(lc.min())}"),
    ]
    return pd.DataFrame([{"check": n, "passed": bool(ok), "detail": d} for n, ok, d in checks])


# --------------------------------------------------------------------------------------
# Feature engineering (notebook: TCP indicators, Active/Idle flags, port groups, label grouping)
# --------------------------------------------------------------------------------------
def classify_port(port) -> str:
    if pd.isna(port):
        return "Unknown"
    port = int(port)
    common = {20: "FTP", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS", 67: "DHCP", 68: "DHCP",
              80: "HTTP", 110: "POP3", 123: "NTP", 143: "IMAP", 443: "HTTPS", 445: "SMB", 993: "IMAPS", 995: "POP3S"}
    if port in common:
        return common[port]
    if 0 <= port <= 1023:
        return "Other_Well_Known"
    if 1024 <= port <= 49151:
        return "Registered"
    if 49152 <= port <= 65535:
        return "Dynamic"
    return "Unknown"


def engineer(df: pd.DataFrame, with_labels: bool = True) -> pd.DataFrame:
    """Adds forward/backward_unknown (BEFORE -1 -> NaN), *_present flags, Destination_Port_Group, Label_grouped."""
    df = df.copy()
    for col in TCP_COLS:
        if col in df.columns:
            flag = col.replace("Init_Win_bytes_", "").lower() + "_unknown"
            if flag not in df.columns:
                df[flag] = (df[col] == -1).astype("int8")
            df[col] = df[col].replace(-1, np.nan)
    for col in ACTIVE_IDLE_COLS:
        if col in df.columns:
            df[col.lower().replace(" ", "_") + "_present"] = (df[col] > 0).astype("int8")
    if PORT_COL in df.columns:
        # vectorised equivalent of df[PORT_COL].apply(classify_port)
        df["Destination_Port_Group"] = df[PORT_COL].map(classify_port)
    if "Protocol" in df.columns:
        df["Protocol"] = df["Protocol"].astype("string")
    if with_labels and "Label" in df.columns:
        df["Label_grouped"] = df["Label"].map(ATTACK_MAPPING)
        if df["Label_grouped"].isna().any():
            unknown = df.loc[df["Label_grouped"].isna(), "Label"].unique().tolist()
            raise ValueError(f"Labels not in the notebook's attack taxonomy: {unknown}")
    return df


# --------------------------------------------------------------------------------------
# Split + feature selection (training rows only)
# --------------------------------------------------------------------------------------
def make_split(df: pd.DataFrame, test_size: float = 0.20, seed: int = SEED):
    """The one split, stratified on the original 15-class Label (grouped label if a class is too small)."""
    strat = df["Label"]
    if (strat.value_counts() < 2).any():
        strat = df["Label_grouped"]
    return train_test_split(df.index.to_numpy(), test_size=test_size, random_state=seed, stratify=strat)


def spearman_prune(df_train: pd.DataFrame, threshold=0.95, sample=100_000, seed=SEED):
    exclude = {PORT_COL, "is_attack", "Protocol"}
    feats = [c for c in df_train.select_dtypes("number").columns if c not in exclude and df_train[c].nunique(dropna=True) > 1]
    s = df_train[feats].sample(n=min(sample, len(df_train)), random_state=seed)
    s = s.fillna(s.median())
    corr = s.corr(method="spearman")
    absc = corr.abs()
    remaining, removed = list(absc.columns), []
    while True:
        cur = absc.loc[remaining, remaining]
        upper = cur.where(np.triu(np.ones(cur.shape), k=1).astype(bool))
        pairs = upper.stack()
        if pairs.empty or pairs.max() <= threshold:
            break
        f1, f2 = pairs.idxmax()
        mean_corr = cur.mean()
        drop = f1 if mean_corr[f1] >= mean_corr[f2] else f2
        remaining.remove(drop)
        removed.append((drop, float(pairs.max()), f2 if drop == f1 else f1))
    return {"corr_features": feats, "corr": corr, "remaining": remaining, "removed": removed}


def _vif_from_corr(X: np.ndarray) -> np.ndarray:
    """VIF_i = i-th diagonal entry of the inverse correlation matrix (identical to regressing each column on the
    others; used instead of statsmodels' per-column OLS loop because it is far faster for the app)."""
    C = np.corrcoef(X, rowvar=False)
    try:
        inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(C)
    v = np.diag(inv).copy()
    v[~np.isfinite(v) | (v < 0)] = np.inf
    return v


def vif_prune(df_train: pd.DataFrame, features: list[str], threshold=10.0, sample=20_000, seed=SEED):
    s = df_train[features].sample(n=min(sample, len(df_train)), random_state=seed)
    s = s.fillna(s.median())
    feats = [c for c in s.columns if s[c].nunique() > 1]
    s = s[feats]
    removed = []
    while len(feats) > 1:
        Z = (s[feats] - s[feats].mean()) / s[feats].std(ddof=0).replace(0, 1)
        vif = pd.Series(_vif_from_corr(Z.to_numpy()), index=feats)
        if vif.max() <= threshold:
            break
        worst = vif.idxmax()
        removed.append((worst, float(vif.max())))
        feats.remove(worst)
    Z = (s[feats] - s[feats].mean()) / s[feats].std(ddof=0).replace(0, 1)
    final = pd.DataFrame({"Feature": feats, "VIF": _vif_from_corr(Z.to_numpy())}).sort_values("VIF", ascending=False)
    return {"selected": feats, "removed": removed, "final": final}


def feature_lists(selected: list[str], columns) -> tuple[list[str], list[str]]:
    numeric = list(selected)
    for c in ENGINEERED_NUMERIC:
        if c in columns and c not in numeric:
            numeric.append(c)
    numeric = [c for c in numeric if c in columns and c != PORT_COL]
    categorical = ["Destination_Port_Group"]
    if "Protocol" in columns:
        categorical.append("Protocol")
    return numeric, categorical


def setup_columns(setup: str, numeric: list[str], categorical: list[str]):
    if setup == SETUP_A:
        return numeric + [PORT_COL], list(categorical)
    if setup == SETUP_B:
        return list(numeric), [c for c in categorical if c != "Destination_Port_Group"]
    return list(numeric), list(categorical)


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------
def signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


def make_numeric_pipeline():
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("log_transform", FunctionTransformer(signed_log1p, validate=False)),
                     ("scaler", RobustScaler())])


def make_categorical_pipeline():
    return Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("encoder", OneHotEncoder(handle_unknown="ignore"))])


def build_preproc(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([("numerical", make_numeric_pipeline(), num_cols),
                              ("categorical", make_categorical_pipeline(), cat_cols)],
                             remainder="drop", sparse_threshold=0.0)


def _prep_frame(frame: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for c in cat_cols:
        frame[c] = frame[c].astype(object)
    return frame


def transform_names(pre: ColumnTransformer, num_cols, cat_cols) -> list[str]:
    names = list(num_cols)
    if cat_cols:
        enc = pre.named_transformers_["categorical"].named_steps["encoder"]
        names += list(enc.get_feature_names_out(cat_cols))
    return names


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class MLPNumpyClassifier:
    """ReLU MLP: inverted dropout, L2 penalty, Adam, class-weighted cross-entropy, LR halving, early stopping."""

    def __init__(self, n_in, n_out, hidden=(256, 128), dropout=0.3, l2=1e-4, lr=1e-3,
                 batch_size=1024, max_epochs=40, patience=5, seed=0):
        self.dropout, self.l2, self.lr = dropout, l2, lr
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.rng = np.random.default_rng(seed)
        dims = [n_in, *hidden, n_out]
        self.W = [(self.rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32) for a, b in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in dims[1:]]
        self.history = []

    @staticmethod
    def _softmax(logits):
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _forward(self, X, train):
        acts, pre, masks = [X], [], []
        a = X
        for k in range(len(self.W) - 1):
            z = a @ self.W[k] + self.b[k]
            pre.append(z)
            a = np.maximum(z, 0)
            if train and self.dropout > 0:
                m = (self.rng.random(a.shape, dtype=np.float32) >= self.dropout).astype(np.float32) / (1.0 - self.dropout)
                a = a * m
                masks.append(m)
            acts.append(a)
        return a @ self.W[-1] + self.b[-1], acts, pre, masks

    def _weighted_loss(self, X, y, sw):
        total = 0.0
        for s in range(0, len(X), 50_000):
            p = self._softmax(self._forward(X[s:s + 50_000], False)[0])
            total += float((sw[s:s + 50_000] * -np.log(p[np.arange(len(p)), y[s:s + 50_000]] + 1e-12)).sum())
        return total / float(sw.sum())

    def fit(self, X, y, class_w, X_val, y_val, progress=None):
        sw_all, sw_val = class_w[y], class_w[y_val]
        params = self.W + self.b
        m1 = [np.zeros_like(p) for p in params]
        m2 = [np.zeros_like(p) for p in params]
        n, n_layers, step, lr_now = len(X), len(self.W), 0, self.lr
        best_loss, best_params, bad_epochs = np.inf, None, 0
        for epoch in range(1, self.max_epochs + 1):
            order = self.rng.permutation(n)
            for s in range(0, n, self.batch_size):
                ids = order[s:s + self.batch_size]
                xb, yb, wb = X[ids], y[ids], sw_all[ids]
                logits, acts, pre, masks = self._forward(xb, True)
                delta = self._softmax(logits)
                delta[np.arange(len(yb)), yb] -= 1.0
                delta *= (wb / wb.sum())[:, None]
                gW, gb = [None] * n_layers, [None] * n_layers
                for k in range(n_layers - 1, -1, -1):
                    gW[k] = acts[k].T @ delta + self.l2 * self.W[k]
                    gb[k] = delta.sum(axis=0)
                    if k > 0:
                        delta = delta @ self.W[k].T
                        if self.dropout > 0:
                            delta = delta * masks[k - 1]
                        delta = delta * (pre[k - 1] > 0)
                step += 1
                for j, (p, g) in enumerate(zip(params, gW + gb)):
                    m1[j] = 0.9 * m1[j] + 0.1 * g
                    m2[j] = 0.999 * m2[j] + 0.001 * g * g
                    p -= lr_now * (m1[j] / (1 - 0.9 ** step)) / (np.sqrt(m2[j] / (1 - 0.999 ** step)) + 1e-8)
            val_loss = self._weighted_loss(X_val, y_val, sw_val)
            self.history.append(val_loss)
            if progress:
                progress(epoch, val_loss)
            if val_loss < best_loss - 1e-4:
                best_loss, best_params, bad_epochs = val_loss, [p.copy() for p in params], 0
            else:
                bad_epochs += 1
                lr_now *= 0.5
                if bad_epochs >= self.patience:
                    break
        for p, bp in zip(params, best_params):
            p[...] = bp
        self.best_epoch = int(np.argmin(self.history)) + 1
        return self

    def predict_proba(self, X):
        return np.concatenate([self._softmax(self._forward(X[s:s + 50_000], False)[0]) for s in range(0, len(X), 50_000)])


def clip_nn_input(mat):
    return np.clip(mat, -NN_INPUT_CLIP, NN_INPUT_CLIP)


class ProbaWrap:
    def __init__(self, model, clip=False):
        self.model, self.clip = model, clip

    def proba(self, X_in):
        return self.model.predict_proba(clip_nn_input(X_in) if self.clip else X_in)


def class_weights_from_counts(y_codes, power, n_classes):
    counts = np.bincount(y_codes, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    return ((counts.max() / counts) ** power).astype(np.float32)


def fit_xgb(mat_tr, y_tr, cfg, seed, n_classes, n_estimators=None):
    """n_estimators=None -> early stopping on an inner 10% split; otherwise a fixed number of rounds."""
    import xgboost as xgb
    wvec = class_weights_from_counts(y_tr, cfg["wpow"], n_classes)
    params = dict(XGB_BASE_PARAMS, max_depth=cfg["max_depth"], min_child_weight=cfg["mcw"],
                  learning_rate=cfg["lr"], random_state=seed)
    if n_estimators is None:
        try:
            ix_in, ix_es = train_test_split(np.arange(len(y_tr)), test_size=0.1, stratify=y_tr, random_state=seed)
        except ValueError:  # a class with a single row cannot be stratified
            ix_in, ix_es = train_test_split(np.arange(len(y_tr)), test_size=0.1, random_state=seed)
        model = xgb.XGBClassifier(n_estimators=cfg.get("max_rounds", 400), early_stopping_rounds=20, **params)
        model.fit(mat_tr[ix_in], y_tr[ix_in], sample_weight=wvec[y_tr[ix_in]],
                  eval_set=[(mat_tr[ix_es], y_tr[ix_es])], sample_weight_eval_set=[wvec[y_tr[ix_es]]], verbose=False)
        used = int(model.best_iteration) + 1
    else:
        model = xgb.XGBClassifier(n_estimators=n_estimators, **params)
        model.fit(mat_tr, y_tr, sample_weight=wvec[y_tr], verbose=False)
        used = n_estimators
    return ProbaWrap(model), used


def fit_nn(mat_tr, y_tr, cfg, seed, n_classes, progress=None):
    wvec = class_weights_from_counts(y_tr, cfg["wpow"], n_classes)
    try:
        ix_in, ix_es = train_test_split(np.arange(len(y_tr)), test_size=0.1, stratify=y_tr, random_state=seed)
    except ValueError:
        ix_in, ix_es = train_test_split(np.arange(len(y_tr)), test_size=0.1, random_state=seed)
    model = MLPNumpyClassifier(n_in=mat_tr.shape[1], n_out=n_classes, hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"],
                               max_epochs=cfg.get("epochs", 40), seed=seed)
    model.fit(clip_nn_input(mat_tr[ix_in]), y_tr[ix_in], wvec, clip_nn_input(mat_tr[ix_es]), y_tr[ix_es], progress=progress)
    return ProbaWrap(model, clip=True), model.best_epoch


def cv_oof_probs(fit_fn, raw_frame, preproc_template, y_codes, n_splits, n_classes, cat_cols, seed=SEED):
    """Out-of-fold probabilities with FOLD-SAFE preprocessing (cloned preprocessing fitted on each training fold only)."""
    oof = np.zeros((len(y_codes), n_classes), dtype=np.float32)
    infos = []
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for ix_tr, ix_te in skf.split(np.zeros(len(y_codes)), y_codes):
        pre = clone(preproc_template)
        mat_tr = np.asarray(pre.fit_transform(_prep_frame(raw_frame.iloc[ix_tr], cat_cols)), dtype=np.float32)
        mat_te = np.asarray(pre.transform(_prep_frame(raw_frame.iloc[ix_te], cat_cols)), dtype=np.float32)
        wrap, info = fit_fn(mat_tr, y_codes[ix_tr], seed)
        oof[ix_te] = wrap.proba(mat_te)
        infos.append(info)
    return oof, infos


def weighted_macro_f1(y_true, y_pred, w, n_classes):
    cm = np.bincount(y_true * n_classes + y_pred, weights=w, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    tp = np.diag(cm)
    fp, fn = cm.sum(axis=0) - tp, cm.sum(axis=1) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    return float(f1.mean()), f1


def natural_mix_weights(y_codes, natural_counts):
    return (natural_counts / np.bincount(y_codes, minlength=len(natural_counts)))[y_codes]


def tune_class_scales(prob, y_codes, w, n_classes):
    scales = np.ones(n_classes, dtype=np.float32)
    best = weighted_macro_f1(y_codes, (prob * scales).argmax(axis=1), w, n_classes)[0]
    for _ in range(3):
        for k in range(n_classes):
            for cand in THR_GRID:
                trial = scales.copy()
                trial[k] = cand
                f = weighted_macro_f1(y_codes, (prob * trial).argmax(axis=1), w, n_classes)[0]
                if f > best + 1e-4:
                    best, scales = f, trial
    return scales, best


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def evaluate(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {"macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "per_class": pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=labels),
            "confusion": confusion_matrix(y_true, y_pred, labels=labels), "y_pred": np.asarray(y_pred)}


def binary_view(y_true, y_pred):
    """BENIGN vs attack view of multi-class predictions."""
    t, p = np.asarray(y_true) != "BENIGN", np.asarray(y_pred) != "BENIGN"
    return {"binary_macro_f1": float(f1_score(t, p, average="macro")),
            "attack_detection_rate": float(p[t].mean()) if t.any() else float("nan"),
            "benign_false_alarm_rate": float(p[~t].mean()) if (~t).any() else float("nan")}


def run_isolation_forest(mat_pool, mat_test, y_pool_codes, benign_code, fpr_target=0.05, fit_rows=100_000, seed=SEED):
    from sklearn.ensemble import IsolationForest
    ix = np.random.default_rng(seed).permutation(np.where(y_pool_codes == benign_code)[0])
    n_fit = int(0.7 * len(ix))
    fit_ix, cal_ix = ix[:n_fit][:fit_rows], ix[n_fit:]
    iso = IsolationForest(n_estimators=300, max_samples=min(2048, len(fit_ix)), contamination="auto", n_jobs=-1,
                          random_state=seed).fit(mat_pool[fit_ix])
    cutoff = float(np.quantile(iso.score_samples(mat_pool[cal_ix]), fpr_target))
    return iso.score_samples(mat_test), cutoff


def record_anomaly(score_test, cutoff, y_true):
    is_att = np.asarray(y_true) != "BENIGN"
    flagged = score_test < cutoff
    tp, fp, fn = int((flagged & is_att).sum()), int((flagged & ~is_att).sum()), int((~flagged & is_att).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"attack_precision": prec, "attack_recall": rec, "attack_f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "binary_macro_f1": float(f1_score(is_att, flagged, average="macro")),
            "roc_auc": float(roc_auc_score(is_att, -score_test)), "pr_auc": float(average_precision_score(is_att, -score_test)),
            "benign_flagged": fp, "benign_total": int((~is_att).sum()), "flagged": flagged}


# --------------------------------------------------------------------------------------
# Inference (the notebook's predict_flows) and the exported joblib bundle
# --------------------------------------------------------------------------------------
def prepare_for_inference(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw CIC-style rows -> engineered feature frame (no rows are dropped, inf -> NaN, labels optional)."""
    raw = raw.copy()
    raw.columns = raw.columns.astype(str).str.strip()
    raw = raw.drop(columns=["Fwd Header Length.1"], errors="ignore")
    num = raw.select_dtypes("number").columns
    raw[num] = raw[num].replace([np.inf, -np.inf], np.nan).astype("float32")
    return engineer(raw, with_labels=False)


def predict_flows(batch_df: pd.DataFrame, bundle: dict, top_k: int = 3, explain: bool = True) -> pd.DataFrame:
    """Verdict, attack family, confidence, attack probability and top-k TreeSHAP features per flow."""
    import xgboost as xgb
    cols = bundle["input_columns"]
    missing = [c for c in cols if c not in batch_df.columns]
    if missing:
        raise KeyError(f"{len(missing)} model input columns missing from the data, e.g. {missing[:5]}")
    cat_cols = bundle.get("cat_columns", [])
    mat = np.asarray(bundle["preprocessor"].transform(_prep_frame(batch_df[cols], cat_cols)), dtype=np.float32)

    prob_x = np.mean([m.predict_proba(mat) for m in bundle["xgb_models"]], axis=0) if bundle["xgb_models"] else None
    prob = prob_x
    if bundle["xgb_weight"] < 1.0 and bundle["nn_models"]:
        prob_nn = np.mean([m.predict_proba(clip_nn_input(mat)) for m in bundle["nn_models"]], axis=0)
        prob = prob_nn if prob_x is None else bundle["xgb_weight"] * prob_x + (1.0 - bundle["xgb_weight"]) * prob_nn
    prob = prob * np.asarray(bundle["class_scales"], dtype=np.float32)
    prob = prob / prob.sum(axis=1, keepdims=True)

    classes = np.array(bundle["class_names"])
    pred_idx = prob.argmax(axis=1)
    family = classes[pred_idx]
    out = pd.DataFrame({
        "verdict": np.where(family == "BENIGN", "BENIGN", "ALERT"),
        "attack_family": family,
        "confidence": prob[np.arange(len(mat)), pred_idx].round(4),
        "attack_probability": (1.0 - prob[:, list(classes).index("BENIGN")]).round(4),
    }, index=batch_df.index)
    for k, c in enumerate(classes):
        out[f"p({c})"] = prob[:, k].round(4)

    if explain and bundle["xgb_models"]:
        contrib = np.mean([m.get_booster().predict(xgb.DMatrix(mat), pred_contribs=True) for m in bundle["xgb_models"]], axis=0)
        cp = contrib[np.arange(len(mat)), pred_idx, :-1]
        names = np.array(bundle["feature_names"])
        top = np.argsort(-cp, axis=1)[:, :top_k]
        out["top_3_features (SHAP)"] = [", ".join(f"{names[j]} ({cp[r, j]:+.2f})" for j in top[r]) for r in range(len(mat))]
        out.attrs["shap"] = (cp, names, pred_idx)
    return out


def load_notebook_bundle(path_or_file):
    """Load the notebook's ids_demo_bundle.joblib. The pickle refers to functions/classes defined in the notebook's
    __main__, so the same names are registered on __main__ before loading."""
    import joblib
    main = sys.modules["__main__"]
    for name, obj in {"signed_log1p": signed_log1p, "MLPNumpyClassifier": MLPNumpyClassifier,
                      "ProbaWrap": ProbaWrap, "clip_nn_input": clip_nn_input}.items():
        setattr(main, name, obj)
    bundle = joblib.load(path_or_file)
    # the notebook's bundle stores the raw column names; mark which of them are categorical
    bundle.setdefault("cat_columns", [c for c in bundle["input_columns"] if c in ("Destination_Port_Group", "Protocol")])
    return bundle


# --------------------------------------------------------------------------------------
# Synthetic CIC-IDS2017-shaped data (demo mode, no download needed)
# --------------------------------------------------------------------------------------
CIC_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
    "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std",
    "Idle Max", "Idle Min",
]

REAL_COUNTS = {"BENIGN": 2_100_000, "DoS Hulk": 230_000, "PortScan": 158_000, "DDoS": 128_000, "DoS GoldenEye": 10_000,
               "FTP-Patator": 7_900, "SSH-Patator": 5_900, "DoS slowloris": 5_800, "DoS Slowhttptest": 5_500,
               "Bot": 1_970, "Web Attack - Brute Force": 1_500, "Web Attack - XSS": 650, "Infiltration": 36,
               "Web Attack - Sql Injection": 21, "Heartbleed": 11}

SESSION_CLASSES = {
    "Monday-WorkingHours": ["BENIGN"],
    "Tuesday-WorkingHours": ["BENIGN", "FTP-Patator", "SSH-Patator"],
    "Wednesday-workingHours": ["BENIGN", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "Heartbleed"],
    "Thursday-WorkingHours-Morning-WebAttacks": ["BENIGN", "Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection"],
    "Thursday-WorkingHours-Afternoon-Infilteration": ["BENIGN", "Infiltration"],
    "Friday-WorkingHours-Morning": ["BENIGN", "Bot"],
    "Friday-WorkingHours-Afternoon-PortScan": ["BENIGN", "PortScan"],
    "Friday-WorkingHours-Afternoon-DDos": ["BENIGN", "DDoS"],
}
BENIGN_SESSION_WEIGHTS = [0.25, 0.15, 0.17, 0.08, 0.08, 0.09, 0.09, 0.09]

# port mix, log-duration (mu, sd), log fwd pkts, log bwd pkts, log fwd len, log bwd len, init-win fwd choices,
# init-win bwd choices, P(syn), P(rst), P(psh), P(idle)
_PROFILES = {
    "BENIGN": ({80: .25, 443: .35, 53: .2, 123: .03, 445: .02, 22: .02, 21: .01, 8080: .02, -1: .10}, (11.0, 3.3), (1.0, .9), (.8, 1.0), (4.0, 1.3), (4.8, 1.5), [8192, 29200, 65535, 256, 229, -1], [-1, 0, 235, 28960, 65535], .05, .02, .45, .08),
    "DoS Hulk": ({80: 1.0}, (13.5, 1.5), (1.8, .5), (1.5, .7), (5.0, .8), (5.5, 1.0), [29200], [235], .05, .02, .7, .2),
    "DDoS": ({80: .98, -1: .02}, (11.8, 1.6), (1.4, .4), (1.3, .4), (3.3, .8), (6.0, .6), [256], [0], .05, .01, .8, .05),
    "DoS GoldenEye": ({80: 1.0}, (15.5, 1.0), (2.2, .6), (2.0, .6), (4.5, .7), (6.2, .6), [29200], [235], .05, .02, .75, .45),
    "DoS slowloris": ({80: 1.0}, (17.5, .8), (1.6, .4), (1.3, .4), (4.8, .5), (2.5, .8), [29200], [235], .05, .02, .6, .7),
    "DoS Slowhttptest": ({80: 1.0}, (16.8, .9), (1.7, .4), (1.4, .4), (5.0, .6), (4.5, .7), [29200], [235], .05, .02, .6, .6),
    "PortScan": ({-1: 1.0}, (4.0, 1.0), (.02, .1), (.02, .1), (1.0, .5), (1.0, .6), [1024, 256, 512, 29200], [0, -1], .9, .8, .02, 0.0),
    "FTP-Patator": ({21: 1.0}, (12.5, 1.0), (1.9, .3), (1.8, .3), (1.5, .4), (2.5, .4), [29200], [227], .05, .02, .9, 0.0),
    "SSH-Patator": ({22: 1.0}, (15.0, .7), (2.4, .3), (2.5, .3), (3.2, .4), (3.8, .4), [29200], [247], .05, .02, .9, 0.05),
    "Web Attack - Brute Force": ({80: 1.0}, (12.3, 2.3), (1.5, .5), (1.5, .6), (4.3, .6), (6.5, .9), [29200, 8192], [235, 28960], .05, .02, .6, .1),
    "Web Attack - XSS": ({80: 1.0}, (12.4, 2.3), (1.5, .5), (1.5, .6), (4.4, .6), (6.4, .9), [29200, 8192], [235, 28960], .05, .02, .6, .1),
    "Web Attack - Sql Injection": ({80: 1.0}, (12.3, 2.4), (1.6, .5), (1.5, .6), (4.4, .6), (6.3, .9), [29200, 8192], [235, 28960], .05, .02, .6, .1),
    "Bot": ({8080: .3, 443: .2, 80: .2, -1: .3}, (12.0, 2.5), (1.1, .6), (1.0, .7), (3.5, 1.0), (4.5, 1.2), [8192, 29200, 256], [-1, 235, 0], .05, .03, .5, .1),
    "Heartbleed": ({444: 1.0}, (18.2, .2), (3.0, .2), (6.0, .2), (4.0, .3), (6.5, .3), [246], [0], .02, .01, .9, .8),
    "Infiltration": ({-1: .4, 443: .3, 8080: .3}, (14.0, 2.0), (1.8, .8), (1.6, .8), (4.0, 1.1), (5.5, 1.3), [8192, 29200, 65535], [-1, 235, 0], .05, .03, .5, .3),
}


def synthetic_counts(total_rows: int) -> dict:
    scale = total_rows / sum(REAL_COUNTS.values())
    return {k: int(max(round(v * scale), min(v, 60))) for k, v in REAL_COUNTS.items()}


def _gen_class(label: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    ports, dur_p, fp_p, bp_p, fl_p, bl_p, win_f, win_b, p_syn, p_rst, p_psh, p_idle = _PROFILES[label]
    keys, probs = list(ports.keys()), np.array(list(ports.values()), dtype=float)
    pick = rng.choice(len(keys), size=n, p=probs / probs.sum())
    port = np.array(keys)[pick].astype(float)
    port = np.where(port == -1, rng.integers(1, 65536, n), port)

    dur = np.round(np.exp(rng.normal(*dur_p, n)))
    dur = np.where(rng.random(n) < 0.003, 0.0, np.maximum(dur, 1))          # a few zero durations -> inf / NaN rates
    fwd = np.maximum(1, np.round(np.exp(rng.normal(*fp_p, n)))).astype(float)
    bwd = np.maximum(0, np.round(np.exp(rng.normal(*bp_p, n)) - 0.6)).astype(float)
    fmean = np.clip(np.exp(rng.normal(*fl_p, n)), 0, 1460)
    bmean = np.where(bwd > 0, np.clip(np.exp(rng.normal(*bl_p, n)), 0, 1460), 0.0)
    tot_f, tot_b = np.round(fwd * fmean), np.round(bwd * bmean)
    npk = fwd + bwd

    def spread(mean, cnt, hi):
        mx = np.clip(mean * (1 + np.abs(rng.normal(0, .6, n))), 0, 1460) * (cnt > 0)
        mn = mean * rng.random(n) * (cnt > 0)
        sd = mean * rng.random(n) * .6 * (cnt > 1)
        return mx, mn, sd
    fmx, fmn, fsd = spread(fmean, fwd, 1460)
    bmx, bmn, bsd = spread(bmean, bwd, 1460)

    secs = dur / 1e6
    with np.errstate(divide="ignore", invalid="ignore"):
        bytes_s, pkts_s = (tot_f + tot_b) / secs, npk / secs
        fpk_s, bpk_s = fwd / secs, bwd / secs
    iat_mean = dur / np.maximum(npk - 1, 1)
    iat_max = np.minimum(iat_mean * (1 + rng.random(n) * 4), dur)
    iat_min = iat_mean * rng.random(n) * .5
    iat_std = iat_mean * rng.random(n) * 2

    def dir_iat(cnt):
        tot = dur * (.6 + .4 * rng.random(n)) * (cnt > 1)
        m = tot / np.maximum(cnt - 1, 1)
        return tot, m, m * rng.random(n) * 2, np.minimum(m * (1 + rng.random(n) * 4), tot), m * rng.random(n) * .5
    f_tot, f_m, f_s, f_mx, f_mn = dir_iat(fwd)
    b_tot, b_m, b_s, b_mx, b_mn = dir_iat(bwd)

    pl_mean = (tot_f + tot_b) / npk
    pl_std = np.sqrt(np.maximum((fsd ** 2 * fwd + bsd ** 2 * bwd) / npk, 0)) + np.abs(fmean - bmean) * .3 * (bwd > 0)
    fhdr = 20 * fwd + 12 * (rng.random(n) < .3)
    idle_on = (dur > 5e6) & (rng.random(n) < p_idle)
    idle_m = np.where(idle_on, dur * (.3 + .6 * rng.random(n)), 0.0)
    act_m = np.where(idle_on, dur * rng.random(n) * .1, 0.0)

    win_fwd = rng.choice(win_f, n).astype(float)
    win_bwd = np.where(bwd > 0, rng.choice(win_b, n).astype(float), -1.0)
    syn = (rng.random(n) < p_syn).astype(float)
    rst = (rng.random(n) < p_rst).astype(float)
    psh = (rng.random(n) < p_psh).astype(float)

    d = {
        "Destination Port": port, "Flow Duration": dur, "Total Fwd Packets": fwd, "Total Backward Packets": bwd,
        "Total Length of Fwd Packets": tot_f, "Total Length of Bwd Packets": tot_b,
        "Fwd Packet Length Max": fmx, "Fwd Packet Length Min": fmn, "Fwd Packet Length Mean": fmean, "Fwd Packet Length Std": fsd,
        "Bwd Packet Length Max": bmx, "Bwd Packet Length Min": bmn, "Bwd Packet Length Mean": bmean, "Bwd Packet Length Std": bsd,
        "Flow Bytes/s": bytes_s, "Flow Packets/s": pkts_s, "Flow IAT Mean": iat_mean, "Flow IAT Std": iat_std,
        "Flow IAT Max": iat_max, "Flow IAT Min": iat_min,
        "Fwd IAT Total": f_tot, "Fwd IAT Mean": f_m, "Fwd IAT Std": f_s, "Fwd IAT Max": f_mx, "Fwd IAT Min": f_mn,
        "Bwd IAT Total": b_tot, "Bwd IAT Mean": b_m, "Bwd IAT Std": b_s, "Bwd IAT Max": b_mx, "Bwd IAT Min": b_mn,
        "Fwd PSH Flags": psh * (rng.random(n) < .5), "Bwd PSH Flags": np.zeros(n), "Fwd URG Flags": np.zeros(n), "Bwd URG Flags": np.zeros(n),
        "Fwd Header Length": fhdr, "Bwd Header Length": 20 * bwd,
        "Fwd Packets/s": fpk_s, "Bwd Packets/s": bpk_s,
        "Min Packet Length": np.minimum(fmn, np.where(bwd > 0, bmn, fmn)), "Max Packet Length": np.maximum(fmx, bmx),
        "Packet Length Mean": pl_mean, "Packet Length Std": pl_std, "Packet Length Variance": pl_std ** 2,
        "FIN Flag Count": (rng.random(n) < .25).astype(float), "SYN Flag Count": syn, "RST Flag Count": rst, "PSH Flag Count": psh,
        "ACK Flag Count": (rng.random(n) < .6).astype(float), "URG Flag Count": (rng.random(n) < .05).astype(float),
        "CWE Flag Count": np.zeros(n), "ECE Flag Count": rst, "Down/Up Ratio": np.floor(bwd / np.maximum(fwd, 1)),
        "Average Packet Size": pl_mean * (1 + .02 * rng.random(n)), "Avg Fwd Segment Size": fmean, "Avg Bwd Segment Size": bmean,
        "Fwd Header Length.1": fhdr,
        "Fwd Avg Bytes/Bulk": np.zeros(n), "Fwd Avg Packets/Bulk": np.zeros(n), "Fwd Avg Bulk Rate": np.zeros(n),
        "Bwd Avg Bytes/Bulk": np.zeros(n), "Bwd Avg Packets/Bulk": np.zeros(n), "Bwd Avg Bulk Rate": np.zeros(n),
        "Subflow Fwd Packets": fwd, "Subflow Fwd Bytes": tot_f, "Subflow Bwd Packets": bwd, "Subflow Bwd Bytes": tot_b,
        "Init_Win_bytes_forward": win_fwd, "Init_Win_bytes_backward": win_bwd,
        "act_data_pkt_fwd": np.floor(np.maximum(fwd - 1, 0) * rng.random(n)),
        "min_seg_size_forward": rng.choice([20, 32, 40], n, p=[.6, .3, .1]).astype(float),
        "Active Mean": act_m, "Active Std": act_m * rng.random(n) * .3, "Active Max": act_m * (1 + rng.random(n) * .5), "Active Min": act_m * rng.random(n),
        "Idle Mean": idle_m, "Idle Std": idle_m * rng.random(n) * .3, "Idle Max": idle_m * (1 + rng.random(n) * .5), "Idle Min": idle_m * rng.random(n),
    }
    out = pd.DataFrame({c: d[c] for c in CIC_FEATURES})
    out["Label"] = label
    return out


def make_synthetic(total_rows: int = 80_000, seed: int = SEED, dirty: bool = True) -> pd.DataFrame:
    """CIC-IDS2017-shaped raw data: same 78 column names, 15 labels, 8 sessions with the real class-to-session layout,
    plus inf / NaN / negative / duplicate rows so the cleaning step has something to do. NOT real traffic."""
    rng = np.random.default_rng(seed)
    counts = synthetic_counts(total_rows)
    parts = []
    sessions = list(SESSION_CLASSES)
    for label, n in counts.items():
        frame = _gen_class(label, n, rng)
        if label == "BENIGN":
            sess = rng.choice(len(sessions), size=n, p=np.array(BENIGN_SESSION_WEIGHTS) / sum(BENIGN_SESSION_WEIGHTS))
        else:
            holders = [i for i, s in enumerate(sessions) if label in SESSION_CLASSES[s]]
            sess = rng.choice(holders, size=n)
        frame["session"] = np.array(sessions)[sess]
        parts.append(frame)
    df = pd.concat(parts, ignore_index=True)
    df["Label"] = df["Label"].str.replace("Web Attack - ", "Web Attack \ufffd ", regex=False)   # mimics the raw file's odd dash
    if dirty:
        n = len(df)
        bad = rng.choice(n, size=max(3, int(0.0015 * n)), replace=False)
        df.loc[bad, "Flow IAT Min"] = -1.0
        bad2 = rng.choice(n, size=max(3, int(0.0005 * n)), replace=False)
        df.loc[bad2, "Flow Duration"] = -1.0
        dup = df.sample(frac=0.02, random_state=seed)
        df = pd.concat([df, dup], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    num = df.select_dtypes("number").columns
    df[num] = df[num].astype("float32")
    return df


# --------------------------------------------------------------------------------------
# One-call training used by the app
# --------------------------------------------------------------------------------------
def sample_pool(y_train: pd.Series, cap: int, seed=SEED) -> np.ndarray:
    """Per-class capped sample of the training split (rare classes stay whole). Returns index labels."""
    rng = np.random.default_rng(seed)
    parts = []
    for cls in sorted(y_train.unique()):
        idx = y_train.index[(y_train == cls).to_numpy()].to_numpy()
        parts.append(rng.choice(idx, size=min(len(idx), cap), replace=False))
    return rng.permutation(np.concatenate(parts))


def train_setup(X_train, y_train, X_test, y_test, num_cols, cat_cols, class_names, *, pool_cap, xgb_cfg, nn_cfg,
                use_nn=True, n_seeds=1, ens_w=0.5, tune=False, cv_folds=3, tune_cap=8_000, log=print):
    """Fit preprocessing on the training pool, train XGBoost (+NN), optional fold-safe OOF tuning of the ensemble
    weight and class multipliers, then score the untouched test set once."""
    n_cls = len(class_names)
    code = {c: i for i, c in enumerate(class_names)}
    cls_arr = np.array(class_names)
    idx_pool = sample_pool(y_train, pool_cap)
    pool_raw = X_train.loc[idx_pool, num_cols + cat_cols]
    y_pool = y_train.loc[idx_pool].map(code).to_numpy(dtype=np.int64)
    nat_counts = y_train.value_counts().reindex(class_names).to_numpy(dtype=np.float64)

    pre = build_preproc(num_cols, cat_cols)
    mat_pool = np.asarray(pre.fit_transform(_prep_frame(pool_raw, cat_cols)), dtype=np.float32)
    mat_test = np.asarray(pre.transform(_prep_frame(X_test[num_cols + cat_cols], cat_cols)), dtype=np.float32)
    names = transform_names(pre, num_cols, cat_cols)
    log(f"Preprocessed pool {mat_pool.shape}, test {mat_test.shape}")

    scales, w_use, tune_info = np.ones(n_cls, dtype=np.float32), (ens_w if use_nn else 1.0), None
    if tune:
        rng = np.random.default_rng(SEED)
        pos = []
        for k in range(n_cls):
            p = np.where(y_pool == k)[0]
            pos.append(rng.choice(p, size=min(len(p), tune_cap), replace=False))
        pos = rng.permutation(np.concatenate(pos))
        y_t, frame_t = y_pool[pos], pool_raw.iloc[pos]
        w_eval = natural_mix_weights(y_t, nat_counts)
        template = build_preproc(num_cols, cat_cols)
        log(f"Fold-safe OOF ({cv_folds} folds) for XGBoost ...")
        oof_x, infos = cv_oof_probs(lambda m, y, s: fit_xgb(m, y, xgb_cfg, s, n_cls), frame_t, template, y_t, cv_folds, n_cls, cat_cols)
        f_x = weighted_macro_f1(y_t, oof_x.argmax(axis=1), w_eval, n_cls)[0]
        tune_info = {"oof_macro_f1_xgb": f_x, "cv_rounds": infos}
        oof_ens = oof_x
        if use_nn:
            log("Fold-safe OOF for the neural net ...")
            oof_n, _ = cv_oof_probs(lambda m, y, s: fit_nn(m, y, nn_cfg, s, n_cls), frame_t, template, y_t, cv_folds, n_cls, cat_cols)
            tune_info["oof_macro_f1_nn"] = weighted_macro_f1(y_t, oof_n.argmax(axis=1), w_eval, n_cls)[0]
            ens_scores = {w: weighted_macro_f1(y_t, (w * oof_x + (1 - w) * oof_n).argmax(axis=1), w_eval, n_cls)[0] for w in ENS_W_GRID}
            w_use = max(ens_scores, key=ens_scores.get)
            tune_info["ens_scores"] = ens_scores
            oof_ens = w_use * oof_x + (1 - w_use) * oof_n
        scales, f_thr = tune_class_scales(oof_ens, y_t, w_eval, n_cls)
        tune_info.update({"ens_w": w_use, "oof_macro_f1_thresholds": f_thr, "class_multipliers": dict(zip(class_names, scales.round(2)))})
        n_rounds = int(round(1.1 * np.mean(infos)))
    else:
        log("Finding the number of boosting rounds (early stopping on an inner 10% split) ...")
        _, n_used = fit_xgb(mat_pool, y_pool, xgb_cfg, SEED, n_cls)
        n_rounds = max(int(round(1.1 * n_used)), 10)

    log(f"Final refits on the whole pool: {n_seeds} seed(s), {n_rounds} XGBoost rounds")
    xgb_w, nn_w = [], []
    for s in range(n_seeds):
        xgb_w.append(fit_xgb(mat_pool, y_pool, xgb_cfg, SEED + s, n_cls, n_estimators=n_rounds)[0])
        if use_nn:
            nn_w.append(fit_nn(mat_pool, y_pool, nn_cfg, SEED + s, n_cls)[0])
    prob_x = np.mean([w.proba(mat_test) for w in xgb_w], axis=0)
    prob_n = np.mean([w.proba(mat_test) for w in nn_w], axis=0) if use_nn else None
    prob_ens = w_use * prob_x + (1 - w_use) * prob_n if use_nn else prob_x

    y_true = y_test.to_numpy()
    variants = {"XGBoost": prob_x}
    if use_nn:
        variants["Neural net"] = prob_n
        variants["Ensemble"] = prob_ens
    if tune:
        variants["Ensemble + class thresholds" if use_nn else "XGBoost + class thresholds"] = prob_ens * scales
    results = {k: evaluate(y_true, cls_arr[p.argmax(axis=1)], class_names) for k, p in variants.items()}
    for k, p in variants.items():
        results[k]["proba"] = p

    return {"pre": pre, "num_cols": num_cols, "cat_cols": cat_cols, "feature_names": names, "class_names": class_names,
            "xgb_models": [w.model for w in xgb_w], "nn_models": [w.model for w in nn_w], "xgb_weight": float(w_use),
            "class_scales": scales, "n_rounds": n_rounds, "results": results, "tune_info": tune_info,
            "mat_pool": mat_pool, "mat_test": mat_test, "y_pool_codes": y_pool, "pool_size": len(idx_pool)}


def bundle_from_training(t: dict) -> dict:
    """Same dictionary layout as the notebook's ids_demo_bundle.joblib."""
    return {"setup": t.get("setup"), "headline": t.get("headline"), "class_names": list(t["class_names"]),
            "input_columns": list(t["num_cols"]) + list(t["cat_cols"]), "cat_columns": list(t["cat_cols"]),
            "preprocessor": t["pre"], "feature_names": t["feature_names"], "xgb_models": t["xgb_models"],
            "nn_models": t["nn_models"], "xgb_weight": float(t["xgb_weight"]), "class_scales": np.asarray(t["class_scales"], dtype=np.float32)}


def leave_one_session_out(df: pd.DataFrame, num_cols, cat_cols, xgb_cfg, n_rounds, train_cap=5_000, log=print):
    """[REV2] Hold out one capture session at a time, train on the others (preprocessing fitted on training sessions only)."""
    import xgboost as xgb
    y_sess = df["Label_grouped"].to_numpy()
    sess = df["session"].to_numpy()
    X = df[num_cols + cat_cols]
    rng = np.random.default_rng(SEED)
    pred = np.empty(len(df), dtype=object)
    rows = []
    for held in sorted(np.unique(sess)):
        te, tr = np.where(sess == held)[0], np.where(sess != held)[0]
        present = np.array(sorted(np.unique(y_sess[tr])))
        cmap = {c: i for i, c in enumerate(present)}
        parts = []
        for c in present:
            p = tr[y_sess[tr] == c]
            parts.append(rng.choice(p, size=min(len(p), train_cap), replace=False))
        fit_pos = rng.permutation(np.concatenate(parts))
        y_fit = np.array([cmap[c] for c in y_sess[fit_pos]], dtype=np.int64)
        pre = build_preproc(num_cols, cat_cols)
        M = np.asarray(pre.fit_transform(_prep_frame(X.iloc[fit_pos], cat_cols)), dtype=np.float32)
        cnt = np.bincount(y_fit, minlength=len(present)).astype(np.float64)
        w_cls = ((cnt.max() / cnt) ** xgb_cfg["wpow"]).astype(np.float32)
        params = dict(XGB_BASE_PARAMS, max_depth=xgb_cfg["max_depth"], min_child_weight=xgb_cfg["mcw"],
                      learning_rate=xgb_cfg["lr"], random_state=SEED)
        mdl = xgb.XGBClassifier(n_estimators=n_rounds, **params)
        mdl.fit(M, y_fit, sample_weight=w_cls[y_fit], verbose=False)
        M_te = np.asarray(pre.transform(_prep_frame(X.iloc[te], cat_cols)), dtype=np.float32)
        pred[te] = present[mdl.predict_proba(M_te).argmax(axis=1)]
        yt, yp = y_sess[te], pred[te]
        is_ben, is_att = yt == "BENIGN", yt != "BENIGN"
        seen = is_att & np.isin(yt, present)
        unseen = is_att & ~np.isin(yt, present)
        rows.append({"held-out session": held, "flows": len(te),
                     "classes in session": ", ".join(f"{k}:{v}" for k, v in pd.Series(yt).value_counts().items()),
                     "BENIGN false-alarm rate": float((yp[is_ben] != "BENIGN").mean()) if is_ben.any() else np.nan,
                     "attack detection rate": float((yp[is_att] != "BENIGN").mean()) if is_att.any() else np.nan,
                     "exact-class recall (class seen in training)": float((yp[seen] == yt[seen]).mean()) if seen.any() else np.nan,
                     "detection rate (class NOT in training)": float((yp[unseen] != "BENIGN").mean()) if unseen.any() else np.nan,
                     "unseen classes": ", ".join(sorted(set(yt[unseen]))) or "-"})
        log(f"held out {held}")
    return pd.DataFrame(rows).set_index("held-out session"), pred


# --------------------------------------------------------------------------------------
# Logistic baseline and leakage checks
# --------------------------------------------------------------------------------------
def logreg_baseline(t: dict, y_test: pd.Series, class_weight="balanced"):
    """The notebook's logistic baseline (L2, saga, C=1, class_weight='balanced') on an already preprocessed setup."""
    from sklearn.linear_model import LogisticRegression
    cls = np.array(t["class_names"])
    lr = LogisticRegression(C=1.0, solver="saga", max_iter=100, tol=1e-3, class_weight=class_weight, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(t["mat_pool"], t["y_pool_codes"])
    return evaluate(y_test.to_numpy(), cls[lr.predict(t["mat_test"])], list(t["class_names"]))


def leakage_checks(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                   feature_columns: list[str], numeric_features: list[str], sessions_train, sessions_test, sample=50_000):
    """[REV2] Re-implementation of the notebook's feature-leakage cells."""
    from scipy.stats import rankdata
    out = {}
    forbidden = {"flow id", "source ip", "destination ip", "src ip", "dst ip", "timestamp", "source port", "session",
                 "label", "label_grouped", "binary_label", "multiclass_label", "is_attack"}
    out["forbidden_columns_used"] = [c for c in feature_columns if c.strip().lower() in forbidden]

    yb = (y_train != "BENIGN").to_numpy()
    ix = np.random.default_rng(SEED).choice(len(X_train), size=min(sample, len(X_train)), replace=False)
    scr = X_train[numeric_features].iloc[ix]
    scr = scr.fillna(scr.median())
    ybs = yb[ix]
    n1, n0 = int(ybs.sum()), int((~ybs).sum())
    auc = {c: (rankdata(scr[c].to_numpy())[ybs].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1) for c in numeric_features}
    strength = pd.Series({c: max(a, 1 - a) for c, a in auc.items()}).sort_values(ascending=False)
    out["single_feature_auc"] = strength
    out["suspicious_auc_gt_099"] = list(strength[strength > 0.99].index)

    cols = [c for c in feature_columns if c in X_train.columns]
    h_tr = pd.util.hash_pandas_object(X_train[cols].astype(str), index=False).to_numpy()
    h_te = pd.util.hash_pandas_object(X_test[cols].astype(str), index=False).to_numpy()
    overlap = np.isin(h_te, h_tr)
    out["overlap_share"] = float(overlap.mean())
    out["overlap_by_class"] = pd.Series(overlap, index=y_test.index).groupby(y_test).mean().round(3)

    s_tr, s_te = y_train.value_counts(normalize=True), y_test.value_counts(normalize=True)
    out["max_class_share_gap"] = float((s_tr - s_te.reindex(s_tr.index)).abs().max())
    out["sessions_train"], out["sessions_test"] = int(pd.Series(sessions_train).nunique()), int(pd.Series(sessions_test).nunique())
    return out
