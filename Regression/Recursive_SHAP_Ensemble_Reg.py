"""
End-to-end regression ensemble with CatBoost recursive SHAP feature selection.

Pipeline:
  1. Load training data and keep true electrons only.
  2. Split once into train / validation / local test.
  3. Either:
       a. Run CatBoost recursive SHAP elimination to select the best 20 features, OR
       b. Bypass SHAP and use the preset PRESET_SELECTED_FEATURES list.
  4. For each of XGBoost, LightGBM, CatBoost, NN:
       a. Optionally run RandomizedSearchCV (k-fold CV on X_train) to find params, OR
       b. Bypass search and use the preset params dict for that model.
     The NN uses a custom k-fold random search since it is not sklearn-compatible.
  5. Refit each model on (X_train, X_val) with early stopping where applicable.
  6. Choose ensemble weights on the validation split.
  7. Report local train / validation / test RelMAD.
  8. Predict the official regression test set and write only ensemble submission files.

Important:
  - Component models are not saved to disk.
  - Only the final ensemble submission CSV and variable-list CSV are written.
  - All models are trained on log(E_true); the ensemble is a weighted geometric
    mean, implemented as a weighted average in log-space.
  - Run with USE_RANDOM_SEARCH all True once; paste the printed best_params into
    the *_PRESET_PARAMS dicts and flip the toggles to False for fast reruns.

Run from anywhere — paths are anchored to PROJECT_ROOT.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#os.environ["OMP_NUM_THREADS"] = "1"

# Make Lightning's rank-zero info logs visible (the "GPU available: ..." banner
# and similar messages route through Python logging, not stdout).
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

# Avoid OpenMP deadlock between torch and the tree-boosting libs on macOS.
# After XGBoost/LightGBM/CatBoost initialize their thread pools, torch's first
# CPU op can hang. Forcing torch to 1 thread sidesteps it; the NN is small
# enough that this costs ~nothing.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from pytorch_lightning.callbacks import EarlyStopping
from scipy.stats import loguniform as sp_loguniform
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor


# ============================================================ CONFIG
PROJECT_ROOT = Path("/Users/prometheus/Documents/Python/Electron_Project")
DATA_DIR     = PROJECT_ROOT / "Data"
SUBMIT_DIR   = PROJECT_ROOT / "Submission"

TRAIN_H5 = DATA_DIR / "AppML_InitialProject_train.h5"
TEST_H5  = DATA_DIR / "AppML_InitialProject_test_regression.h5"

TARGET_COL = "p_Truth_Energy"
ELECTRON_FLAG_COL = "p_Truth_isElectron"
SUBMITTER  = "RasmusReimer"
MODEL_NAME = "CatSHAP20_XGB_LGB_CAT_NN_WeightedEnsemble"

TEST_SIZE    = 0.20
VAL_SIZE     = 0.10
RANDOM_STATE = 42

N_FINAL_FEATURES = 20
CATEGORICAL_FEATURES = ["pX_ambiguityType"]

# ---------- Feature-selection toggle ----------
# True  -> run recursive CatBoost SHAP elimination down to N_FINAL_FEATURES.
# False -> bypass SHAP and use PRESET_SELECTED_FEATURES (from a prior run).
USE_RECURSIVE_SHAP = False

PRESET_SELECTED_FEATURES: list[str] = [
    "pX_ecore",
    "p_pt_track",
    "pX_MultiLepton",
    "pX_E3x5_Lr1",
    "pX_topoetcone20",
    "pX_e233",
    "pX_deltaPhiFromLastMeasurement",
    "p_sigmad0",
    "p_ptcone40",
    "p_etcone20",
    "pX_E_Lr0_HiG",
    "pX_deltaPhi2",
    "pX_maxEcell_energy",
    "pX_E_Lr2_HiG",
    "p_eta",
    "pX_deltaEta1",
    "pX_maxEcell_z",
    "p_Eratio",
    "pX_topoetcone40ptCorrection",
    "p_deltaPhiRescaled2",
]

# Recursive SHAP feature-selection settings (used only when USE_RECURSIVE_SHAP=True).
SHAP_SELECT_ITERATIONS = 1_500
SHAP_SELECT_OD_WAIT = 100
SHAP_SELECT_VERBOSE = False
SHAP_SELECTION_SAMPLE = 30_000       # sample rows for SHAP calculation; set None to use all train rows
SHAP_REMOVE_PER_ROUND = 1            # true recursive elimination; increase to speed up

# If None, candidate features are inferred automatically from train/test columns.
CANDIDATE_FEATURES: list[str] | None = None

# ---------- RandomizedSearchCV toggles (per model) ----------
# True  -> run RandomizedSearchCV on X_train with k-fold CV, then refit with early stopping.
# False -> skip the search and use the corresponding *_PRESET_PARAMS dict.
# After one search pass, paste the printed best_params into *_PRESET_PARAMS and flip to False.
USE_RANDOM_SEARCH = {
    "XGB":      False,
    "LGB":      False,
    "CatBoost": False,
    "NN":       False,
}

RS_N_ITER = 60         # candidates per model
RS_CV     = 5          # k-fold CV
RS_SEED   = 42
RS_N_JOBS = 1          # >1 parallelizes CV fits; tree models already multithread internally
RS_TREE_ITERATIONS = 500  # fixed n_estimators / iterations during search (no early stopping)

# Preset params (used when USE_RANDOM_SEARCH[name] is False).
XGB_PRESET_PARAMS = {
    "learning_rate":    0.024775725106121492,
    "max_depth":        9,
    "min_child_weight": 6,
    "gamma":            0.0033083055212689886,
    "subsample":        0.6873063073132356,
    "colsample_bytree": 0.7367358853902828,
    "reg_alpha":        0.34703605955441524,
    "reg_lambda":       0.0015178052094448862,
}

LGB_PRESET_PARAMS = {
    "learning_rate":     0.03199075386502134,
    "num_leaves":        83,
    "min_child_samples": 10,
    "colsample_bytree":  0.8129299578571182,
    "subsample":         0.9719458022803786,
    "subsample_freq":    1,
    "reg_alpha":         0.5443209847025573,
    "reg_lambda":        0.29067297631440864,
}

CAT_PRESET_PARAMS = {
    "learning_rate":       0.10220655100897388,
    "depth":               6,
    "l2_leaf_reg":         0.7148510793512985,
    "bagging_temperature": 0.5442644987692706,
    "random_strength":     1.7214611665126869,
}

# NN training device toggle.
# True  -> use GPU/MPS if available (cuda > mps > cpu).
# False -> force CPU (workaround for MPS hangs on some torch/macOS combos).
NN_USE_GPU = False

NN_PRESET_PARAMS = {
    "first_layer":  128,
    "second_layer": 256,
    "third_layer":  64,
    "dropout":      0.04305224728593615,
    "lr":           0.0014369944935679968,
    "weight_decay": 0.0028348524707913665,
    "batch_size":   64,
    "max_epochs":   300,
    "patience":     25,
}

# Param search spaces (used when the corresponding toggle is True).
XGB_PARAM_SPACE = {
    "learning_rate":    sp_loguniform(5e-3, 2e-1),
    "max_depth":        sp_randint(3, 12),
    "min_child_weight": sp_randint(1, 10),
    "gamma":            sp_loguniform(1e-3, 1.0),
    "subsample":        sp_uniform(0.5, 0.5),       # uniform on [0.5, 1.0]
    "colsample_bytree": sp_uniform(0.5, 0.5),
    "reg_alpha":        sp_loguniform(1e-3, 10),
    "reg_lambda":       sp_loguniform(1e-3, 10),
}

LGB_PARAM_SPACE = {
    "learning_rate":     sp_loguniform(5e-3, 2e-1),
    "num_leaves":        sp_randint(15, 256),
    "min_child_samples": sp_randint(10, 100),
    "colsample_bytree":  sp_uniform(0.5, 0.5),
    "subsample":         sp_uniform(0.5, 0.5),
    "subsample_freq":    sp_randint(0, 10),
    "reg_alpha":         sp_loguniform(1e-3, 10),
    "reg_lambda":        sp_loguniform(1e-3, 10),
}

CAT_PARAM_SPACE = {
    "learning_rate":       sp_loguniform(5e-3, 2e-1),
    "depth":               sp_randint(4, 11),
    "l2_leaf_reg":         sp_loguniform(1e-1, 10),
    "bagging_temperature": sp_uniform(0.0, 2.0),
    "random_strength":     sp_uniform(0.0, 2.0),
}

NN_PARAM_SPACE = {
    "first_layer":   [32, 64, 128, 256],
    "second_layer":  [32, 64, 128, 256],
    "third_layer":   [32, 64, 128, 256],
    "dropout":       (0.0, 0.5),       # uniform range
    "lr":            (1e-4, 1e-2),     # log-uniform range
    "weight_decay":  (1e-5, 1e-1),     # log-uniform range
    "batch_size":    [64, 128, 256, 512],
}

# Ensemble weighting.
OPTIMIZE_WEIGHTS = True
N_WEIGHT_SAMPLES = 20_000
WEIGHT_SEARCH_SEED = 123
MANUAL_WEIGHTS = {
    "XGB": 0.35,
    "LGB": 0.35,
    "CatBoost": 0.25,
    "NN": 0.05,
}

# Upper cap on predicted log(E_pred). Caps each model's log-prediction before
# the ensemble average; set None to disable. exp(14) ~ 1.2 M GeV.
LOG_E_CLIP = 14.0

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================ NN model
class ThreeLayerRegressor(nn.Module):
    def __init__(self, input_size: int, first_layer: int, second_layer: int,
                 third_layer: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_size, first_layer)
        self.fc2 = nn.Linear(first_layer, second_layer)
        self.fc3 = nn.Linear(second_layer, third_layer)
        self.fc4 = nn.Linear(third_layer, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        return self.fc4(x).view(-1)


class ThreeLayerLitModule(pl.LightningModule):
    """LightningModule wrapper around ThreeLayerRegressor.

    Logs train_mse / val_mse each epoch and uses ReduceLROnPlateau on val_mse,
    matching the previous bare-PyTorch loop.
    """

    def __init__(self, input_size: int, first_layer: int, second_layer: int,
                 third_layer: int, dropout: float, lr: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters()
        self.net = ThreeLayerRegressor(
            input_size=input_size,
            first_layer=first_layer,
            second_layer=second_layer,
            third_layer=third_layer,
            dropout=dropout,
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train_mse", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("val_mse", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mse",
                "interval": "epoch",
            },
        }


class EpochProgressBar(pl.Callback):
    """Single tqdm bar that advances one tick per validation epoch.

    Replaces the per-batch + per-epoch text output. Shows current train/val MSE
    in the bar's postfix so you can still watch convergence at a glance.
    """

    def __init__(self):
        super().__init__()
        self.bar = None

    def on_train_start(self, trainer, pl_module):
        from tqdm.auto import tqdm
        self.bar = tqdm(
            total=trainer.max_epochs,
            desc="NN epochs",
            unit="epoch",
            dynamic_ncols=True,
            leave=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.bar is None:
            return
        m = trainer.callback_metrics
        postfix = {}
        tr = m.get("train_mse")
        va = m.get("val_mse")
        if tr is not None:
            postfix["train"] = f"{float(tr):.5f}"
        if va is not None:
            postfix["val"] = f"{float(va):.5f}"
        if postfix:
            self.bar.set_postfix(postfix, refresh=False)
        self.bar.update(1)

    def on_train_end(self, trainer, pl_module):
        if self.bar is not None:
            self.bar.close()
            self.bar = None


class BestStateCallback(pl.Callback):
    """Track the best val_mse weights in memory so we can avoid disk checkpoints."""

    def __init__(self, monitor: str = "val_mse"):
        super().__init__()
        self.monitor = monitor
        self.best = float("inf")
        self.best_state = None
        self.best_epoch = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        v = float(metric)
        if v < self.best:
            self.best = v
            self.best_state = copy.deepcopy(pl_module.state_dict())
            self.best_epoch = trainer.current_epoch + 1


def lightning_accelerator() -> str:
    if not NN_USE_GPU:
        return "cpu"
    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================ utilities
def relmad(y_true, y_pred) -> float:
    """Project metric: mean(|E_pred - E_true| / E_true)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true) / y_true))


def relmad_from_log(y_true_log, y_pred_log) -> float:
    e_true = np.exp(np.asarray(y_true_log, dtype=float))
    e_pred = np.exp(np.asarray(y_pred_log, dtype=float))
    return float(np.mean(np.abs(e_pred - e_true) / e_true))


# Used by RandomizedSearchCV so the search optimizes RelMAD on E, not on log E.
relmad_scorer = make_scorer(relmad_from_log, greater_is_better=False)


def infer_candidate_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    """Infer usable regression features (intersection minus truth/label columns)."""
    if CANDIDATE_FEATURES is not None:
        return list(CANDIDATE_FEATURES)

    common_cols = [c for c in train_df.columns if c in test_df.columns]

    excluded_exact = {
        TARGET_COL,
        ELECTRON_FLAG_COL,
        "p_Truth_PDGID",
        "p_Truth_pdgId",
        "p_Truth_MotherID",
        "p_Truth_Origin",
        "p_Truth_Type",
    }

    features = []
    for col in common_cols:
        if col in excluded_exact:
            continue
        if col.startswith("p_Truth"):
            continue
        if col.startswith("Truth"):
            continue
        if col.lower() in {"index", "eventnumber", "runnumber"}:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            features.append(col)

    if len(features) < N_FINAL_FEATURES:
        raise ValueError(
            f"Only found {len(features)} usable candidate features, "
            f"but N_FINAL_FEATURES={N_FINAL_FEATURES}."
        )

    return features


def prepare_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Features missing from data: {missing}")

    X = df[features].copy()

    for cat in CATEGORICAL_FEATURES:
        if cat in X.columns:
            X.loc[:, cat] = X[cat].astype("int32")

    return X


def active_cat_features(features: list[str]) -> list[str]:
    return [f for f in CATEGORICAL_FEATURES if f in features]


def make_cat_pool(X: pd.DataFrame, y_log=None) -> Pool:
    cat_features = active_cat_features(list(X.columns))
    return Pool(data=X, label=y_log, cat_features=cat_features)


def log_energy(y) -> np.ndarray:
    return np.log(np.asarray(y, dtype=float))


def normalize_weights(weights_dict: dict[str, float], model_names: list[str]) -> np.ndarray:
    weights = np.array([float(weights_dict[name]) for name in model_names], dtype=float)
    if np.any(weights < 0):
        raise ValueError(f"Weights must be non-negative. Got {weights_dict}")
    total = weights.sum()
    if total <= 0:
        raise ValueError(f"At least one weight must be positive. Got {weights_dict}")
    return weights / total


def weighted_log_average(log_pred_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(log_pred_matrix, axis=1, weights=weights)


def print_relmad_table(title: str, y_true, log_pred_by_name: dict[str, np.ndarray],
                       weights: np.ndarray | None = None):
    print(f"\n=== {title} ===")
    names = list(log_pred_by_name.keys())
    for name in names:
        rm = relmad(y_true, np.exp(log_pred_by_name[name]))
        print(f"  {name:9s}  RelMAD = {rm:.5f}")

    if weights is not None:
        mat = np.column_stack([log_pred_by_name[name] for name in names])
        ens = weighted_log_average(mat, weights)
        rm = relmad(y_true, np.exp(ens))
        print(f"  {'Ensemble':9s}  RelMAD = {rm:.5f}")


# ============================================================ data loading
def load_data():
    print(f"Loading train data: {TRAIN_H5}")
    train_df = pd.read_hdf(TRAIN_H5)
    train_df = train_df[train_df[ELECTRON_FLAG_COL] == 1].copy()
    print(f"After electron filter: {len(train_df)} rows")

    print(f"Loading official test data: {TEST_H5}")
    official_test_df = pd.read_hdf(TEST_H5)
    if TARGET_COL in official_test_df.columns:
        official_test_df = official_test_df.drop(columns=[TARGET_COL])
    print(f"Official test set: {official_test_df.shape[0]} rows, {official_test_df.shape[1]} columns")

    candidate_features = infer_candidate_features(train_df, official_test_df)
    print(f"Candidate features for recursive SHAP: {len(candidate_features)}")

    y = train_df[TARGET_COL]
    X = prepare_features(train_df, candidate_features)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=RANDOM_STATE,
    )

    print(f"Split: train {len(X_train)}  val {len(X_val)}  local test {len(X_test)}")
    print(f"Train E range: [{y_train.min():.1f}, {y_train.max():.1f}] GeV")

    return X_train, X_val, X_test, y_train, y_val, y_test, official_test_df


# ============================================================ recursive CatBoost SHAP
def train_catboost_for_shap(X_train, X_val, y_train, y_val) -> CatBoostRegressor:
    log_y_train = log_energy(y_train)
    log_y_val = log_energy(y_val)

    model = CatBoostRegressor(
        iterations=SHAP_SELECT_ITERATIONS,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        eval_metric="MAPE",
        random_seed=RANDOM_STATE,
        od_type="Iter",
        od_wait=SHAP_SELECT_OD_WAIT,
        verbose=SHAP_SELECT_VERBOSE,
        bagging_temperature=1.0,
        random_strength=1.0,
        allow_writing_files=False,
    )

    model.fit(
        make_cat_pool(X_train, log_y_train),
        eval_set=make_cat_pool(X_val, log_y_val),
        use_best_model=True,
    )
    return model


def mean_abs_shap_importance(model: CatBoostRegressor, X: pd.DataFrame) -> pd.Series:
    if SHAP_SELECTION_SAMPLE is not None and len(X) > SHAP_SELECTION_SAMPLE:
        X_shap = X.sample(n=SHAP_SELECTION_SAMPLE, random_state=RANDOM_STATE)
    else:
        X_shap = X

    shap_values = model.get_feature_importance(
        make_cat_pool(X_shap),
        type="ShapValues",
    )
    feature_shap = shap_values[:, :-1]
    imp = pd.Series(np.abs(feature_shap).mean(axis=0), index=X.columns)
    return imp.sort_values(ascending=False)


def recursive_catboost_shap_select(X_train, X_val, y_train, y_val) -> list[str]:
    """Recursively remove the lowest-SHAP features until N_FINAL_FEATURES remain."""
    remaining = list(X_train.columns)
    round_idx = 1

    print(f"\nStarting recursive CatBoost SHAP selection: {len(remaining)} -> {N_FINAL_FEATURES} features")

    while len(remaining) > N_FINAL_FEATURES:
        print(f"\nSHAP round {round_idx}: fitting CatBoost with {len(remaining)} features")

        Xtr = X_train[remaining]
        Xva = X_val[remaining]
        model = train_catboost_for_shap(Xtr, Xva, y_train, y_val)
        imp = mean_abs_shap_importance(model, Xtr)

        n_remove = min(SHAP_REMOVE_PER_ROUND, len(remaining) - N_FINAL_FEATURES)
        to_remove = list(imp.tail(n_remove).index)

        print("  Removing:")
        for f in to_remove:
            print(f"    {f:40s}  mean_abs_SHAP={imp[f]:.6g}")

        remaining = [f for f in remaining if f not in to_remove]
        round_idx += 1

    print("\nFinal SHAP ranking on selected features")
    final_model = train_catboost_for_shap(X_train[remaining], X_val[remaining], y_train, y_val)
    final_imp = mean_abs_shap_importance(final_model, X_train[remaining])
    selected = list(final_imp.index)

    print("\nSelected 20 features:")
    for i, f in enumerate(selected, start=1):
        print(f"  {i:2d}. {f:40s}  mean_abs_SHAP={final_imp[f]:.6g}")

    return selected


# ============================================================ random search helpers
def random_search_estimator(estimator, param_space, X, y_log, name, fit_params=None):
    """Generic RandomizedSearchCV wrapper that scores RelMAD on E (via relmad_scorer).

    verbose=3 makes sklearn print each (candidate x fold) fit on its own line as
    `[CV n/N] END <params> ; score=... time=...s`. With n_jobs=1 those lines
    stream live; with n_jobs>1 joblib batches them.
    """
    total_fits = RS_N_ITER * RS_CV
    print(
        f"\nRandom search for {name}: n_iter={RS_N_ITER}, cv={RS_CV} "
        f"({total_fits} fits, n_jobs={RS_N_JOBS}, refit=False)",
        flush=True,
    )
    t0 = time.time()
    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_space,
        n_iter=RS_N_ITER,
        cv=RS_CV,
        scoring=relmad_scorer,
        random_state=RS_SEED,
        n_jobs=RS_N_JOBS,
        verbose=3,
        refit=False,
    )
    rs.fit(X, y_log, **(fit_params or {}))
    elapsed = time.time() - t0
    best = dict(rs.best_params_)
    print(f"  {name} random search took {elapsed/60:.1f} min", flush=True)
    print(f"  Best {name} params: {best}", flush=True)
    print(f"  Best {name} CV RelMAD: {-rs.best_score_:.5f}", flush=True)
    return best


def sample_nn_params(rng: np.random.Generator) -> dict:
    lr_lo, lr_hi   = NN_PARAM_SPACE["lr"]
    wd_lo, wd_hi   = NN_PARAM_SPACE["weight_decay"]
    drop_lo, drop_hi = NN_PARAM_SPACE["dropout"]
    return {
        "first_layer":  int(rng.choice(NN_PARAM_SPACE["first_layer"])),
        "second_layer": int(rng.choice(NN_PARAM_SPACE["second_layer"])),
        "third_layer":  int(rng.choice(NN_PARAM_SPACE["third_layer"])),
        "dropout":      float(rng.uniform(drop_lo, drop_hi)),
        "lr":           float(10 ** rng.uniform(np.log10(lr_lo), np.log10(lr_hi))),
        "weight_decay": float(10 ** rng.uniform(np.log10(wd_lo), np.log10(wd_hi))),
        "batch_size":   int(rng.choice(NN_PARAM_SPACE["batch_size"])),
        "max_epochs":   NN_PRESET_PARAMS["max_epochs"],
        "patience":     NN_PRESET_PARAMS["patience"],
    }


def nn_random_search(X_train, y_train) -> dict:
    """Custom k-fold random search for the NN regressor, scored on RelMAD."""
    print(f"\nRandom search for NN: n_iter={RS_N_ITER}, cv={RS_CV} "
          f"({RS_N_ITER * RS_CV} fits)", flush=True)
    rng = np.random.default_rng(RS_SEED)
    candidates = [sample_nn_params(rng) for _ in range(RS_N_ITER)]

    X_arr = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
    y_arr = np.asarray(y_train)
    kf = KFold(n_splits=RS_CV, shuffle=True, random_state=RS_SEED)

    best_score = float("inf")
    best_params = None
    search_t0 = time.time()

    for ci, params in enumerate(candidates, start=1):
        print(
            f"\n  candidate {ci:2d}/{RS_N_ITER}: "
            f"layers=({params['first_layer']},{params['second_layer']},{params['third_layer']}) "
            f"dropout={params['dropout']:.4f} lr={params['lr']:.2e} "
            f"wd={params['weight_decay']:.2e} bs={params['batch_size']}",
            flush=True,
        )
        cand_t0 = time.time()
        fold_scores = []
        for fi, (tr_idx, va_idx) in enumerate(kf.split(X_arr), start=1):
            fold_t0 = time.time()
            Xtr, Xva = X_arr[tr_idx], X_arr[va_idx]
            ytr, yva = y_arr[tr_idx], y_arr[va_idx]

            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr)
            Xva_s = scaler.transform(Xva)

            train_loader = make_loader(Xtr_s, log_energy(ytr), params["batch_size"], shuffle=True)
            val_loader   = make_loader(Xva_s, log_energy(yva), params["batch_size"], shuffle=False)

            lit_model, best_epoch, _ = train_nn_lightning(
                input_size=Xtr.shape[1],
                train_loader=train_loader,
                val_loader=val_loader,
                params=params,
                enable_progress_bar=True,
            )

            device = next(lit_model.parameters()).device
            with torch.no_grad():
                Xva_t = torch.from_numpy(
                    np.ascontiguousarray(Xva_s, dtype=np.float32)
                ).to(device)
                preds = lit_model(Xva_t).cpu().numpy()

            score = relmad(yva, np.exp(preds))
            fold_scores.append(score)
            print(
                f"    fold {fi}/{RS_CV}: RelMAD={score:.5f}  "
                f"best_epoch={best_epoch}  took {time.time()-fold_t0:.1f}s",
                flush=True,
            )

        mean_score = float(np.mean(fold_scores))
        improved = mean_score < best_score
        if improved:
            best_score = mean_score
            best_params = params
        print(
            f"    -> mean CV RelMAD = {mean_score:.5f}  "
            f"(best so far {best_score:.5f}{'  *NEW BEST*' if improved else ''})  "
            f"candidate took {time.time()-cand_t0:.1f}s  "
            f"elapsed {(time.time()-search_t0)/60:.1f} min",
            flush=True,
        )

    print(f"\n  NN random search took {(time.time()-search_t0)/60:.1f} min", flush=True)
    print(f"  Best NN params: {best_params}", flush=True)
    print(f"  Best NN CV RelMAD: {best_score:.5f}", flush=True)
    return best_params


# ============================================================ model training
def train_xgb(X_train, X_val, y_train, y_val) -> XGBRegressor:
    if USE_RANDOM_SEARCH["XGB"]:
        base = XGBRegressor(
            n_estimators=RS_TREE_ITERATIONS,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        best_params = random_search_estimator(
            base, XGB_PARAM_SPACE, X_train, log_energy(y_train), name="XGB",
        )
    else:
        best_params = dict(XGB_PRESET_PARAMS)
        print(f"\nUsing preset XGB params: {best_params}")

    print("\nTraining XGBoost regressor on selected features...")
    model = XGBRegressor(
        n_estimators=10_000,
        objective="reg:squarederror",
        eval_metric="mape",
        early_stopping_rounds=200,
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
        **best_params,
    )
    model.fit(
        X_train, log_energy(y_train),
        eval_set=[(X_train, log_energy(y_train)), (X_val, log_energy(y_val))],
        verbose=200,
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


def train_lgb(X_train, X_val, y_train, y_val) -> LGBMRegressor:
    cat_feats = active_cat_features(list(X_train.columns))

    if USE_RANDOM_SEARCH["LGB"]:
        base = LGBMRegressor(
            n_estimators=RS_TREE_ITERATIONS,
            objective="regression",
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        fit_params = {"categorical_feature": cat_feats} if cat_feats else None
        best_params = random_search_estimator(
            base, LGB_PARAM_SPACE, X_train, log_energy(y_train),
            name="LGB", fit_params=fit_params,
        )
    else:
        best_params = dict(LGB_PRESET_PARAMS)
        print(f"\nUsing preset LGB params: {best_params}")

    print("\nTraining LightGBM regressor on selected features...")
    model = LGBMRegressor(
        n_estimators=10_000,
        objective="regression",
        metric="mape",
        random_state=RANDOM_STATE,
        verbose=-1,
        **best_params,
    )
    fit_kwargs = dict(
        eval_set=[(X_val, log_energy(y_val))],
        eval_metric="mape",
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=True),
            lgb.log_evaluation(period=200),
        ],
    )
    if cat_feats:
        fit_kwargs["categorical_feature"] = cat_feats
    model.fit(X_train, log_energy(y_train), **fit_kwargs)
    print(f"  Best iteration: {model.best_iteration_}")
    return model


def train_catboost_final(X_train, X_val, y_train, y_val) -> CatBoostRegressor:
    cat_feats = active_cat_features(list(X_train.columns))

    if USE_RANDOM_SEARCH["CatBoost"]:
        base = CatBoostRegressor(
            iterations=RS_TREE_ITERATIONS,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            cat_features=cat_feats or None,
            verbose=False,
            allow_writing_files=False,
        )
        best_params = random_search_estimator(
            base, CAT_PARAM_SPACE, X_train, log_energy(y_train), name="CatBoost",
        )
    else:
        best_params = dict(CAT_PRESET_PARAMS)
        print(f"\nUsing preset CatBoost params: {best_params}")

    print("\nTraining final CatBoost regressor on selected features...")
    model = CatBoostRegressor(
        iterations=10_000,
        loss_function="RMSE",
        eval_metric="MAPE",
        random_seed=RANDOM_STATE,
        od_type="Iter",
        od_wait=200,
        verbose=200,
        cat_features=cat_feats or None,
        allow_writing_files=False,
        **best_params,
    )
    model.fit(
        X_train, log_energy(y_train),
        eval_set=(X_val, log_energy(y_val)),
        use_best_model=True,
    )
    print(f"  Best iteration: {model.get_best_iteration()}")
    return model


def make_loader(X, y_log, batch_size, shuffle):
    # torch.tensor(np_array, dtype=...) can hang on first call with torch 2.11 + macOS
    # while it probes accelerators. from_numpy on an already-typed array avoids it.
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    y_np = np.ascontiguousarray(np.asarray(y_log, dtype=np.float32))
    ds = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_nn_lightning(input_size: int, train_loader, val_loader, params: dict,
                       enable_progress_bar: bool = True):
    """Train one NN with pytorch-lightning. Returns (lit_model, best_epoch, best_val_mse).

    Disk checkpointing and TensorBoard logging are disabled; best weights are
    tracked in memory via BestStateCallback and loaded back at the end.
    """
    print(f"    [nn] building LightningModule (input_size={input_size})", flush=True)
    lit_model = ThreeLayerLitModule(
        input_size=input_size,
        first_layer=params["first_layer"],
        second_layer=params["second_layer"],
        third_layer=params["third_layer"],
        dropout=params["dropout"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    best_cb = BestStateCallback(monitor="val_mse")
    early_cb = EarlyStopping(
        monitor="val_mse",
        mode="min",
        patience=params["patience"],
    )
    callbacks = [early_cb, best_cb]
    if enable_progress_bar:
        callbacks.append(EpochProgressBar())
    acc = lightning_accelerator()
    print(f"    [nn] constructing pl.Trainer (accelerator={acc})", flush=True)
    t_ctor = time.time()
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        accelerator=acc,
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=False,   # disable Lightning's batch-level bar; EpochProgressBar handles it
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        deterministic=False,
        num_sanity_val_steps=0,      # skip pre-fit sanity check
    )
    print(f"    [nn] Trainer ready in {time.time()-t_ctor:.2f}s; calling fit()", flush=True)
    trainer.fit(lit_model, train_loader, val_loader)
    if best_cb.best_state is not None:
        lit_model.load_state_dict(best_cb.best_state)
    lit_model.eval()
    return lit_model, best_cb.best_epoch, best_cb.best


def train_nn(X_train, X_val, y_train, y_val):
    if USE_RANDOM_SEARCH["NN"]:
        searched = nn_random_search(X_train, y_train)
        params = {**NN_PRESET_PARAMS, **searched}
    else:
        params = dict(NN_PRESET_PARAMS)
        print(f"\nUsing preset NN params: {params}")

    print(
        f"\nTraining NN regressor on selected features "
        f"(Lightning, accelerator={lightning_accelerator()})",
        flush=True,
    )
    print(f"  [nn] X_train shape={X_train.shape}  X_val shape={X_val.shape}", flush=True)

    t0 = time.time()
    scaler = StandardScaler().fit(X_train)
    print(f"  [nn] scaler.fit done in {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    print(f"  [nn] scaler.transform done in {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    train_loader = make_loader(
        X_train_s, log_energy(y_train), params["batch_size"], shuffle=True,
    )
    val_loader = make_loader(
        X_val_s, log_energy(y_val), params["batch_size"], shuffle=False,
    )
    print(
        f"  [nn] dataloaders built in {time.time()-t0:.2f}s  "
        f"(train batches={len(train_loader)}, val batches={len(val_loader)})",
        flush=True,
    )

    lit_model, best_epoch, best_val = train_nn_lightning(
        input_size=X_train.shape[1],
        train_loader=train_loader,
        val_loader=val_loader,
        params=params,
        enable_progress_bar=True,
    )
    print(f"  Best epoch: {best_epoch}  best_val_mse={best_val:.5f}")
    return lit_model, scaler


# ============================================================ prediction
def predict_log_xgb(model, X):
    return model.predict(X)


def predict_log_lgb(model, X):
    return model.predict(X)


def predict_log_cat(model, X):
    return model.predict(X)


def predict_log_nn(model, scaler, X):
    X_s = np.ascontiguousarray(scaler.transform(X), dtype=np.float32)
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        X_t = torch.from_numpy(X_s).to(device)
        return model(X_t).cpu().numpy()


def clip_log_pred(log_pred: np.ndarray) -> np.ndarray:
    if LOG_E_CLIP is None:
        return log_pred
    return np.minimum(log_pred, LOG_E_CLIP)


def predict_all_log(models: dict[str, object], X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "XGB":      clip_log_pred(predict_log_xgb(models["XGB"], X)),
        "LGB":      clip_log_pred(predict_log_lgb(models["LGB"], X)),
        "CatBoost": clip_log_pred(predict_log_cat(models["CatBoost"], X)),
        "NN":       clip_log_pred(predict_log_nn(models["NN"][0], models["NN"][1], X)),
    }


# ============================================================ ensemble weights
def optimize_weights_random_search(log_pred_by_name: dict[str, np.ndarray], y_true):
    model_names = list(log_pred_by_name.keys())
    matrix = np.column_stack([log_pred_by_name[name] for name in model_names])

    if not OPTIMIZE_WEIGHTS:
        weights = normalize_weights(MANUAL_WEIGHTS, model_names)
        return weights, relmad(y_true, np.exp(weighted_log_average(matrix, weights)))

    rng = np.random.default_rng(WEIGHT_SEARCH_SEED)
    n_models = len(model_names)

    candidates = []
    candidates.append(np.ones(n_models) / n_models)
    candidates.append(normalize_weights(MANUAL_WEIGHTS, model_names))

    for i in range(n_models):
        w = np.zeros(n_models)
        w[i] = 1.0
        candidates.append(w)

    candidates.extend(rng.dirichlet(alpha=np.ones(n_models), size=N_WEIGHT_SAMPLES))

    best_rm = np.inf
    best_w = None
    for w in candidates:
        log_ens = weighted_log_average(matrix, w)
        rm = relmad(y_true, np.exp(log_ens))
        if rm < best_rm:
            best_rm = rm
            best_w = w

    return best_w, best_rm


# ============================================================ submission
def write_ensemble_submission(official_test_df: pd.DataFrame, selected_features: list[str],
                              models: dict[str, object], weights: np.ndarray):
    X_submit = prepare_features(official_test_df, selected_features)
    log_pred_by_name = predict_all_log(models, X_submit)
    model_names = list(log_pred_by_name.keys())
    matrix = np.column_stack([log_pred_by_name[name] for name in model_names])

    n_rows = len(X_submit)
    for name in model_names:
        log_p = log_pred_by_name[name]
        e = np.exp(log_p)
        clipped = int(np.sum(log_p >= LOG_E_CLIP)) if LOG_E_CLIP is not None else 0
        clip_pct = 100.0 * clipped / n_rows if n_rows else 0.0
        print(
            f"  {name:9s}  E range = [{e.min():.2f}, {e.max():.2f}] GeV  "
            f"at cap: {clipped}/{n_rows} ({clip_pct:.2f}%)"
        )

    log_ens = weighted_log_average(matrix, weights)
    energy = np.exp(log_ens)
    print(f"  {'Ensemble':9s}  E range = [{energy.min():.2f}, {energy.max():.2f}] GeV")

    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"Regression_{SUBMITTER}_{MODEL_NAME}"

    submission_path = SUBMIT_DIR / f"{base}.csv"
    variable_list_path = SUBMIT_DIR / f"{base}_VariableList.csv"

    pd.DataFrame({"index": official_test_df.index, TARGET_COL: energy}) \
        .to_csv(submission_path, index=False, header=False)

    pd.Series(sorted(selected_features)) \
        .to_csv(variable_list_path, index=False, header=False)

    print(f"\nWrote {submission_path}")
    print(f"Wrote {variable_list_path}")


# ============================================================ main
def select_features(X_train_all, X_val_all, y_train, y_val) -> list[str]:
    if USE_RECURSIVE_SHAP:
        return recursive_catboost_shap_select(X_train_all, X_val_all, y_train, y_val)

    missing = [f for f in PRESET_SELECTED_FEATURES if f not in X_train_all.columns]
    if missing:
        raise KeyError(
            "PRESET_SELECTED_FEATURES contains columns not present in training data: "
            f"{missing}"
        )
    selected = list(PRESET_SELECTED_FEATURES)
    print(f"\nBypassing recursive SHAP; using {len(selected)} preset features:")
    for i, f in enumerate(selected, start=1):
        print(f"  {i:2d}. {f}")
    return selected


def main():
    print(f"{'=' * 72}")
    print("CatBoost recursive-SHAP 20-feature weighted ensemble regression")
    print(f"{'=' * 72}")

    X_train_all, X_val_all, X_test_all, y_train, y_val, y_test, official_test_df = load_data()

    selected_features = select_features(X_train_all, X_val_all, y_train, y_val)

    X_train = prepare_features(X_train_all, selected_features)
    X_val   = prepare_features(X_val_all, selected_features)
    X_test  = prepare_features(X_test_all, selected_features)

    models = {
        "XGB": train_xgb(X_train, X_val, y_train, y_val),
        "LGB": train_lgb(X_train, X_val, y_train, y_val),
        "CatBoost": train_catboost_final(X_train, X_val, y_train, y_val),
        "NN": train_nn(X_train, X_val, y_train, y_val),
    }

    train_preds = predict_all_log(models, X_train)
    val_preds = predict_all_log(models, X_val)
    test_preds = predict_all_log(models, X_test)

    weights, best_val_rm = optimize_weights_random_search(val_preds, y_val)
    model_names = list(val_preds.keys())

    print("\nOptimized ensemble weights:" if OPTIMIZE_WEIGHTS else "\nManual ensemble weights:")
    for name, w in zip(model_names, weights):
        print(f"  {name:9s}  weight = {w:.4f}")
    print(f"  Validation ensemble RelMAD used for weight choice = {best_val_rm:.5f}")

    print_relmad_table("Train RelMAD", y_train, train_preds, weights)
    print_relmad_table("Validation RelMAD", y_val, val_preds, weights)
    print_relmad_table("Local test RelMAD", y_test, test_preds, weights)

    print("\nOfficial test-set prediction ranges:")
    write_ensemble_submission(official_test_df, selected_features, models, weights)

    print(f"\n{'=' * 72}")
    print("Done")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
