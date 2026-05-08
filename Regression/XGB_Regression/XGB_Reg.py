"""
XGBoost regressor for electron-energy estimation (Initial Project, Task 2).

Target:      p_Truth_Energy  (electron energy, GeV)
Filter:      train and evaluate on true electrons only (p_Truth_isElectron == 1)
Constraint:  maximum 20 input features
Grading:     RelMAD = mean(|E_pred - E_true| / E_true)   (≡ MAPE for E_true > 0)

Strategy
--------
1.  First pass: fit on the full feature set so we can rank importances.
2.  Persist the top-20 features to Input_lists/XGB_REG_INPUT.txt and re-train
    on that reduced set (the rubric caps inputs at 20).
3.  Train against `log(E)` with the standard squared-error objective.
    For small deviations  log(p) − log(y) ≈ (p − y)/y , so RMSE on log-space
    is approximately the relative error on original space — this aligns the
    optimisation target with the RelMAD grader more cleanly than fitting
    raw GeV with squared error would.
4.  `eval_metric='mape'` makes early-stopping track the grader directly.

Outputs
-------
*  PLOT_DIR / "{model_tag}_*.png"  — diagnostic plots, one set per model.
*  Input_lists / XGB_REG_INPUT.txt — top-20 feature list for downstream reuse.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

# Anchor paths to this file (not CWD) so the script works no matter where
# Python is invoked from.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.append(str(PROJECT_ROOT))

from Modules.Utils import XGB_REG_DATALOADER  # filters to true electrons


# ---------------------------------------------------------------- configuration
DATA_PATH = PROJECT_ROOT / 'Data' / 'AppML_InitialProject_train.h5'
PLOT_DIR = HERE / 'XGB_Reg_plots'
FEATURE_LIST_OUT = PROJECT_ROOT / 'Input_lists' / 'XGB_REG_INPUT.txt'

# NB: the existing codebase capitalises "Truth" (see Modules/Utils.py); the
# project handout writes it lowercase. Adjust this constant if the on-disk
# column name differs.
TARGET_COL = 'p_Truth_Energy'
TOP_N_FEATURES = 20
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_TRIALS = 40                       # Optuna budget for the top-20 tuning pass

# Bits of the param dict that never change across passes (objective, metric,
# early stopping, etc.) — kept separate so the Optuna study only sweeps over
# the things worth sweeping.
FIXED_PARAMS = {
    'n_estimators': 10_000,         # capped in practice by early stopping
    'objective': 'reg:squarederror',
    'eval_metric': 'mape',          # mirrors the RelMAD grading metric
    'early_stopping_rounds': 50,    # 10 (the previous value) stops on noise
    'tree_method': 'hist',
    'random_state': RANDOM_STATE,
}

# Default tunable hyperparameters — used for the first two (un-tuned) passes.
DEFAULT_TUNABLE = {
    'max_depth': 4,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'min_child_weight': 1,
}

PARAMS = {**FIXED_PARAMS, **DEFAULT_TUNABLE}


# ------------------------------------------------------------------- utilities
def relmad(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Project's grading metric: mean(|E_pred − E_true| / E_true)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true) / y_true))


def train(X_train, X_val, y_train, y_val, params=PARAMS) -> XGBRegressor:
    """Fit on log-energy. Inverse transform happens in `predict_geV`."""
    model = XGBRegressor(**params)
    model.fit(
        X_train, np.log(y_train),
        eval_set=[(X_train, np.log(y_train)), (X_val, np.log(y_val))],
        verbose=False,
    )
    return model


def tune_hyperparameters(X_train, X_val, y_train, y_val,
                         n_trials: int = N_TRIALS) -> dict:
    """
    Run an Optuna study minimising RelMAD on the validation set.

    Returns a full param dict (FIXED_PARAMS merged with the best tunable values),
    ready to feed straight into `train()`.

    Note on the val split: we reuse the same val set the earlier passes used
    for early stopping. The tuned model's reported val metrics are therefore
    a slightly optimistic estimate — the true held-out score is whatever the
    grader's test set produces. For our purposes (picking the best config
    among comparable ones) the bias is consistent across trials and harmless.
    """
    log_y_train = np.log(y_train)
    log_y_val = np.log(y_val)

    def objective(trial: optuna.Trial) -> float:
        tunable = {
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma':            trial.suggest_float('gamma', 1e-4, 1.0, log=True),
        }
        model = XGBRegressor(**FIXED_PARAMS, **tunable)
        model.fit(
            X_train, log_y_train,
            eval_set=[(X_val, log_y_val)],
            verbose=False,
        )
        y_pred = np.exp(model.predict(X_val))
        return relmad(y_val, y_pred)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nOptuna best trial #{study.best_trial.number}: "
          f"RelMAD = {study.best_value:.5f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k:<18} {v}")

    return {**FIXED_PARAMS, **study.best_params}


def predict_geV(model: XGBRegressor, X) -> np.ndarray:
    """Undo the log transform applied during training."""
    return np.exp(model.predict(X))


def evaluate(tag: str, model: XGBRegressor, X_train, X_val, y_train, y_val):
    y_pred_val = predict_geV(model, X_val)
    metrics = {
        'RelMAD (val)':           relmad(y_val, y_pred_val),
        'RelMAD (train)':         relmad(y_train, predict_geV(model, X_train)),
        'MAPE  (val, sklearn)':   mean_absolute_percentage_error(y_val, y_pred_val),
        'RMSE  (val, GeV)':       float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
        'R²    (val)':            float(r2_score(y_val, y_pred_val)),
        'best_iteration':         int(model.best_iteration),
    }
    print(f"\n=== {tag} ===")
    for k, v in metrics.items():
        print(f"  {k:<24} {v}")
    return metrics, y_pred_val


# ----------------------------------------------------------------- diagnostics
def _ensure_plot_dir() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def save_training_curves(tag: str, model: XGBRegressor) -> None:
    res = model.evals_result()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res['validation_0']['mape'], label='Train MAPE')
    ax.plot(res['validation_1']['mape'], label='Val MAPE')
    best_relmad = res['validation_1']['mape'][model.best_iteration]
    ax.axvline(model.best_iteration, color='gray', ls='--',
               label=f'best iter = {model.best_iteration}  (RelMAD = {best_relmad:.4f})')
    ax.set_xlabel('Boosting iteration')
    ax.set_ylabel('MAPE  (≡ RelMAD)')
    ax.set_title(f'{tag} — training curves')
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f'{tag}_training_curves.png', dpi=120)
    plt.close(fig)


def save_pred_vs_true(tag: str, y_true, y_pred) -> None:
    # log–log axes because electron energies span a wide dynamic range and a
    # linear scatter is dominated by the high-energy tail.
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, alpha=0.2, s=5)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], 'r--', label='y = x')
    ax.set_xscale('log'); ax.set_yscale('log')
    rm = relmad(y_true, y_pred)
    ax.set_xlabel('True energy (GeV)')
    ax.set_ylabel('Predicted energy (GeV)')
    ax.set_title(f'{tag} — predicted vs true  (RelMAD = {rm:.4f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f'{tag}_pred_vs_true.png', dpi=120)
    plt.close(fig)


def save_rel_error(tag: str, y_true, y_pred) -> None:
    rel = (np.asarray(y_pred) - np.asarray(y_true)) / np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rel, bins=120, range=(-0.5, 0.5))
    ax.axvline(0, color='red', ls='--')
    rm = relmad(y_true, y_pred)
    ax.axvline(rel.mean(), color='orange', ls='--',
               label=f'mean = {rel.mean():+.4f}')
    ax.axvline(np.median(rel), color='green', ls='--',
               label=f'median = {np.median(rel):+.4f}')
    ax.set_xlabel('(E_pred − E_true) / E_true')
    ax.set_ylabel('Count')
    ax.set_title(f'{tag} — relative-error distribution  (RelMAD = {rm:.4f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f'{tag}_rel_error.png', dpi=120)
    plt.close(fig)


def save_feature_importance(
    tag: str, model: XGBRegressor, feature_names, top_n: int = 20,
) -> pd.Series:
    importances = (
        pd.Series(model.feature_importances_, index=feature_names)
          .nlargest(top_n)
    )
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(importances) + 1))
    importances[::-1].plot(kind='barh', ax=ax)  # reverse for visual top-down
    ax.set_xlabel('Gain importance')
    ax.set_title(f'{tag} — top {top_n} features')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f'{tag}_feature_importance.png', dpi=120)
    plt.close(fig)
    return importances


def save_feature_list(features: list[str], path: Path) -> None:
    """Persist the top-N feature list so other notebooks can reuse it verbatim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(features) + '\n')
    print(f"Wrote feature list ({len(features)} features) → {path}")


def save_all_diagnostics(
    tag: str, model: XGBRegressor, y_val, y_pred_val,
    feature_names, top_n: int,
) -> pd.Series:
    _ensure_plot_dir()
    save_training_curves(tag, model)
    save_pred_vs_true(tag, y_val, y_pred_val)
    save_rel_error(tag, y_val, y_pred_val)
    return save_feature_importance(tag, model, feature_names, top_n=top_n)


# -------------------------------------------------------------------- pipeline
def main() -> None:
    X_train, X_val, y_train, y_val = XGB_REG_DATALOADER(
        str(DATA_PATH), TARGET_COL, test_size=TEST_SIZE,
    )
    print(f"Train: {X_train.shape}   Val: {X_val.shape}")
    print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}] GeV")

    # ----- Pass 1: full feature set, used purely to rank importances ---------
    tag_full = 'full_features'
    model_full = train(X_train, X_val, y_train, y_val)
    _, y_pred_full = evaluate(tag_full, model_full, X_train, X_val, y_train, y_val)
    importances = save_all_diagnostics(
        tag_full, model_full, y_val, y_pred_full,
        feature_names=X_train.columns, top_n=TOP_N_FEATURES,
    )

    top_features = importances.index.tolist()
    save_feature_list(top_features, FEATURE_LIST_OUT)

    # ----- Pass 2: top-N features only — this is the model the rubric scores --
    tag_top = f'top{TOP_N_FEATURES}_features'
    X_train_top = X_train[top_features]
    X_val_top = X_val[top_features]

    model_top = train(X_train_top, X_val_top, y_train, y_val)
    _, y_pred_top = evaluate(tag_top, model_top, X_train_top, X_val_top, y_train, y_val)
    save_all_diagnostics(
        tag_top, model_top, y_val, y_pred_top,
        feature_names=top_features, top_n=TOP_N_FEATURES,
    )

    # ----- Pass 3: Optuna-tuned top-N — the model used for grading ----------
    tag_tuned = f'top{TOP_N_FEATURES}_tuned'
    print(f"\nTuning hyperparameters with {N_TRIALS} Optuna trials...")
    tuned_params = tune_hyperparameters(X_train_top, X_val_top, y_train, y_val)

    model_tuned = train(X_train_top, X_val_top, y_train, y_val, params=tuned_params)
    _, y_pred_tuned = evaluate(
        tag_tuned, model_tuned, X_train_top, X_val_top, y_train, y_val,
    )
    save_all_diagnostics(
        tag_tuned, model_tuned, y_val, y_pred_tuned,
        feature_names=top_features, top_n=TOP_N_FEATURES,
    )

    print(f"\nAll plots saved to {PLOT_DIR}")


if __name__ == '__main__':
    main()
