"""
LightGBM regressor for electron energy with the XGB-derived 20-feature set.

Trains on log(E_true), reports test-set RelMAD, and writes a submission CSV
in the same format as the XGB and CatBoost pipelines.

Run from anywhere — paths are anchored to PROJECT_ROOT.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================ CONFIG
PROJECT_ROOT = Path("/Users/prometheus/Documents/Python/Electron_Project")
DATA_DIR     = PROJECT_ROOT / "Data"
SAVED_DIR    = PROJECT_ROOT / "Regression" / "LGB_Reg" / "saved_models"
SUBMIT_DIR   = PROJECT_ROOT / "Submission"

TRAIN_H5 = DATA_DIR / "AppML_InitialProject_train.h5"
TEST_H5  = DATA_DIR / "AppML_InitialProject_test_regression.h5"

TARGET_COL   = "p_Truth_Energy"
SUBMITTER    = "RasmusReimer"
MODEL_NAME   = "LGB_Reg"

TEST_SIZE    = 0.20
VAL_SIZE     = 0.10
RANDOM_STATE = 42

# XGB-derived 20-feature set (same as your tuned XGB pipeline)
TOP_FEATURES = TOP_FEATURES = [
    "p_ptcone40",
    "pX_ecore",
    "p_etcone20",
    "pX_E3x5_Lr2",
    "pX_deltaPhi2",
    "pX_MultiLepton",
    "p_pt_track",
    "pX_E3x5_Lr0",
    "pX_etcone20",
    "pX_E3x5_Lr1",
    "pX_maxEcell_energy",
    "pX_nCells_Lr1_MedG",
    "pX_E_Lr2_MedG",
    "pX_E_Lr2_HiG",
    "pX_deltaPhiFromLastMeasurement",
    "pX_wtots1",
    "p_numberOfPixelHits",
    "pX_topoetcone40ptCorrection",
    "pX_E_Lr0_HiG",
    "pX_ambiguityType",
]


# LightGBM handles categoricals natively when told which columns they are.
CATEGORICAL_FEATURES = ["pX_ambiguityType"]

assert len(TOP_FEATURES) == 20, f"Expected 20 features, got {len(TOP_FEATURES)}"


# ============================================================ utilities
def relmad(y_true, y_pred):
    """Project's grading metric: mean(|E_pred − E_true| / E_true)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true) / y_true))


def load_train_data():
    """Load training HDF5, filter to true electrons, three-way split."""
    print(f"Loading {TRAIN_H5}")
    df = pd.read_hdf(TRAIN_H5)
    df = df[df["p_Truth_isElectron"] == 1].copy()
    print(f"After electron filter: {len(df)} rows")

    missing = [f for f in TOP_FEATURES if f not in df.columns]
    if missing:
        raise KeyError(f"Features missing from training data: {missing}")

    y = df[TARGET_COL]
    X = df[TOP_FEATURES]

    # LightGBM expects categorical columns as int or pandas Categorical.
    # Cast pX_ambiguityType to int (assuming it's integer-valued already).
    for cat in CATEGORICAL_FEATURES:
        X.loc[:, cat] = X[cat].astype("int32")

    # Three-way split: 70/10/20
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=RANDOM_STATE,
    )

    print(f"Split: train {len(X_train)}  val {len(X_val)}  test {len(X_test)}")
    print(f"Train E range: [{y_train.min():.1f}, {y_train.max():.1f}] GeV")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================ training
def train_lgb(X_train, X_val, y_train, y_val):
    """Train on log(E). Returns the fitted booster."""
    print("\nTraining LightGBM regressor on log(E)...")

    log_y_train = np.log(np.asarray(y_train, dtype=float))
    log_y_val   = np.log(np.asarray(y_val,   dtype=float))

    train_ds = lgb.Dataset(
        X_train, label=log_y_train,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    val_ds = lgb.Dataset(
        X_val, label=log_y_val,
        categorical_feature=CATEGORICAL_FEATURES,
        reference=train_ds,
        free_raw_data=False,
    )

    params = {
        "objective": "regression",          # squared error on log(E)
        "metric": "mape",                   # tracks RelMAD-equivalent on val
        "learning_rate": 0.03,
        "num_leaves": 63,                   # roughly equivalent to depth=6
        "max_depth": -1,                    # let num_leaves control complexity
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,           # ~ colsample_bytree
        "bagging_fraction": 0.85,           # ~ subsample
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": RANDOM_STATE,
    }

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=10_000,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=True),
            lgb.log_evaluation(period=200),
        ],
    )

    print(f"Best iteration: {model.best_iteration}")
    return model


def predict_geV(model, X):
    """Inverse log transform: log(E) → E in GeV. Always uses best iteration."""
    return np.exp(model.predict(X, num_iteration=model.best_iteration))


# ============================================================ evaluation
def evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Print RelMAD on all three splits."""
    print("\n=== RelMAD ===")
    rm_train = relmad(y_train, predict_geV(model, X_train))
    rm_val   = relmad(y_val,   predict_geV(model, X_val))
    rm_test  = relmad(y_test,  predict_geV(model, X_test))
    print(f"  train: {rm_train:.5f}")
    print(f"  val:   {rm_val:.5f}")
    print(f"  test:  {rm_test:.5f}")
    return rm_train, rm_val, rm_test


def feature_importance(model):
    """Show feature importance ranking (gain-based)."""
    imp = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=TOP_FEATURES,
    )
    imp = imp / imp.sum() * 100   # normalize to percent
    imp = imp.sort_values(ascending=False)
    print("\n=== Feature Importance (% of total gain) ===")
    for f, v in imp.items():
        print(f"  {v:6.2f}  {f}")
    return imp


# ============================================================ save / submit
def save_artifacts(model, metrics):
    """Save model + metadata for reuse."""
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    model_path = SAVED_DIR / f"{MODEL_NAME}.joblib"
    joblib.dump(model, model_path)

    meta_path = SAVED_DIR / f"{MODEL_NAME}_params.json"
    payload = {
        "features": list(TOP_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "best_iteration": int(model.best_iteration),
        "metrics": {
            "relmad_train": metrics[0],
            "relmad_val":   metrics[1],
            "relmad_test":  metrics[2],
        },
        "params": dict(model.params),
    }
    meta_path.write_text(json.dumps(payload, indent=2, default=str))

    print(f"\nSaved model    → {model_path}")
    print(f"Saved metadata → {meta_path}")


def write_submission(model):
    """Predict on the held-out test set and write submission CSVs."""
    print(f"\nLoading {TEST_H5}")
    test_df = pd.read_hdf(TEST_H5)
    if TARGET_COL in test_df.columns:
        test_df = test_df.drop(columns=[TARGET_COL])
    print(f"Test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    missing = [f for f in TOP_FEATURES if f not in test_df.columns]
    if missing:
        raise KeyError(f"Features missing from test set: {missing}")

    X_submit = test_df[TOP_FEATURES].copy()
    for cat in CATEGORICAL_FEATURES:
        X_submit.loc[:, cat] = X_submit[cat].astype("int32")

    energy = predict_geV(model, X_submit)
    print(f"Predicted energy range: [{energy.min():.2f}, {energy.max():.2f}] GeV")

    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"Regression_{SUBMITTER}_{MODEL_NAME}"

    pd.DataFrame({"index": test_df.index, "p_Truth_Energy": energy}) \
      .to_csv(SUBMIT_DIR / f"{base}.csv", index=False, header=False)

    pd.Series(sorted(TOP_FEATURES)) \
      .to_csv(SUBMIT_DIR / f"{base}_VariableList.csv", index=False, header=False)

    print(f"Wrote {SUBMIT_DIR / f'{base}.csv'}")
    print(f"Wrote {SUBMIT_DIR / f'{base}_VariableList.csv'}")


# ============================================================ main
def main():
    print(f"{'='*60}\nLightGBM Regression Pipeline\n{'='*60}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_train_data()

    model = train_lgb(X_train, X_val, y_train, y_val)

    metrics = evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test)

    feature_importance(model)

    save_artifacts(model, metrics)

    write_submission(model)

    print(f"\n{'='*60}\nDone\n{'='*60}")


if __name__ == "__main__":
    main()