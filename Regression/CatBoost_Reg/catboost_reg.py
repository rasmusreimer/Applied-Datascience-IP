"""
CatBoost regressor for electron energy with physics-motivated 20-feature set.

Trains on log(E_true), reports test-set RelMAD, and writes a submission CSV
in the same format as the XGB pipeline.

Run from anywhere — paths are anchored to PROJECT_ROOT.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split


# ============================================================ CONFIG
PROJECT_ROOT = Path("/Users/prometheus/Documents/Python/Electron_Project")
DATA_DIR     = PROJECT_ROOT / "Data"
SAVED_DIR    = PROJECT_ROOT / "Regression" / "CatBoost_Reg" / "saved_models"
SUBMIT_DIR   = PROJECT_ROOT / "Submission"

TRAIN_H5 = DATA_DIR / "AppML_InitialProject_train.h5"
TEST_H5  = DATA_DIR / "AppML_InitialProject_test_regression.h5"

TARGET_COL   = "p_Truth_Energy"
SUBMITTER    = "RasmusReimer"
MODEL_NAME   = "CatBoost_Reg"

TEST_SIZE    = 0.20
VAL_SIZE     = 0.10
RANDOM_STATE = 42


TOP_FEATURES = [
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

    # Verify all required features exist
    missing = [f for f in TOP_FEATURES if f not in df.columns]
    if missing:
        raise KeyError(f"Features missing from training data: {missing}")

    y = df[TARGET_COL]
    X = df[TOP_FEATURES]

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
def train_catboost(X_train, X_val, y_train, y_val):
    """Train on log(E). Returns the fitted model."""
    print("\nTraining CatBoost regressor on log(E)...")

    log_y_train = np.log(np.asarray(y_train, dtype=float))
    log_y_val   = np.log(np.asarray(y_val,   dtype=float))

    # CatBoost Pool objects bundle features + categoricals together
    train_pool = Pool(
        data=X_train, label=log_y_train, cat_features=CATEGORICAL_FEATURES,
    )
    val_pool = Pool(
        data=X_val, label=log_y_val, cat_features=CATEGORICAL_FEATURES,
    )

    model = CatBoostRegressor(
        iterations=10_000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        loss_function="RMSE",      # MSE on log(E)
        eval_metric="MAPE",        # tracks RelMAD-equivalent
        random_seed=RANDOM_STATE,
        od_type="Iter",            # early stopping by iteration count
        od_wait=200,               # stop if no improvement in 200 iters
        verbose=200,               # print progress every 200 iters
        bagging_temperature=1.0,
        random_strength=1.0,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    print(f"Best iteration: {model.get_best_iteration()}")
    return model


def predict_geV(model, X):
    """Inverse log transform: log(E) → E in GeV."""
    return np.exp(model.predict(X))


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
    """Show feature importance ranking."""
    imp = pd.Series(model.get_feature_importance(), index=TOP_FEATURES)
    imp = imp.sort_values(ascending=False)
    print("\n=== Feature Importance ===")
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
        "best_iteration": int(model.get_best_iteration()),
        "metrics": {
            "relmad_train": metrics[0],
            "relmad_val":   metrics[1],
            "relmad_test":  metrics[2],
        },
        "model_params": model.get_all_params(),
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

    X_submit = test_df[TOP_FEATURES]
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
    print(f"{'='*60}\nCatBoost Regression Pipeline\n{'='*60}")

    # 1. Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_train_data()

    # 2. Train
    model = train_catboost(X_train, X_val, y_train, y_val)

    # 3. Evaluate
    metrics = evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test)

    # 4. Feature importance (diagnostic)
    feature_importance(model)

    # 5. Save model + metadata
    save_artifacts(model, metrics)

    # 6. Inference on held-out test set → submission CSV
    write_submission(model)

    print(f"\n{'='*60}\nDone\n{'='*60}")


if __name__ == "__main__":
    main()