"""
Run inference on the held-out classification test set with the three final
models (NN_final_model_NN, NN_final_model_MI, XGB_final) and write one
submission pair per model to Electron_Project/Submission:

    Classification_RasmusReimer_<ModelName>.csv             (index, p_isElectron)
    Classification_RasmusReimer_<ModelName>_VariableList.csv (one feature per line)
"""

import os
# PyTorch and XGBoost both ship their own libomp on macOS; loading both into
# the same process segfaults inside OpenMP. Allow the duplicate and serialise
# OMP before either library is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


# --- Paths --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../Electron_Project
DATA_DIR     = PROJECT_ROOT / "Data"
CLASS_DIR    = PROJECT_ROOT / "Classification"
NN_DIR       = CLASS_DIR / "NN_Classifier" / "saved_models"
XGB_DIR      = CLASS_DIR / "XGB_Classifier" / "saved_models"
FEATURES_DIR = CLASS_DIR / "Input_lists"
SUBMIT_DIR   = PROJECT_ROOT / "Submission"

TRAIN_H5 = DATA_DIR / "AppML_InitialProject_train.h5"
TEST_H5  = DATA_DIR / "AppML_InitialProject_test_classification.h5"

TARGET_COL = "p_Truth_isElectron"
SUBMITTER  = "RasmusReimer"

# Make Modules/ importable so we can reuse the trained-architecture class.
sys.path.append(str(PROJECT_ROOT))
from Modules.models import ThreeLayerNN  # noqa: E402


# --- Solutions ----------------------------------------------------------------
# Each entry maps a submission name to (kind, model file, feature list file).
# NN feature lists come from Input_lists/. XGB takes its features straight from
# the booster (`features_path = None`) because the saved model is the source of
# truth — the text file in Input_lists/ does not match it.
# The NN architecture below matches NN_final_params.txt (256 / 32 / 32).

NN_HIDDEN = (256, 32, 32)

SOLUTIONS = [
    ("NN_FI",  "nn",  NN_DIR / "NN_final_model_NN.pth",
                   FEATURES_DIR / "NN_feature_importance_input_features.txt"),
    ("NN_MI",  "nn",  NN_DIR / "NN_final_model_MI.pth",
                   FEATURES_DIR / "NN_Mutual_Information_input_features.txt"),
    ("XGB", "xgb", XGB_DIR / "XGB_final.json", None),
]


# --- Helpers ------------------------------------------------------------------

def read_feature_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def scale_with_train(test_df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Fit StandardScaler on the training set's feature subset, transform test."""
    train_df = pd.read_hdf(TRAIN_H5)[features]
    scaler = StandardScaler().fit(train_df.values)
    return scaler.transform(test_df[features].values)


def predict_nn(weights_path: Path, X: np.ndarray) -> np.ndarray:
    model = ThreeLayerNN(X.shape[1], *NN_HIDDEN)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        probs = model(torch.from_numpy(X.astype(np.float32))).squeeze(-1).numpy()
    return probs


def predict_xgb(model_path: Path, test_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    # Use Booster directly: XGBClassifier.load_model can segfault on macOS when
    # libomp has already been pulled in by PyTorch.
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    features = list(booster.feature_names)
    dmat = xgb.DMatrix(test_df[features].values, feature_names=features)
    return booster.predict(dmat), features


def write_submission(name: str, index: pd.Index, probs: np.ndarray, features: list[str]) -> None:
    """Format required by SubmissionChecker: no headers in either file."""
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"Classification_{SUBMITTER}_{name}"

    pd.DataFrame({"index": index, "p_isElectron": probs}) \
      .to_csv(SUBMIT_DIR / f"{base}.csv", index=False, header=False)

    pd.Series(sorted(features)) \
      .to_csv(SUBMIT_DIR / f"{base}_VariableList.csv", index=False, header=False)

    print(f"  wrote {base}.csv  and  {base}_VariableList.csv")


# --- Main ---------------------------------------------------------------------

def main() -> None:
    test_df = pd.read_hdf(TEST_H5)
    if TARGET_COL in test_df.columns:                 # held-out set should not have it
        test_df = test_df.drop(columns=[TARGET_COL])
    print(f"Loaded test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    for name, kind, model_path, features_path in SOLUTIONS:
        src = features_path.name if features_path else "<from booster>"
        print(f"\n[{name}] model={model_path.name}  features={src}")

        if kind == "nn":
            features = read_feature_list(features_path)
            X = scale_with_train(test_df, features)
            probs = predict_nn(model_path, X)
        elif kind == "xgb":
            probs, features = predict_xgb(model_path, test_df)
        else:
            raise ValueError(f"unknown model kind: {kind}")

        write_submission(name, test_df.index, probs, features)


if __name__ == "__main__":
    main()
