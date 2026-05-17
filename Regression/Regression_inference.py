"""
Run inference on the held-out regression test set with the two regression models
(NN_Reg_SHAP_artifact, XGB_reg) and write one submission pair per model to
Electron_Project/Submission:

    Regression_RasmusReimer_<ModelName>.csv             (index, p_Truth_Energy)
    Regression_RasmusReimer_<ModelName>_VariableList.csv (one feature per line)
"""

import os
# PyTorch and XGBoost both ship their own libomp on macOS; loading both into
# the same process segfaults inside OpenMP. Allow the duplicate and serialise
# OMP before either library is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import sys
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


# --- Paths --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../Electron_Project
DATA_DIR     = PROJECT_ROOT / "Data"
REG_DIR      = PROJECT_ROOT / "Regression"
NN_DIR       = REG_DIR / "NN_Reg" / "saved_models"
XGB_DIR      = REG_DIR / "XGB_Regression" / "saved_models"
SUBMIT_DIR   = PROJECT_ROOT / "Submission"

TEST_H5 = DATA_DIR / "AppML_InitialProject_test_regression.h5"

TARGET_COL = "p_Truth_Energy"
SUBMITTER  = "RasmusReimer"

# Make Modules/ importable for project-local imports if needed.
sys.path.append(str(PROJECT_ROOT))


# --- Solutions ----------------------------------------------------------------
# Each entry maps a submission name to (kind, model file, params file).
# NN .pth is a dict bundling state_dict / scaler / features; the .txt holds
# the architecture hyperparameters. XGB ships the model in a joblib and the
# feature list + hyperparameters in a JSON sidecar.

SOLUTIONS = [
    ("NN_Reg",  "nn",  NN_DIR  / "NN_Reg_SHAP_artifact.pth", NN_DIR  / "NN_Reg_SHAP_params.txt"),
    ("XGB_Reg", "xgb", XGB_DIR / "top20_tuned.joblib",       XGB_DIR / "top20_tuned_params.json"),
]


# --- Helpers ------------------------------------------------------------------

# Sanity cap on log(E) before exp(). The training-set energies are in MeV
# Cap just above the training max so extrapolated rows are
# pinned to a physically plausible value instead of inflating the score.
LOG_E_CLIP = 14.0   # exp(14) ≈ 1.2e6 MeV ≈ 1200 GeV — just above training max


def _exp_with_clip(log_pred: np.ndarray, tag: str) -> np.ndarray:
    extreme = int(np.sum(log_pred > LOG_E_CLIP))
    if extreme:
        print(f"  WARNING [{tag}]: {extreme} row(s) have log-pred > {LOG_E_CLIP}; "
              f"clipping before exp() to keep CSV finite.")
    return np.exp(np.clip(log_pred, -LOG_E_CLIP, LOG_E_CLIP))


class SavedThreeLayerRegressor(nn.Module):
    """Matches the fc1..fc4 keys saved by the NN_Reg_SHAP training notebook."""
    def __init__(self, input_size, first_layer, second_layer, third_layer, dropout):
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


def _parse_params_txt(path: Path) -> dict:
    """Parse a 'key: value' file, coercing values to int or float when possible."""
    out: dict = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        try:
            out[key.strip()] = int(value)
        except ValueError:
            try:
                out[key.strip()] = float(value)
            except ValueError:
                out[key.strip()] = value
    return out


def predict_nn(model_path: Path, params_path: Path, test_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Architecture from the .txt; state_dict, scaler, and feature list from the .pth dict."""
    params = _parse_params_txt(params_path)
    # weights_only=False because the artifact bundles a sklearn scaler alongside tensors.
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    features = list(artifact["features"])
    scaler = artifact["scaler"]

    # Pass the DataFrame (not .values) so the scaler matches by feature name.
    X = scaler.transform(test_df[features]).astype(np.float32)

    model = SavedThreeLayerRegressor(
        input_size=len(features),
        first_layer=int(params["first_layer"]),
        second_layer=int(params["second_layer"]),
        third_layer=int(params["third_layer"]),
        dropout=float(params.get("dropout", 0.0)),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    with torch.no_grad():
        log_pred = model(torch.from_numpy(X)).cpu().numpy()

    return _exp_with_clip(log_pred, "NN"), features


def predict_xgb(model_path: Path, params_path: Path, test_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """XGBRegressor was trained on log(E); feature list comes from the params JSON."""
    payload = json.loads(params_path.read_text())
    features = list(payload["features"])
    model = joblib.load(model_path)
    log_pred = model.predict(test_df[features])
    return _exp_with_clip(log_pred, "XGB"), features


def write_submission(name: str, index: pd.Index, energy: np.ndarray, features: list[str]) -> None:
    """Format required by SubmissionChecker: no headers in either file."""
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"Regression_{SUBMITTER}_{name}"

    pd.DataFrame({"index": index, "p_Truth_Energy": energy}) \
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

    for name, kind, model_path, params_path in SOLUTIONS:
        print(f"\n[{name}] model={model_path.name}  params={params_path.name}")

        if kind == "nn":
            energy, features = predict_nn(model_path, params_path, test_df)
        elif kind == "xgb":
            energy, features = predict_xgb(model_path, params_path, test_df)
        else:
            raise ValueError(f"unknown model kind: {kind}")

        print(f"  predicted energy range: [{energy.min():.2f}, {energy.max():.2f}] GeV")
        write_submission(name, test_df.index, energy, features)


if __name__ == "__main__":
    main()
