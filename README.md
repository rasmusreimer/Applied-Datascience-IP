# Electron Classification & Energy Regression

Two ATLAS-style particle-physics ML tasks done two ways each, with feature-selection ablations and Optuna-tuned hyperparameters.

> Initial Project for the *Applied Machine Learning* course at the Niels Bohr Institute, University of Copenhagen.

---

## The data

A 180k-event electron PID dataset with ~120 calorimeter and tracking features and two labels:

- `p_Truth_isElectron` — binary, true electron vs. non-electron (~21% positive class)
- `p_Truth_Energy` — continuous, electron energy in GeV

Files live in `Data/` (~125 MB on disk):

- `AppML_InitialProject_train.h5` — 180k rows, all labels present
- `AppML_InitialProject_test_classification.h5` — blind classification test set
blind regression set - to be implemented
clustering - to be implemented 

The regression task is restricted to true electrons only (`p_Truth_isElectron == 1`). True-label columns are dropped from `X` everywhere to prevent leakage.

---

## Task 1 — Classification (electron vs. non-electron)

Four models in a 2 × 2 grid: **{NN, XGBoost} × {Mutual-Information features, XGB-Feature-Importance features}**. Same 15-feature budget for every model so the comparison isolates the feature-selection choice.

**Mutual-Information selector** (`Modules/Utils.py:fast_preprocess_data`)
Drop pairs with |corr| > 0.95 (keep the higher-MI side), then take the top 15 by `mutual_info_classif`. Standard-scaled for the NN.

**XGB-Feature-Importance selector** (`full_feature_data_preprocess` + a baseline XGB)
Train an unconstrained XGB on the full feature set, persist its top 15 by feature importance gain, reuse that subset for both the FI-track XGB.

**NN-SHAP selector (full-features NN + SHAP)**
Symmetric construction for the NN: train a NN on the full feature set, take the top 15 by mean absolute SHAP value. There's no a priori reason a NN should rank features the same way a boosted tree does, and a full-features NN run is much more expensive than its XGB counterpart — so it's worth checking whether the cheaper shortcut (reuse XGB's gain-ranked list for the NN) leaves performance on the table. The two lists overlap heavily but not perfectly. Downstream, the difference doesn't show up: the NN reaches AUC ≈ 0.99 on either list, with accuracy and F1 indistinguishable.
Architectures:

- **NN** — `ThreeLayerNN` (`Modules/models.py`): 15 → 256 → 32 → 32 → 1, sigmoid head, BCE loss, AdamW, ReduceLROnPlateau, early stopping on val, BCE on a held-out val fold for Optuna's objective.
- **XGB** — `n_estimators=10 000` capped by `early_stopping_rounds=10`, `max_depth=4`, `lr=0.1`, `subsample=0.8`, `eval_metric='logloss'`. Optuna sweeps depth, learning rate, subsample, and L2 over 500 trials maximising F1.

### Results (held-out test split, 36 000 events)

| Model | Features | Accuracy | Precision | Recall | F1 (electron) |
|---|---|---|---|---|---|
| NN — MI (untuned) | 15 (MI) | 0.938 | 0.951 | 0.739 | 0.832 |
| NN — MI (Optuna-tuned) | 15 (MI) | 0.938 | 0.957 | 0.738 | 0.833 |
| NN — FI (Optuna-tuned) | 15 (FI) | 0.962 | 0.928 | 0.886 | 0.907 |
| XGB — MI (untuned) | 15 (MI) | 0.943 | 0.94 | 0.77 | 0.85 |
| XGB — MI (Optuna-tuned) | 15 (MI) | 0.944 | 0.95 | 0.77 | 0.85 |
| **XGB — FI (full → top-15)** | **15 (FI)** | **0.974** | **0.95** | **0.92** | **0.94** |

**Headline.** The feature subset selected by the boosted tree itself is more informative than the MI-ranked subset for *both* model families. The NN gains ~7 pp F1 just by switching feature lists at fixed architecture; the XGB gains ~9 pp. MI-track recall caps near 0.74 — those features carry the easy positives and miss the hard ones.
Interestingly the MI based NN approach results in 50% less falsepositives, but around twice as many falsenegatives.

A SHAP summary on the tuned NN-FI model (cell 10 of `NN_Class_feature_importance.ipynb`) shows the top contributors are `p_TRTPID`, `pX_MultiLepton`, and `p_Eratio` — consistent with the physics: TRT particle ID and shower-shape ratios dominate electron PID at ATLAS.

---

## Task 2 — Regression (electron energy, GeV)

Two models — XGBoost script + NN notebook — sharing the same setup so they're directly comparable:

- **20-feature cap** (rubric constraint), selected by feature importance from a full-features XGB pass; persisted to `Regression/Input_lists/XGB_REG_INPUT.txt` and reused by both pipelines.
- **Log-target trick.** Train on `log(E)` with squared error; undo with `exp` at inference. For small deviations `log(p) − log(y) ≈ (p − y) / y`, so the optimisation signal is approximately the grading metric — *RelMAD* — rather than absolute GeV error, which would bias the model toward the high-energy tail.
- **`eval_metric='mape'`** on the XGB side: early stopping tracks RelMAD directly.
- **Filter to true electrons** at preprocessing (`XGB_REG_DATALOADER` in `Modules/Utils.py`).

### XGB pipeline (`Regression/XGB_Regression/XGB_Reg.py`)

A single self-contained script with three passes, each emitting a tagged set of diagnostic plots into `XGB_Reg_plots/`:

1. `full_features` — fit on the full feature set, rank importances, write the top-20 list.
2. `top20_features` — refit on those 20 features (untuned). The honest baseline for the rubric.
3. `top20_tuned` — Optuna over depth, lr, subsample, colsample, reg_alpha/lambda, min_child_weight, and gamma; minimises RelMAD on val.


### NN pipeline (`Regression/NN_Reg/NN_Reg.ipynb`)

`ThreeLayerRegressor` (same architecture family as the classifier, no sigmoid, tunable dropout). Optuna sweeps four layer widths, dropout, learning rate, weight decay, and batch size — minimising RelMAD on val. The final tuned model bundles weights + scaler + feature ordering + hyperparameters + log-target flag into a single `.pth` artifact (`NN_Reg_artifact.pth`), so submitting against a blind test set is one path change.

### Results (RelMAD on val)

| Model | Features | RelMAD ↓ |
|---|---|---|
| XGB — full features (unconstrained baseline) | ~75 | 0.2188 |
| XGB — top 20, untuned | 20 | 0.2386 |
| **XGB — top 20, Optuna-tuned** | **20** | **0.2244** |
| NN — top 20, Optuna-tuned (best trial) | 20 | 0.228 |

**Headline.** Tuning the 20-feature XGB recovers most of the ~2 pp gap from the unconstrained baseline (0.2244 vs. 0.2188). The NN gets close (0.228) but doesn't beat the boosted tree on tabular physics features — as expected. Final submitted model: tuned XGB.

The relative-error histograms (`Regression/XGB_Regression/XGB_Reg_plots/top20_tuned_rel_error.png`) show median ≈ 0, mean +0.09 — i.e. the tail is right-skewed: the model under-predicts a small population of high-energy events. Possible follow-up: a separate high-E head, or per-event uncertainty estimates.

---

## Repository layout

```
.
├── Classification/
│   ├── NN_Classifier/
│   │   ├── NN_Class_mutual_information.ipynb     # NN on MI features (+ Optuna)
│   │   ├── NN_Class_feature_importance.ipynb     # NN on XGB-FI features (+ SHAP)
│   │   ├── NN_CLASS_Tuned_Params.txt             # Optuna best params
│   │   └── saved_models/                          # *.pth checkpoints
│   ├── XGB_Classifier/
│   │   ├── XGB_Mutual_information.ipynb          # XGB on MI features (+ Optuna)
│   │   ├── XGB_feature_importance.ipynb          # XGB on full → top-15 by FI
│   │   └── saved_models/                          # XGB_final.json + params
│   └── Input_lists/                               # persisted feature subsets
│
├── Regression/
│   ├── NN_Reg/
│   │   ├── NN_Reg.ipynb                          # NN regressor (+ Optuna + artifact)
│   │   ├── NN_Reg_artifact.pth                   # weights + scaler + features bundle
│   │   └── NN_Reg_plots/                         # training / pred-vs-true / rel-error
│   ├── XGB_Regression/
│   │   ├── XGB_Reg.py                            # 3-pass script (path-agnostic)
│   │   └── XGB_Reg_plots/                        # one plot set per pass
│   └── Input_lists/XGB_REG_INPUT.txt             # top-20 features (shared with NN)
│
├── Modules/
│   ├── Utils.py     # 3 dataloaders: fast_preprocess (NN cls),
│   │               # full_feature_data_preprocess (XGB cls),
│   │               # XGB_REG_DATALOADER (regression, e-only)
│   └── models.py    # TwoLayer / ThreeLayer / FourLayer NN classifiers,
│                    # ThreeLayerRegressor, XGBoostModel wrapper
│
└── Data/            # train + blind classification test (HDF5)
```

---

## FUTURE IMPLEMENTATION

Path invariance
~~Path invariance — currently uses absolute paths in multiple places.~~ - fixed

~~NN features decided via a feature importance test on the full input feature set, instead of inheriting the input features from XGB. They will likely be similar, and the compute demand for the training is vastly greater, but for a fair comparison this should be implemented~~ - implemented
