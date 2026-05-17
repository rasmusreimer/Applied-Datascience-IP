# Electron Classification & Energy Regression

Two ATLAS-style particle-physics ML tasks done several ways each, with feature-selection ablations and Optuna / RandomizedSearchCV-tuned hyperparameters.

> Initial Project for the *Applied Machine Learning* course at the Niels Bohr Institute, University of Copenhagen.


---

## The data

A 180k-event electron PID dataset with ~120 calorimeter and tracking features and two labels:

- `p_Truth_isElectron` — binary, true electron vs. non-electron (~21% positive class)
- `p_Truth_Energy` — continuous, electron energy in GeV

Files live in `Data/` (~125 MB on disk):

- `AppML_InitialProject_train.h5` — 180k rows, all labels present
- `AppML_InitialProject_test_classification.h5` — blind classification test set
- `AppML_InitialProject_test_regression.h5` — blind regression test set

The regression task is restricted to true electrons only (`p_Truth_isElectron == 1`). True-label columns are dropped from `X` everywhere to prevent leakage.

---

## Task 1 — Classification (electron vs. non-electron)

Four models. Three of them form a **2 × 2 ablation grid**: `{NN, XGBoost} × {Mutual-Information features, XGB-Feature-Importance features}`, same 15-feature budget so the comparison isolates the feature-selection choice. A CatBoost classifier with recursive SHAP feature elimination is added on top.

**Mutual-Information selector** (`Modules/Utils.py:fast_preprocess_data`)
Drop pairs with |corr| > 0.95 (keep the higher-MI side), then take the top 15 by `mutual_info_classif`. Standard-scaled for the NN.

**XGB-Feature-Importance selector** (`full_feature_data_preprocess` + a baseline XGB)
Train an unconstrained XGB on the full feature set, persist its top 15 by feature importance gain, reuse that subset for both the FI-track XGB and (separately) the NN.

**NN-SHAP selector (full-features NN + SHAP)**
Symmetric construction for the NN: train a NN on the full feature set, take the top 15 by mean absolute SHAP value. There's no a priori reason a NN should rank features the same way a boosted tree does, and a full-features NN run is much more expensive than its XGB counterpart — so it's worth checking whether the cheaper shortcut (reuse XGB's gain-ranked list for the NN) leaves performance on the table. The two lists overlap heavily but not perfectly. Downstream, the difference doesn't show up: the NN reaches AUC ≈ 0.99 on either list.

**Recursive SHAP (CatBoost)** — used by the CatBoost classifier described below.

Architectures:

- **NN** — `ThreeLayerNN` (`Modules/models.py`): 15 → 256 → 32 → 32 → 1, sigmoid head, BCE loss, AdamW, ReduceLROnPlateau, early stopping on val.
- **XGB** — `n_estimators=10 000` capped by `early_stopping_rounds=10`, `max_depth=4`, `lr=0.1`, `subsample=0.8`, `eval_metric='logloss'`. Optuna sweeps depth, learning rate, subsample, and L2 over 500 trials maximising F1.
- **CatBoost** — recursive SHAP elimination chooses the feature set; hyperparameters via RandomizedSearchCV (60 fits). On the held-out test split: logloss 0.0903, AUC 0.9938.

### Results (held-out test split, 36 000 events)

| Model | Features | Accuracy | Precision | Recall | F1 (electron) | AUC |
|---|---|---|---|---|---|---|
| NN — MI (Optuna-tuned) | 15 (MI) | 0.938 | 0.957 | 0.738 | 0.833 | ~0.96 |
| NN — FI (Optuna-tuned) | 15 (FI) | 0.962 | 0.928 | 0.886 | 0.907 | ~0.99 |
| XGB — MI (Optuna-tuned) | 15 (MI) | 0.944 | 0.95 | 0.77 | 0.85 | — |
| **XGB — FI (full → top-15)** | **15 (FI)** | **0.974** | **0.95** | **0.92** | **0.94** | ~0.995 |
| **CatBoost (recursive SHAP)** | — | — | — | — | — | **0.9938** |

Submitted classification models: **NN-MI, NN-FI, XGB-FI, CatBoost** (CSVs under `Submission/`).

**Headline.** The feature subset selected by the boosted tree itself is more informative than the MI-ranked subset for *both* model families. The NN gains ~7 pp F1 just by switching feature lists at fixed architecture; the XGB gains ~9 pp. CatBoost with its own recursive-SHAP selection lands in the same AUC ballpark as the FI-track XGB, very slightly above on AUC alone.

A SHAP summary on the tuned NN-FI model (cell 10 of `NN_Class_XGB_feature_importance.ipynb`) shows the top contributors are `p_TRTPID`, `pX_MultiLepton`, and `p_Eratio` — consistent with the physics: TRT particle ID and shower-shape ratios dominate electron PID at ATLAS.

---

## Task 2 — Regression (electron energy, GeV)

Five trained models and one ensemble — all sharing the same setup so they're directly comparable:

- **20-feature cap** (rubric constraint). Two selectors used in practice: XGB feature-importance (for the standalone XGB / NN scripts) and **recursive CatBoost SHAP** (for the ensemble and its component models).
- **Log-target trick.** Train on `log(E)` with squared error; undo with `exp` at inference. For small deviations `log(p) − log(y) ≈ (p − y) / y`, so the optimisation signal is approximately the grading metric — *RelMAD* — rather than absolute GeV error, which would bias the model toward the high-energy tail.
- **`eval_metric='mape'`** on the boosted-tree side: early stopping tracks RelMAD directly.
- **Filter to true electrons** at preprocessing.

### Pipelines

- `Regression/Recursive_SHAP_Ensemble_Reg.py` — **headline submission.** End-to-end ensemble: CatBoost recursive-SHAP feature selection (20 features), then XGB + LightGBM + CatBoost + a 3-layer MLP trained on `log(E)`, combined as a weighted geometric mean. Per-model RandomizedSearchCV; Dirichlet search for ensemble weights on val.
- `Regression/XGB_Regression/XGB_Reg.py` — standalone XGB, three passes: full-features → top-20 by FI → top-20 RandomizedSearchCV-tuned.
- `Regression/NN_Reg/NN_Reg.ipynb` — standalone NN (`ThreeLayerRegressor`), full-features SHAP → top-20 → Optuna over layer widths, dropout, lr, weight decay, batch size. Bundles weights + scaler + feature ordering + hyperparameters + log-target flag into `NN_Reg_artifact.pth`.
- `Regression/CatBoost_Reg/catboost_reg.py` — standalone CatBoost.
- `Regression/LGB_Reg/LGB_REG.py` — standalone LightGBM.

### Results (local held-out test, 20% split)

| Model | Features | RelMAD ↓ |
|---|---|---|
| XGB (top 20 FI, RandomSearchCV) | 20 | 0.2009 |
| LightGBM (top 20 SHAP) | 20 | 0.2014 |
| CatBoost (top 20 SHAP) | 20 | 0.2058 |
| NN (top 20 SHAP, Optuna) | 20 | 0.2303 |
| **Ensemble (weighted geo mean)** | **20** | **0.1987** |

Ensemble weights (Dirichlet search on val): XGB 0.385 / LGB 0.372 / CatBoost 0.240 / NN 0.003. The NN gets down-weighted close to zero but the ensemble still edges the best single learner.

**Headline.** Tree-based models dominate on this tabular feature set, and the ensemble beats every standalone learner — small absolute gain (~1 pp) but consistent across val and test. The NN has a heavy right tail on extrapolated events and is capped at `exp(14)` in `Regression_inference.py` to keep RelMAD finite.

### Omitted from submission — PySR (symbolic regression)

`Regression/PySr_Regression/PySR.ipynb` runs PySR over the true-electron rows to recover closed-form `log(E)` expressions. Best run reached only ~0.55 mean-RelMAD — outclassed by every trained model above, so not submitted, but kept in-repo as a sanity check on what a hand-derivable formula can do. The seven candidates from the best run, evaluated against the project's `mean(|E_pred − E_true| / E_true)` metric on all true-electron rows of the training file:

| Tag | MSE (log E) | RelMAD | Expression |
|---|---|---|---|
| C1  | 1.096 | 162.98% | `y = 10.66` |
| C2  | 0.810 | 102.18% | `y = log(pX_E3x5_Lr2)` |
| C4  | 0.576 | 112.57% | `y = log(pX_E3x5_Lr2 + p_pt_track)` |
| C5  | 0.503 | 113.84% | `y = abs(p_eta) + log(p_pt_track)` |
| C6  | 0.434 |  70.83% | `y = log(p_pt_track + 0.534·pX_E3x5_Lr2)` |
| C7  | 0.358 |  64.67% | `y = log(p_pt_track · (1 + eta²))` |
| C10 | 0.291 |  54.89% | `y = log(0.686·pX_ecore / (p_ptcone40 + 1.25) + p_pt_track)` |

C10 is the best closed-form expression and still ~2.8× worse than the ensemble. Useful as a floor for "what could a physicist guess unaided" rather than a competitive model.

---

## Repository layout

```
.
├── Classification/
│   ├── NN_Classifier/
│   │   ├── NN_Class_mutual_information.ipynb        # NN-MI
│   │   ├── NN_Class_XGB_feature_importance.ipynb    # NN on XGB-FI features
│   │   ├── NN_Class_Full_feature_importance.ipynb   # NN-SHAP (full → top-15)
│   │   ├── NN_CLASS_Tuned_Params.txt
│   │   └── saved_models/
│   ├── XGB_Classifier/
│   │   ├── XGB_Mutual_information.ipynb
│   │   ├── XGB_feature_importance.ipynb
│   │   └── saved_models/
│   ├── CatBoost_classifier/
│   │   └── CatBoost_Classifie.ipynb                 # recursive-SHAP CatBoost
│   └── Input_lists/                                  # persisted feature subsets
│
├── Regression/
│   ├── Recursive_SHAP_Ensemble_Reg.py               # ensemble — headline
│   ├── Regression_inference.py                      # shared inference / NN cap
│   ├── XGB_Regression/
│   │   ├── XGB_Reg.py                               # 3-pass XGB script
│   │   ├── XGB_Reg_plots/
│   │   └── saved_models/
│   ├── NN_Reg/
│   │   ├── NN_Reg.ipynb                             # NN regressor (+ Optuna)
│   │   ├── NN_Reg_SHAP.ipynb                        # full-feature SHAP selection
│   │   ├── NN_Reg_artifact.pth                      # weights + scaler bundle
│   │   └── NN_Reg_plots/
│   ├── CatBoost_Reg/
│   │   ├── catboost_reg.py
│   │   └── saved_models/
│   ├── LGB_Reg/
│   │   ├── LGB_REG.py
│   │   └── saved_models/
│   ├── PySr_Regression/                             # symbolic regression 
│   │   ├── PySR.ipynb
│   │   └── outputs/                                 # per-run PySR equation dumps
│   └── Input_lists/                                  # XGB_REG_INPUT, 
│
├── Modules/
│   ├── Utils.py     # 3 dataloaders: fast_preprocess (NN cls),
│   │               # full_feature_data_preprocess (XGB cls),
│   │               # XGB_REG_DATALOADER (regression, e-only)
│   └── models.py    # NN classifiers + ThreeLayerRegressor + XGBoostModel wrapper
│
├── Submission/      # final CSVs + description (gitignored)
└── Data/            # train + blind classification + blind regression (HDF5)
```

---

## FUTURE IMPLEMENTATION

Path invariance - currently uses absolute paths for dataloading in config files of some scripts, should be fixed.

Dataloader utils, currently multple functions that should be merged into one with configs triggering the different variances, instead of having whole different functions.
Eg standardize as config option, sort by mi, remove correlation etc.

Unify the standalone regression scripts. XGB_Reg.py, catboost_reg.py, and LGB_REG. Highly similar so. They should be a single config driven entry point.

Streamline and organize models (input lists, parameters and so forth currently new ones save directly to submission format, old ones to /input or /saved_models)
Write callable scripts of the old models, which only exists in notebooks for now.
