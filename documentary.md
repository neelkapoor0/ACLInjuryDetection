### 08/11/2026

- **Dataset Setup & Challenges:**
  - Initially used RSNA 3D knee MRI data (~24,000+ slices, ~4,407 studies).
  - ACL labels were limited to 58 studies (34 intact, 24 torn).
  - Soft-label filtering produced extreme imbalance (4,383 intact vs. 24 torn), making training unreliable.

- **Pivot to Stanford MRNet:**
  - Switched to Stanford MRNet for cleaner, supervised ACL labels.
  - Began integrating MRNet MRI data and ACL label metadata.

### 08/14/2026

- **Data Pipeline:**
  - Set up automated MRNet downloading and case-ID filtering.
  - Standardized ACL classification as binary: normal vs. torn.
  - Added preprocessing and caching for faster experiments.

- **Initial Model Results:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| SGD-SVM | 0.622 | 0.588 | 0.556 | 0.571 |
| SGD-LogReg | 0.605 | 0.556 | 0.648 | 0.598 |
| CNN | **0.714** | **0.652** | **0.796** | **0.717** |

- CNN performed best in the initial experiment.

### 08/24/2026

- **Model Improvements:**
  - Fixed issues that prevented the CNN, SVM, and Logistic Regression models from learning effectively.
  - Added model-specific training settings.
  - Added hyperparameter tuning.
  - Added 5-fold cross-validation.

- **5-Fold Results:**

| Model | Mean Accuracy | Std. Dev. |
|---|---:|---:|
| SGD-SVM | 0.645 | 0.050 |
| SGD-LogReg | 0.680 | 0.022 |
| CNN | **0.685** | 0.047 |

- **Out-of-Fold Results:**

| Model | Accuracy | Normal F1 | Torn F1 |
|---|---:|---:|---:|
| SGD-SVM | 0.65 | 0.65 | 0.64 |
| SGD-LogReg | 0.68 | 0.68 | 0.68 |
| CNN | **0.69** | 0.67 | **0.70** |

### 08/31/2026

- **Class Balancing:**
  - Balanced the training data because the natural dataset was approximately 79% Normal / 21% Torn.
  - Undersampled the Normal class to match the Torn class.

- **Preprocessing:**
  - Used raw MRI pixel data only.
  - Center slice resized to 64×64 grayscale and normalized to [0,1].
  - No HOG or additional feature engineering.

- **Model Expansion:**
  - Tested 3 models across all 3 MRI views:
    - SVM
    - Logistic Regression
    - CNN
  - Tuned each model separately for each view.
  - Saved all 9 optimized models.

- **10-Fold Results:**

| Model | View | Test Accuracy | Test F1 |
|---|---|---:|---:|
| SVM | Axial | **0.676** | **0.676** |
| LogReg | Axial | 0.629 | 0.628 |
| CNN | Axial | 0.657 | 0.655 |
| SVM | Coronal | 0.514 | 0.503 |
| LogReg | Coronal | 0.581 | 0.581 |
| CNN | Coronal | 0.571 | 0.571 |
| SVM | Sagittal | 0.581 | 0.581 |
| LogReg | Sagittal | 0.581 | 0.580 |
| CNN | Sagittal | 0.648 | 0.644 |

- **Key Findings:**
  - Best individual model: **Axial SVM** (67.6% test accuracy, 67.6% F1).
  - Best MRI view: **Axial** (65.4% average test accuracy).
  - Coronal was consistently the weakest view.
  - CNN performed particularly well on sagittal images (64.8% test accuracy, 64.4% F1).
  - The best cross-validation result was the Axial SVM at **71.9% ± 5.7%**.
  - Sagittal models showed noticeable CV-to-test performance drops, suggesting some overfitting.
  - Fixed a bug in the SVM hyperparameter search where `gamma` had previously been omitted.
  - Fixed CNN training to consistently use 8 epochs across tuning, CV, and final training.
  - Verified that reloaded saved models reproduce the original test results exactly.

### 09/07/2026

- **Dataset Validation:**
  - Removed class balancing to preserve the natural training distribution.
  - Verified that only labeled cases are included and that axial, coronal, and sagittal views contain matching case IDs.
  - Dropped the Stanford `valid_acl_labels.csv`/`StanfordMRNet/valid` split entirely — it's no longer read anywhere in the pipeline. `main.ipynb` now loads only `train_acl_labels.csv`/`StanfordMRNet/train`, and carves its own stratified case-level 80/20 train/test split out of that single pool (`random_state=SEED`), with the inner tuning split further carved from the 80% train side.

- **Final Dataset (train-only pool, self-split 80/20):**

| Split | Normal | Torn | Total |
|---|---:|---:|---:|
| Train | 738 | 166 | 904 |
| Test | 184 | 42 | 226 |
| Inner Train | 590 | 133 | 723 |
| Inner Holdout | 148 | 33 | 181 |

- All 1,130 labeled cases in `train_acl_labels.csv` have a matching MRI file for every view, so nothing is excluded.
- No unlabeled MRI images are included.
- No class balancing, oversampling, or undersampling is used on the dataset itself.

- **Pipeline & Imbalance Handling Rework:**
  - Replaced case-level undersampling (which had thrown away over half the data to force a 50/50 split) with `class_weight="balanced"` instead, so SVM, Logistic Regression, and CNN now train on the full, naturally-imbalanced pool (~82% Normal / 18% Torn) and compensate via per-class loss weighting rather than dropping data.
  - Added a `StandardScaler` step before SVM and Logistic Regression (`sklearn.pipeline.make_pipeline`), since raw flattened pixel features were previously fed in unscaled.
  - CNN now computes fold-specific and final `class_weight` via `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` instead of training unweighted.
  - Model/view selection now uses CV metrics only — the test set is no longer looked at when picking the best combo (previously the writeup also reported "best combo by test F1", which peeks at the held-out set).
  - Simplified the Redivis download cell (removed unused label-inference helpers; case filtering now reads the headerless CSVs directly).

- **10-Fold CV Results (train-only 80/20 split, class-weighted):**

| Model | View | CV Accuracy | CV F1 | Test Accuracy | Test F1 |
|---|---|---:|---:|---:|---:|
| SVM | Axial | 0.537 | 0.502 | 0.535 | 0.508 |
| LogReg | Axial | 0.759 | 0.590 | **0.783** | 0.638 |
| CNN | Axial | 0.683 | 0.619 | 0.717 | 0.654 |
| SVM | Coronal | 0.480 | 0.441 | 0.323 | 0.323 |
| LogReg | Coronal | 0.722 | 0.550 | 0.637 | 0.474 |
| CNN | Coronal | 0.551 | 0.490 | 0.535 | 0.494 |
| SVM | Sagittal | **0.812** | 0.634 | 0.792 | 0.640 |
| LogReg | Sagittal | 0.785 | 0.642 | 0.748 | 0.608 |
| CNN | Sagittal | 0.725 | **0.645** | 0.690 | 0.629 |

- **Key Findings:**
  - Best combo by CV F1 (CV-only selection): **CNN — Sagittal** (CV accuracy 0.725, CV F1 0.645); test result was 0.690 accuracy / 0.629 F1.
  - Best model by mean CV accuracy: **Logistic Regression**. Best view by mean CV accuracy: **Sagittal**.
  - Least stable across folds: **SVM — Coronal** (fold accuracy std 0.208) — Coronal remains the weakest, most volatile view.
  - Scaling features before SVM/LogReg and switching to `class_weight="balanced"` avoided the earlier majority-class collapse (seen 08/31 on Coronal) without discarding any data.
  - Verified all 9 reloaded models from `optimized_models/` reproduce their reported test accuracy exactly.
  - Not directly comparable to 08/31's undersampled results: the pool size, class ratio, and test-set source all changed at once.