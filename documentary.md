### 08/11/2026

* **Dataset Setup & Challenges (RSNA Knee):**
  * Downloaded RSNA 3D knee MRI data (~24,000+ slices, ~4,407 studies).
  * Early deletion/cleanup loops failed due to index-shift and null-handling issues (`IndexError`, `NaN` edge cases).
  * Manual expert ACL labels were limited (58 total studies: 34 intact, 24 torn), with no strong isolated ACL-tear subset.
  * Soft-label thresholding (`pseudo_ACL > 0.5`) produced severe imbalance (4,383 intact vs. 24 torn), making training unstable.

* **Pivot to Stanford MRNet:**
  * Switched from noisy pseudo-labeling to Stanford MRNet for cleaner supervised ACL targets.
  * Added MRNet-aligned image assets and metadata workflows to the project.
  * Began Redivis-based integration for split tables and ACL labels.

### 08/14/2026

* **Redivis Pipeline Stabilization:**
  * Updated MRNet loading to the current Redivis API pattern (`table.file(...).download(...)`).
  * Downloaded split metadata (`train`, `valid`) and ACL label tables (`train-acl`, `valid-acl`) into local project storage.
  * Implemented split-wise filtering by case ID so only files referenced by split ACL labels are retained.
  * Added safe preview mode for cleanup (`DELETE = False`) before destructive removal (`DELETE = True`).

* **Notebook Pipeline Update (ACL, Sagittal):**
  * Standardized training/evaluation on MRNet `train/valid` sagittal `.npy` volumes.
  * Added robust label loading (`case_id -> binary ACL label`) with flexible column inference.
  * Cached preprocessing outputs to `cache/preprocessed_*_sagittal.npz` for faster reruns.

* **Imbalance Handling:**
  * Added inverse-frequency class weighting for HOG models (via `sample_weight` in `SGDClassifier.partial_fit`).
  * Added inverse-frequency class weighting for CNN training (`class_weight` in `model.fit`).

* **CNN Simplification (Binary Setup):**
  * Replaced 2-logit softmax head with a 1-unit sigmoid output layer.
  * Switched loss to `binary_crossentropy`.
  * Switched prediction step from `argmax` to thresholding (`p >= 0.5`).

* **Reporting:**
  * Kept per-model classification reports.
  * Kept unified comparison table (accuracy / precision / recall / F1).
  * Kept side-by-side and individual confusion matrix plots.

* **Model comparison:**
```
     model  accuracy  precision   recall  f1_score
   SGD-SVM  0.621849   0.588235 0.555556  0.571429
SGD-LogReg  0.605042   0.555556 0.648148  0.598291
       CNN  0.714286   0.651515 0.796296  0.716667
```

### 08/24/2026

* **Reproducible, unbiased splits:**
  * Replaced the fixed train/test split with a pooled shuffle — every run now pools all available images and draws a fresh random 80% train / 20% test split, giving a more honest, less "lucky" read on generalization.

* **CNN training bug (image model was learning nothing):**
  * Diagnosed a layer that was collapsing almost all useful information out of each image before the model could use it — the CNN was effectively guessing (~coin-flip predictions) while appearing to train normally.
  * Removed the offending step so the model can actually learn from image features.

* **SVM / Logistic Regression training bug (frozen accuracy):**
  * Found both simpler models were sharing a single learning-rate setting with the CNN, sized far too small for them, which made their accuracy look flat across training.
  * Gave each model its own properly-scaled learning rate — both now show real improvement (and eventual overfitting) during training, as expected.

* **Proper hyperparameter tuning:**
  * Swept a range of settings per model (learning rate, training rounds, etc.) and scored each combination with k-fold cross-validation, keeping the best-performing combination per model.

* **Added k-fold cross-validation:**
  * Replaced single train/test evaluation with 5-fold CV — data is split into 5 groups, each model is trained and tested 5 times (holding out a different group each time), and results are aggregated across all 5 for a more reliable performance estimate.

* **Results (5-fold CV, mean ± std accuracy):**
  * SGD-SVM: 0.645 ± 0.050
  * SGD-LogReg: 0.680 ± 0.022
  * CNN: 0.685 ± 0.047

**What training looked like:**

![alt text](image.png)

**Results — out-of-fold, all folds combined:**

*SGD-SVM classification report*

| | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Normal | 0.64 | 0.65 | 0.65 | 200 |
| Torn | 0.65 | 0.64 | 0.64 | 200 |
| **Accuracy** | | | **0.65** | 400 |
| Macro avg | 0.65 | 0.65 | 0.64 | 400 |
| Weighted avg | 0.65 | 0.65 | 0.64 | 400 |

*SGD-SVM confusion matrix (rows=true, cols=predicted)*

| | Pred: Normal | Pred: Torn |
|---|---|---|
| **True: Normal** | 130 | 70 |
| **True: Torn** | 72 | 128 |

*SGD-LogReg classification report*

| | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Normal | 0.68 | 0.68 | 0.68 | 200 |
| Torn | 0.68 | 0.68 | 0.68 | 200 |
| **Accuracy** | | | **0.68** | 400 |
| Macro avg | 0.68 | 0.68 | 0.68 | 400 |
| Weighted avg | 0.68 | 0.68 | 0.68 | 400 |

*SGD-LogReg confusion matrix (rows=true, cols=predicted)*

| | Pred: Normal | Pred: Torn |
|---|---|---|
| **True: Normal** | 136 | 64 |
| **True: Torn** | 64 | 136 |

*CNN classification report*

| | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Normal | 0.70 | 0.64 | 0.67 | 200 |
| Torn | 0.67 | 0.73 | 0.70 | 200 |
| **Accuracy** | | | **0.69** | 400 |
| Macro avg | 0.69 | 0.69 | 0.68 | 400 |
| Weighted avg | 0.69 | 0.69 | 0.68 | 400 |

*CNN confusion matrix (rows=true, cols=predicted)*

| | Pred: Normal | Pred: Torn |
|---|---|---|
| **True: Normal** | 128 | 72 |
| **True: Torn** | 54 | 146 |

### 08/31/2026

* **Fixed the train/test split methodology (case-level, leak-proof):**
  * The `train/`/`valid/` folders no longer define the experimental split — all cases from both are pooled together, then split once at the case-ID level into a stratified 80% train / 20% test partition. Axial/coronal/sagittal now share the exact same train and test case IDs by construction (verified programmatically), instead of relying on incidental agreement across three independent per-view splits.
  * The 20% test set is now a true held-out set: it is never touched during hyperparameter tuning or 10-fold CV, only for a single final evaluation per model/view — this final-test-set evaluation didn't exist in the notebook before today and has been added.
  * Hyperparameter tuning now uses an inner 80/20 split carved out of the 80% train pool only (fit on ~64% of the data, score on ~16%), so tuning decisions can never leak into either the 10-fold CV metrics or the final test metrics. The 10-fold CV itself now runs over the *entire* 80% train pool (not a further-subsampled slice), using one shared fold split reused identically across all 9 combinations for a fair comparison.

* **Reintroduced class balancing:**
  * The natural dataset is imbalanced (~79% Normal / 21% Torn), and a same-day run without any balancing produced degenerate results on the coronal view (SVM/CNN both collapsed to always predicting "Normal"). Reintroduced case-level class balancing — undersampling Normal down to match the Torn count (262 cases each, 524 pooled total) — before the train/test split, so the balanced ratio is preserved through the split, the inner tuning holdout, and every CV fold.
  * Preprocessing is otherwise unchanged: raw center-slice MRI pixels only (resize to 64×64, normalize to [0,1]), no HOG or other feature engineering.

* **Final optimized hyperparameters:**
  * SVM — Axial `C=1, linear` · Coronal `C=0.1, rbf` · Sagittal `C=0.1, linear`. LogReg — Axial `C=1, lbfgs` · Coronal `C=0.1, lbfgs` · Sagittal `C=0.1, liblinear`. CNN — Axial `lr=1e-3, epochs=8` · Coronal `lr=1e-4, epochs=8` · Sagittal `lr=1e-4, epochs=8` (epochs fixed at 8 for all views; only learning rate is tuned).
  * All 9 final models (refit on the full 80% train pool) saved to `optimized_models/` (not committed to git).

* **Results (10-fold CV over the 80% train pool, mean ± std accuracy, vs. final evaluation on the untouched 20% test set):**

| Model | View | CV Accuracy | Test Accuracy | Test F1 |
|---|---|---:|---:|---:|
| SVM | Axial | 0.636 ± 0.049 | 0.610 | 0.609 |
| Logistic Regression | Axial | 0.666 ± 0.048 | 0.629 | 0.628 |
| CNN | Axial | 0.706 ± 0.051 | 0.657 | 0.655 |
| SVM | Coronal | 0.511 ± 0.033 | 0.495 | 0.331 |
| Logistic Regression | Coronal | 0.556 ± 0.067 | 0.581 | 0.581 |
| CNN | Coronal | 0.595 ± 0.039 | 0.571 | 0.571 |
| SVM | Sagittal | 0.643 ± 0.033 | 0.581 | 0.581 |
| Logistic Regression | Sagittal | 0.680 ± 0.047 | 0.581 | 0.580 |
| CNN | Sagittal | 0.689 ± 0.064 | 0.648 | 0.644 |

**CNN training curves (tuned hyperparameters, per view):**

![alt text](image_multiview_cnn_curves.png)

* **Findings:**
  * Best combination: **CNN — Axial** (CV accuracy 0.706, test accuracy 0.657, test F1 0.655) — the strongest result across all 9 combinations, on both CV and the untouched test set, though modestly lower than the earlier tuned-epoch run (test accuracy 0.686) since epochs are now fixed at 8 rather than allowed up to 20.
  * Best model by mean test accuracy across views: **CNN** (0.625), ahead of LogReg (0.597) and SVM (0.562).
  * Best view by mean test accuracy across models: **Axial** (0.632), ahead of Sagittal (0.603) and Coronal (0.549, the weakest view again).
  * SVM — Coronal is essentially at chance (test accuracy 0.495, F1 0.331) — the weakest combination overall.
  * Sagittal still shows the largest CV-vs-test gap (LogReg 0.099, SVM 0.061) — consistent with the earlier run, unaffected by the CNN epoch change since neither model is a CNN.
  * SVM/LogReg results are unaffected by the CNN epoch change and are byte-identical to the previous run (both are fully deterministic given the fixed seed and unchanged code path) — confirmed via a new verification cell (below) that reloads all 9 saved models from `optimized_models/` and re-evaluates them on the test set, matching every reported metric exactly.
  * Balancing removed the majority-class-collapse failure mode seen in the same-day unbalanced run, at the cost of roughly halving the usable dataset (1248 → 524 pooled cases) — a real precision/recall/data-volume tradeoff, not a free improvement.

* **Fixed CNN epochs to 8:**
  * Simplified the CNN hyperparameter grid to fix `epochs=8` for every view (previously tuned over {10, 20}), leaving learning rate as the only tuned CNN hyperparameter. Applied consistently everywhere CNN training happens — tuning, training-curve plot, 10-fold CV, and the final saved model — so there's no mismatch between what was tuned and what was deployed.

* **Added a model-reload verification cell:**
  * New final cell reloads each of the 9 saved models directly from `optimized_models/` (not the in-memory objects from training) and re-evaluates them on the untouched 20% test set, asserting the reloaded predictions reproduce the exact `test_accuracy`/`test_precision`/`test_recall`/`test_f1` already in `results_df` — confirms the saved artifacts are trustworthy for later reuse.