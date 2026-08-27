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