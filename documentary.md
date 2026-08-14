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