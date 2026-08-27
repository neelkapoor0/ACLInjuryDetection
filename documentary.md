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

Today was mostly about making the results we get from the notebook actually trustworthy, and about finding a couple of hidden problems that were quietly making two of our three models look worse than they really are.

**What we changed:**

* **Testing data now shuffles every time.** Before, the notebook always used the same fixed group of images for training and the same fixed group for testing. Now it pools every available image together and randomly picks a fresh 80% training / 20% testing split each time the notebook runs. This gives a more honest, less "lucky" read on how well a model generalizes.
* **Found and fixed a bug where the image model (CNN) wasn't learning anything at all.** It looked like it was training, but it was really just guessing — every single prediction came out close to a coin flip. The cause was a step in the network that was squashing away almost all of the useful information from each image before the model got to make a decision. Removing that step let the model actually start learning from the images.
* **Found and fixed a second bug where the two simpler models (SVM and Logistic Regression) looked frozen.** Their accuracy wasn't changing at all no matter how long we trained them, even though other numbers behind the scenes were moving. It turned out both of these models were accidentally sharing one "how fast should you learn" setting with the image model, and that setting was far too small for them. Giving each model its own separate, properly-sized setting fixed it — you can now see them actually improve (and eventually overfit) during training, which is normal and expected.
* **Tuned each model's settings properly instead of guessing.** We tried a range of settings for each model (things like how fast it learns and how many training rounds it gets) and tested each combination using k-fold cross-validation (explained below), keeping whichever combination scored best on average.
* **Added k-fold cross-validation.** Instead of judging a model on just one train/test split, we now split the data into 5 different groups, train and test the model 5 separate times (each time holding out a different group), and look at all 5 results together. This gives a much more reliable picture than any single split, since it's less likely to be thrown off by one lucky or unlucky split.
* **Cleaned up the notebook code** — removed unused leftover code and all comments to keep things shorter and easier to follow.

**Results — the big picture:**

Across all 5 test rounds, correctly identifying whether a knee scan showed a torn or normal ACL:

| Model | Overall accuracy | How good at catching "Normal" | How good at catching "Torn" |
|---|---|---|---|
| SVM | 71% | 71% | 72% |
| Logistic Regression | 73% | 73% | 73% |
| CNN (image model) | 69% | 68% | 70% |

All three models land in a similar, respectable range (69–73%) once evaluated fairly across 5 rounds — a big improvement in trustworthiness from before, when the CNN's real performance was being hidden by the training bug above. Logistic Regression came out slightly ahead this round, but the three are close enough that this can shift between runs (especially since the test data is reshuffled every run).

**Results — full detail, round by round:**

*Round 1*

| Model | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM | Normal | 0.69 | 0.55 | 0.61 |
| SVM | Torn | 0.62 | 0.75 | 0.68 |
| Logistic Regression | Normal | 0.69 | 0.62 | 0.66 |
| Logistic Regression | Torn | 0.66 | 0.72 | 0.69 |
| CNN | Normal | 0.62 | 0.65 | 0.63 |
| CNN | Torn | 0.63 | 0.60 | 0.62 |

Overall accuracy this round — SVM: 65%, Logistic Regression: 68%, CNN: 62%

*Round 2*

| Model | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM | Normal | 0.70 | 0.57 | 0.63 |
| SVM | Torn | 0.64 | 0.75 | 0.69 |
| Logistic Regression | Normal | 0.71 | 0.62 | 0.67 |
| Logistic Regression | Torn | 0.67 | 0.75 | 0.71 |
| CNN | Normal | 0.72 | 0.70 | 0.71 |
| CNN | Torn | 0.71 | 0.72 | 0.72 |

Overall accuracy this round — SVM: 66%, Logistic Regression: 69%, CNN: 71%

*Round 3*

| Model | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM | Normal | 0.76 | 0.80 | 0.78 |
| SVM | Torn | 0.79 | 0.75 | 0.77 |
| Logistic Regression | Normal | 0.76 | 0.80 | 0.78 |
| Logistic Regression | Torn | 0.79 | 0.75 | 0.77 |
| CNN | Normal | 0.71 | 0.62 | 0.67 |
| CNN | Torn | 0.67 | 0.75 | 0.71 |

Overall accuracy this round — SVM: 78%, Logistic Regression: 78%, CNN: 69%

*Round 4*

| Model | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM | Normal | 0.79 | 0.75 | 0.77 |
| SVM | Torn | 0.76 | 0.80 | 0.78 |
| Logistic Regression | Normal | 0.82 | 0.78 | 0.79 |
| Logistic Regression | Torn | 0.79 | 0.82 | 0.80 |
| CNN | Normal | 0.71 | 0.62 | 0.67 |
| CNN | Torn | 0.67 | 0.75 | 0.71 |

Overall accuracy this round — SVM: 78%, Logistic Regression: 80%, CNN: 69%

*Round 5*

| Model | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM | Normal | 0.66 | 0.82 | 0.73 |
| SVM | Torn | 0.77 | 0.57 | 0.66 |
| Logistic Regression | Normal | 0.69 | 0.82 | 0.75 |
| Logistic Regression | Torn | 0.78 | 0.62 | 0.69 |
| CNN | Normal | 0.74 | 0.72 | 0.73 |
| CNN | Torn | 0.73 | 0.75 | 0.74 |

Overall accuracy this round — SVM: 70%, Logistic Regression: 72%, CNN: 74%

("Normal" = no ACL tear, "Torn" = ACL tear. Each round tests on a different 1/5 slice of the training data, so some natural bounce between rounds is expected — that's exactly why we look at all 5 together instead of trusting just one.)