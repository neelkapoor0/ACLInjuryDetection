# ACL Injury Detection Notebook

This project trains and compares two modeling approaches for binary ACL injury detection:
- Classical ML on HOG features (SGD-SVM and SGD-LogReg)
- CNN on image-like tensors built from MRI slices

The main workflow is in [main.ipynb](main.ipynb).

## Data Inputs
- MRI volumes are stored as `.pck` files under `MRI_Images/vol*/`.
- Labels come from `metadata.csv`.
- Original label mapping in metadata: `0=Normal`, `1=Torn`, `2=Partially torn`.
- Notebook converts labels to binary:
  - `0 -> Normal`
  - `1 -> Torn`
  - `2 -> Torn` (merged into positive class)

## Preprocessing Pipeline

### 1) File Discovery
The notebook scans all `vol*` folders and collects `.pck` file paths.

### 2) Label Attachment
Each file name is matched to `metadata.csv`, then converted to a binary label.

### 3) Volume-to-Model Input Conversion
For each selected MRI volume, preprocessing creates two parallel representations:

1. HOG input (classical models):
- Extract a center grayscale slice
- Resize to `IMG_SIZE = (64, 64)`
- Compute HOG descriptor using:
  - `orientations=9`
  - `pixels_per_cell=(8, 8)`
  - `cells_per_block=(2, 2)`

2. CNN input (deep model):
- Preferred: stack center slice and neighboring slices (`center-1`, `center`, `center+1`) as 3 channels
- Fallback: repeat a single center slice into 3 channels if needed
- Resize each slice to `64x64`
- Keep values as float32 and clip to `[0, 1]` before CNN training

### 4) Balanced Sampling + Cache
- Target max samples: `MAX_SAMPLES = 400`
- Sampling is approximately class-balanced
- Preprocessed arrays are cached in `cache/preprocessed.npz`:
  - `X_hog`
  - `X_images`
  - `y_labels`

### 5) Shared Train/Validation Split
A deterministic split (seeded) is created once and reused across models:
- HOG split for SGD-SVM / SGD-LogReg
- Image split for CNN

## Model Training Sections in Notebook
- Cell 4: preprocessing + HOG model training/evaluation
- Cell 5: CNN training/evaluation
- Cell 6: side-by-side metrics and confusion matrix plots

## Reproducibility
The notebook sets fixed seeds for:
- Python random
- NumPy
- TensorFlow

This keeps sampling, split behavior, and training initialization more consistent across runs.
