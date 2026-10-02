# Pneumonia Detection from Chest X-Rays (PyTorch)

Classifying pediatric chest X-rays as **NORMAL** or **PNEUMONIA** with convolutional
neural networks, with a focus on handling dataset pitfalls and evaluating the model
the way a clinician would.

> ⚠️ Educational project, not a medical device. Not for clinical use.

## Project structure

```
PytorchXRay/
├── data/chest_xray/        # the dataset (you download this; not committed to git)
├── outputs/<model_name>/   # saved models, plots, metrics (created when you train)
├── src/
│   ├── config.py           # all settings (hyperparameters) in one place
│   ├── utils.py            # random seeds + picking GPU/CPU
│   ├── dataset.py          # loading images, train/val split, augmentation, class weights
│   ├── model.py            # SimpleCNN baseline, ResNet-18, DenseNet-121
│   ├── train.py            # the training loop
│   ├── check_leakage.py    # checks train/val/test for shared patients or images
│   ├── shape_check.py      # does the model use image shape instead of the lungs?
│   ├── gradcam.py          # heatmaps of where the model looks
│   ├── evaluate.py         # test-set metrics, confusion matrix, ROC curve
│   └── predict.py          # run a trained model on new X-ray images
└── requirements.txt
```

Suggested reading order: `config → dataset → model → train → evaluate`.

## Setup (Mac)

Open **Terminal** and run these one at a time:

```bash
cd ~/Desktop/PytorchXRay

# 1. Create a virtual environment: a private Python install just for this project,
#    so package versions here don't clash with other projects.
python3 -m venv .venv

# 2. Activate it. You must do this every time you open a new terminal.
#    Your prompt will show (.venv) when it's active.
source .venv/bin/activate

# 3. Install the libraries.
pip install -r requirements.txt
```

## Get the data

1. Make a free account at kaggle.com.
2. Download **"Chest X-Ray Images (Pneumonia)"** by Paul Mooney
   (kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), ~1.2 GB.
3. Unzip it and move the `chest_xray` folder into `data/`, so you have
   `data/chest_xray/train`, `data/chest_xray/val`, `data/chest_xray/test`.
   (The zip sometimes contains a duplicate nested `chest_xray/chest_xray` and a
   `__MACOSX` folder — you only need one copy of train/val/test.)

Check that it worked:

```bash
python -m src.dataset
```

You should see image and patient counts for train/val/test, a check that no
patient appears in more than one split, and the shape of one batch.

Then run the full leakage check once (takes a minute or two):

```bash
python -m src.check_leakage
```

It looks for shared patients, identical files and near-identical images across
train/val/test, plus filenames it can't read a patient ID from. It should end with
`RESULT: no leakage found`. If it doesn't, fix or document what it lists before
trusting any test results.

## Train and evaluate

```bash
# Quick test run first (1 epoch) to make sure everything works:
python -m src.train --model_name simple_cnn --epochs 1

# Real runs:
python -m src.train --model_name simple_cnn --epochs 15
python -m src.train --model_name resnet18 --epochs 10
python -m src.train --model_name densenet121 --epochs 10 --batch_size 16

# Evaluate each one on the test set:
python -m src.evaluate --model_name resnet18

# Predict on new images (uses the threshold tuned by evaluate, if available):
python -m src.predict path/to/xray.jpeg --model_name resnet18
```

On an Apple Silicon Mac, PyTorch uses the GPU automatically ("mps"). Expect a few
minutes per epoch for ResNet-18. If you run into problems, add `--num_workers 0`.
For faster training, the same code runs on Google Colab's free GPU.

## Results

All numbers are on the official test set (234 normal, 390 pneumonia images) at
threshold 0.5. The threshold tuned on validation gave the same or almost the same
results for the ResNet-18 runs, because validation is nearly perfect and can't
guide it. Each model was trained once.

| Model | Preprocessing | Epochs (best) | Params | Test AUC | Sensitivity | Specificity | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| SimpleCNN (scratch) | squash | 1 (1) | 0.39 M | 0.873 | 94.6% | 50.9% | 0.844 | 78.2% |
| ResNet-18 (pretrained) | squash | 10 (8) | 11.2 M | 0.969 | 99.5% | 68.4% | 0.911 | 87.8% |
| **ResNet-18 (pretrained)** | **crop + blur 0.3** | **10 (10)** | **11.2 M** | **0.976** | **99.7%** | **71.4%** | **0.920** | **89.1%** |
| DenseNet-121 (pretrained) | crop + blur 0.3 | _to run_ | 7.0 M | | | | | |

- **Main model:** ResNet-18 with crop. It misses 1 of 390 pneumonia cases but
  flags 67 of 234 healthy children. Most of those false alarms are one type of
  X-ray; see Limitations.
- **SimpleCNN row is a 1-epoch test run** and will be replaced by a 15-epoch run
  with crop preprocessing. It shows how much pretraining helps.
- **Validation vs test:** both ResNet-18 runs reach about 99% validation
  accuracy but 88–89% on test, because the test set differs from the training
  distribution.

## Design decisions

- **Re-split validation data:** the official val set has only 16 images, so train+val
  are pooled and about 15% is held out, keeping the class ratio the same.
- **Split by patient, not by image:** many children have several X-rays. A random
  image-level split would put the same child in both train and validation, making
  validation scores (which pick the saved epoch and the threshold) look too good.
  Patient IDs come from the filenames, and `StratifiedGroupKFold` keeps each child
  on one side.
- **Leakage check on the real data:** `src/check_leakage.py` finds no shared
  patients, identical files or near-identical images across train/val/test.
  170 `personN` numbers appear in both training and test data, but always as
  bacterial on one side and viral on the other. Within each infection type the
  numbers rise in step with the image numbers, and 808 training IDs would
  otherwise have both infection types, which is implausible. So the numbers are
  very likely counted separately per infection type, and these are different
  children. The train/val split still groups them together, to be safe.
- **Class imbalance (~3:1 pneumonia:normal):** handled with a class-weighted loss.
- **No horizontal flips** in augmentation: they would move the heart to the wrong side.
- **Threshold tuned on validation, never on test.**

## Shortcut experiment: image shape

**Finding.** In this dataset NORMAL X-rays are larger and closer to square
(training median 1640 × 1328, width/height 1.22) while PNEUMONIA X-rays are
smaller and wider (1168 × 776, 1.49). A classifier given only width, height and
aspect ratio (no pixels) reaches **test AUC 0.895**. The default preprocessing
squashes every image into a square, so aspect ratio survives as distortion.

**Evidence the model uses it** (ResNet-18, `resize_mode squash`, threshold 0.5,
from `python -m src.shape_check --model_name resnet18`):

| Original shape of NORMAL test image | Images | Wrongly flagged as pneumonia |
|---|---|---|
| Square-ish (w/h < 1.30) | 98 | 12.2% |
| Medium (1.30–1.40) | 41 | 24.4% |
| Wide (≥ 1.40) | 95 | **54.7%** |

Among NORMAL test images, wider images get higher pneumonia scores (correlation
+0.54). Test normals are wider than training normals (41% vs 9% have w/h ≥ 1.40),
which explains much of the low test specificity. Grad-CAM on the most confident
false alarms (`python -m src.gradcam --model_name resnet18`) often highlights
corners, the "R" marker and image edges rather than the lung fields.

**Experiment.** `--resize_mode crop` keeps proportions and cuts out the centre
square, so every image has the same shape. `--blur_prob` adds random blur to
weaken resolution differences. Compare against the squash run:

```bash
caffeinate -i python -m src.train --model_name resnet18 --epochs 10 --resize_mode crop --blur_prob 0.3 --run_name resnet18_crop
python -m src.evaluate --run_name resnet18_crop
python -m src.shape_check --run_name resnet18_crop
python -m src.gradcam --run_name resnet18_crop
```

Caveats: cropping trims the sides of wide images (about 16% per side at w/h 1.5),
and wider normal images might also differ in other ways (for example patient age).
Report both runs, not only the better one.

**Result.** Cropping helped a little but did not remove the effect:

| Run (threshold 0.5) | AUC | Sensitivity | Specificity | False alarms: square-ish / medium / wide normals | Shape vs score correlation |
|---|---|---|---|---|---|
| ResNet-18, squash | 0.969 | 99.5% | 68.4% | 12.2% / 24.4% / 54.7% | +0.54 |
| ResNet-18, crop + blur 0.3 | 0.976 | 99.7% | 71.4% | 9.2% / 19.5% / 52.6% | +0.51 |

The gain (7 fewer false alarms out of 234 normals) is within what a different
random seed could produce. Two follow-up tests on normal test images:

- **Not pixel resolution.** Shrinking large healthy images to typical pneumonia
  image size (about 0.85 MP) changed no predictions in either model.
- **Content, not geometry.** Among wide normals, the smaller images are flagged far
  more often (81% vs 25% for the larger half). Since shrinking alone does nothing,
  these images most likely differ in what they show (for example smaller or
  younger children, or different positioning), and only 124 of 1,341 training
  normals look like this.

Conclusion: the remaining false alarms come from a kind of normal X-ray that is
rare in training, not from the resize step. Preprocessing cannot fix that; more
training examples of that kind (or rebalancing them) might.

## Roadmap

- [x] Data pipeline, baselines, clinical metrics
- [x] Grad-CAM heatmaps (what is the model looking at?)
- [x] Shape shortcut found and measured (see above)
- [x] Retrain with `--resize_mode crop` and compare (small gain, effect remains)
- [ ] Rebalance training so rare normal image types are seen more often
- [ ] Out-of-distribution test on another dataset (e.g. RSNA Pneumonia)
- [ ] Calibration analysis
- [ ] Gradio demo on Hugging Face Spaces

## Limitations

This is a learning project, not a diagnostic tool. The main limitations, most
important first:

**1. Performance differs sharply by type of X-ray.** On healthy test children,
specificity depends on the original image shape (ResNet-18, crop, threshold 0.5,
from `python -m src.shape_check --run_name resnet18_crop`):

| Healthy test X-rays, by original width/height | Images | Correctly cleared (specificity) | Wrongly flagged |
|---|---|---|---|
| Square-ish (< 1.30) | 98 | **90.8%** (89) | 9 |
| Medium (1.30–1.40) | 41 | 80.5% (33) | 8 |
| Wide (≥ 1.40) | 95 | **47.4%** (45) | 50 |

Sensitivity is 98–100% in every group, so the gap is only in false alarms. Three
quarters of all false alarms (50 of 67) are wide healthy X-rays. The overall
specificity of 71.4% hides this: a healthy child whose X-ray looks like this group
is flagged about half the time. The likely cause is that this kind of healthy X-ray
is rare in training (124 of 1,341 normals) while most pneumonia images look like
it; changing the preprocessing did not remove the gap (see "Shortcut experiment").
A clinical tool would need this gap closed, or at least flagged, before use.

**2. One hospital, one age group.** All images come from one children's hospital
(Guangzhou Women and Children's Medical Center), from patients aged one to five.
Performance on adults, other hospitals, other X-ray machines or other patient
positions is unknown and could be much worse.

**3. The validation set overstates performance.** Validation accuracy is about 99%
but test accuracy is 89%, because the validation images come from the same pool as
training while the official test set differs, especially in its normal X-rays.
Only the test numbers should be quoted.

**4. Labels are simple.** Each image has one label, NORMAL or PNEUMONIA, with no
location of the infection, no bacterial/viral distinction in training, no other
diseases and no confirmation beyond the original graders. The model cannot say
where the infection is; Grad-CAM heatmaps are only a rough, unreliable hint.

**5. Patient grouping is inferred from filenames.** The patient-level split relies
on patient numbers in the filenames. A child with both a normal and a pneumonia
X-ray cannot be detected, because the two folders use different naming.

**6. Small test set, single runs.** The test set has 234 normal and 390 pneumonia
images, and each model was trained once with one random seed. Differences of a few
percentage points between runs (for example squash vs crop) may be chance.

**7. Probabilities are not calibrated.** The model is often 100% confident even when
wrong (the 8 most confident false alarms of the squash run all scored 1.00). The percentages
printed by `predict.py` should not be read as real probabilities.
