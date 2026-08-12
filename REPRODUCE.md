# Reproducing the BraTS-GoAT 2026 Results

The pipeline is one notebook, `goat_pipeline.ipynb`, built for Kaggle (GPU T4 x2 — P100 won't work, its PyTorch build has no `sm_60` kernels). Sections A-D need re-running at the start of every new Kaggle session since `/tmp` gets wiped when a session ends.

## Before you start

- `goat_to_nnunet.py` and `goat_splits.py` need to be in the same working directory as the notebook (`/kaggle/working/` on Kaggle).
- BraTS-GoAT training and validation data, registered and downloaded from [Synapse](https://www.synapse.org/brats2026). Not redistributed in this repo.

## A — Install and verify (~2 min)

```bash
pip install -q nnunetv2 nibabel "numpy>=2.0,<2.1"
```

The numpy pin has to be on the same install line as `nnunetv2`. Install them separately and pip resolves numpy 2.5.x first, which breaks nnU-Net's own C extensions (`cannot import name '_center'`) at import time.

Environment variables (Kaggle session paths, adjust for elsewhere):

```python
import os
os.environ["nnUNet_raw"]          = "/tmp/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/tmp/nnUNet_preprocessed"
os.environ["nnUNet_results"]      = "/kaggle/working/nnUNet_results"
os.environ["nnUNet_n_proc_DA"]    = "4"   # higher can OOM a 31GB-RAM host
```

## B — Stage the data (~2 min)

Files are expected as `.niigz`, not `.nii.gz`, so Kaggle doesn't auto-gunzip them on upload (22GB balloons to ~160GB otherwise). The notebook symlinks them to `.nii.gz` names so `nibabel`/`SimpleITK` decompress correctly — no copying, no extra disk. Expect 8,559 files linked.

## C — Convert + preprocess (~30 min, CPU-bound, nothing prints until it's done)

```bash
python goat_to_nnunet.py \
  --train-gt "<path-to>/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth" \
  --val      "<path-to>/MICCAI2024-BraTS-GoAT-ValidationData" \
  --out /tmp/nnUNet_raw --verify 40
# expect: 1351 training cases, 451 validation cases, round-trip OK

nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncM -c 3d_fullres -np 4
```

## D — batch_dice patch + stratified splits (~8 min)

Both steps get wiped if you re-run section C, since `plan_and_preprocess` rewrites the plans file. Re-apply both after any re-preprocess, every time.

```python
import json
P = "/tmp/nnUNet_preprocessed/Dataset501_GoAT/nnUNetResEncUNetMPlans.json"
pl = json.load(open(P))
pl["configurations"]["3d_fullres"]["batch_dice"] = True
json.dump(pl, open(P, "w"), indent=4)
# expect: batch_dice True | patch [128, 160, 112] | batch_size 2
```

```bash
python goat_splits.py \
  --raw /tmp/nnUNet_raw/Dataset501_GoAT \
  --preprocessed /tmp/nnUNet_preprocessed/Dataset501_GoAT \
  --k 5 --folds 5
```

Seed is fixed at 42 in `goat_splits.py` — that's what lets fold checkpoints trained by different people combine into one ensemble. Check fold 2 comes out to 1081 train / 270 val before moving on.

## E — Train one fold (~24h on a single T4, so 2 Kaggle sessions)

```bash
nnUNetv2_train 501 3d_fullres <FOLD> \
  -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs --npz --c
```

`--npz` saves softmax output, needed for ensembling. `--c` resumes from a checkpoint if one's there, harmless on a fresh start. Checkpoints save every 50 epochs — hit "Save Version" before Kaggle's 12h wall or `/kaggle/working` gets wiped. Next session: re-run A-D (~40 min), then this cell picks back up automatically.

The reported results only use folds 0, 1, and 2 ensembled, so you don't need all 5 folds trained to reproduce the headline numbers.

## F — Predict on the 451 validation cases

```bash
nnUNetv2_predict \
  -i /tmp/nnUNet_raw/Dataset501_GoAT/imagesTs \
  -o /tmp/goat_val_pred \
  -d 501 -c 3d_fullres \
  -f 0 1 2 \
  -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs \
  -chk checkpoint_final.pth -step_size 0.25
```

`-f 0 1 2` averages softmax across the three folds before argmax, that's the ensemble. `-step_size 0.25` (75% window overlap) costs 2-3x the inference time but measurably helps NSD, which is 3 of the 6 ranking columns. Mirror TTA and Gaussian window weighting are on by default, don't pass `--disable_tta`.

## G — Verify, then package

Check prediction count is 451, filenames match `BraTS-GoAT-\d{5}\.nii\.gz`, shape/affine match the source image, and labels are a subset of `{0,1,2,3}`. Mismatched geometry gets the whole submission invalidated on Synapse.

Zip the `.nii.gz` files flat and explicitly, not `shutil.make_archive` on the whole output folder — nnU-Net also drops `dataset.json`/`plans.json` in there, and that got our first submission rejected outright: "Not all files in the archive are NIfTI files."

## Docker (submission container)

```bash
cd docker
docker build -t neuroai-goat:latest .
docker compose up
```

Input mounted read-only, writes a flat prediction structure to `/output`.

## Environment gotchas worth knowing

- On Kaggle, use `subprocess.Popen` with line-by-line streaming for long jobs, not `%%bash` — it buffers everything until the process exits, which looks exactly like a hang.
- Checkpoints aren't in this repo. [Add wherever you end up hosting `checkpoint_best.pth`/`checkpoint_final.pth` per fold — Zenodo, HuggingFace, a GitHub Release, whatever you land on.]
