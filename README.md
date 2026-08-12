# BraTS-GoAT 2026 — Team NeuroAI

Generalisation Across Tumour Entities with a From-Scratch Residual-Encoder U-Net: A Baseline and an Analysis of Training-Validation Distribution Shift

Code for our BraTS-GoAT 2026 submission (Task 3 of the BraTS 2026 Cluster of Challenges, MICCAI 2026 Satellite Event). Team NeuroAI, Delhi Technological University.

Paper link goes here once it's up on OpenReview / the MICCAI proceedings.

## What this is

BraTS-GoAT wants one segmentation model that generalizes across five tumour populations (adult glioma, meningioma, metastases, pediatric HGG, sub-Saharan Africa cohort), even though training data is roughly 91% adult glioma. We train a Residual-Encoder nnU-Net (ResEnc-M) from scratch, no external data or pretrained weights, and use it to actually look at why generalization is hard here. The train/validation split is structurally imbalanced, so we fingerprint the training cohort with unsupervised clustering, build stratified CV folds around that, and report Dice per fold per cluster so the majority-to-minority degradation shows up instead of getting buried in one aggregate number.

## Results

Official Synapse validation leaderboard, 3-fold ensemble (folds 0, 1, 2):

| Region | DSC | NSD |
|---|---|---|
| ET | 0.7745 | 0.5290 |
| TC | 0.7942 | 0.4721 |
| WT | 0.8540 | 0.4443 |

3-fold beats the full 5-fold ensemble on every NSD column here. Paper has the fold-selection writeup.

## Method, briefly

Architecture: nnU-Net v2, ResEnc-M, `3d_fullres`, trained from scratch. 101.94M params.

Training: 250 epochs, 5-fold CV, `batch_dice=True` patched in after preprocessing (not the nnU-Net default). Needed because ~2.4% of cases have no enhancing tumour at all, so per-sample Dice goes degenerate on those, and because the leaderboard pools voxels across the whole cohort before scoring rather than per case. Single T4, roughly 300s/epoch, ~24h a fold.

Folds come from k-means fingerprinting the training cohort (k=5) on sub-region presence, volume, brain bounding-box extent, and per-modality intensity percentiles, not a random split. Seed 42. Two of the five clusters turned out to separate on brain bbox extent rather than actual tumour content — anatomical scale, not morphology, confirmed with an ablation. The other three separate on real tumour-content features.

Inference: mirror TTA and Gaussian-weighted sliding window (both on by default), `step_size 0.25` for denser windows since it measurably helps NSD (costs more compute though), `checkpoint_final.pth` rather than the EMA `checkpoint_best`. ET volume-thresholding and largest-WT-component filtering were both tried and both hurt pooled Dice under GoAT's scoring, so neither is used.

## Repo layout

```
.
├── README.md
├── REPRODUCE.md
├── LICENSE
├── goat_pipeline.ipynb   # run top to bottom, see REPRODUCE.md
├── goat_to_nnunet.py     # raw data -> nnU-Net format
├── goat_splits.py        # k-means fingerprint -> stratified 5-fold split, seed 42

```

`goat_pipeline.ipynb` runs as lettered sections on a fresh Kaggle session. GPU needs to be T4 x2 — P100 won't work, its PyTorch build has no `sm_60` kernels.

| Section | What it does |
|---|---|
| A | Install nnunetv2 + pinned numpy, set env vars |
| B | Stage data, `.niigz` -> `.nii.gz` symlinks (dodges Kaggle's auto-gunzip inflation) |
| C | Convert to nnU-Net format, preprocess |
| D | Patch `batch_dice=True`, build stratified splits |
| E | Train one fold |
| F | Predict on the 451 official validation cases |
| G | Verify geometry/labels, package the submission zip |

There are also cells for the analysis behind the paper's tables: pulling the training-config table out of logs/plans, checking convergence (EMA Dice / val-loss plateau), the stratified-vs-random Monte Carlo comparison, and the per-fold x per-cluster Dice scoring.

Checkpoints (`checkpoint_best.pth` / `checkpoint_final.pth` per fold) aren't in this repo — a few hundred MB each, too big for a plain git push. See `REPRODUCE.md`.

## Data

Get the training/validation data from [Synapse](https://www.synapse.org/brats2026) directly, under the BraTS-GoAT data-usage agreement. Not redistributed here.

## Citation

```bibtex
@inproceedings{neuroai2026goat,
  title     = {Generalisation Across Tumour Entities with a From-Scratch
               Residual-Encoder U-Net: A Baseline and an Analysis of
               Training--Validation Distribution Shift},
  author    = {Sinha, Aarush and Vasudeva, Mehak and Raj, Aagnik and
               Singh, Raghav and Pillai, Rohan},
  booktitle = {MICCAI 2026 BraTS-GoAT Challenge (BrainWorks Satellite Event),
               Lecture Notes in Computer Science, Springer},
  year      = {2026}
}
```

## License

MIT, see [`LICENSE`](LICENSE).

## Team

NeuroAI, Delhi Technological University — Aarush Sinha, Mehak Vasudeva, Aagnik Raj, Raghav Singh, Rohan Pillai.
