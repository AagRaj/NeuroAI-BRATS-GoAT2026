#!/usr/bin/env python3
"""
BraTS-GoAT: build a stratified splits_final.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

NCR, ED, ET = 1, 2, 3
PCTS = (50, 95)


def load_seg(labels_dir, cid):
    return np.asarray(nib.load(labels_dir / f"{cid}.nii.gz").dataobj).astype(np.uint8)


def load_modality(images_dir, cid, chan):
    return np.asarray(nib.load(images_dir / f"{cid}_{chan:04d}.nii.gz").dataobj).astype(np.float32)


def fingerprint(images_dir, labels_dir, cid):
    seg = load_seg(labels_dir, cid)
    vols = np.array([(seg == l).sum() for l in (NCR, ED, ET)], dtype=np.float64)
    present = (vols > 0).astype(np.float64)
    log_vols = np.log1p(vols)
    et_frac = vols[2] / max(vols.sum(), 1.0)

    t1n = load_modality(images_dir, cid, 0)
    brain = t1n > 0
    if not brain.any():
        raise ValueError(f"{cid}: empty t1n")
    idx = np.argwhere(brain)
    bbox = (idx.max(0) - idx.min(0) + 1).astype(np.float64)
    brain_vol = np.array([np.log1p(brain.sum())])

    inten = []
    for chan in (1, 3):
        v = load_modality(images_dir, cid, chan)[brain]
        p = np.percentile(v, PCTS)
        denom = p[1] if abs(p[1]) > 1e-6 else 1.0
        inten.extend([p[0] / denom, np.log1p(abs(p[1]))])

    return np.concatenate([present, log_vols, [et_frac], bbox, brain_vol, inten])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--preprocessed", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images_dir, labels_dir = args.raw / "imagesTr", args.raw / "labelsTr"
    cids = sorted(p.name[: -len(".nii.gz")] for p in labels_dir.glob("*.nii.gz"))
    print(f"[1/4] fingerprinting {len(cids)} cases")

    feats = []
    for i, cid in enumerate(cids):
        feats.append(fingerprint(images_dir, labels_dir, cid))
        if (i + 1) % 200 == 0:
            print(f"      {i+1}/{len(cids)}", flush=True)
    X = np.vstack(feats)

    print(f"[2/4] k-means, k={args.k}")
    Xs = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit_predict(Xs)

    print("\n      cluster   n    NCR+    ED+    ET+   med_ET_frac")
    for c in range(args.k):
        m = labels == c
        n = int(m.sum())
        print(f"      {c:^7d} {n:4d}  {X[m,0].mean():5.2f}  {X[m,1].mean():5.2f}  "
              f"{X[m,2].mean():5.2f}   {np.median(X[m,6]):.3f}")

    print(f"\n[3/4] stratified {args.folds}-fold over clusters")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = []
    for f, (tr, va) in enumerate(skf.split(cids, labels)):
        splits.append({"train": [cids[i] for i in tr], "val": [cids[i] for i in va]})
        comp = np.bincount(labels[va], minlength=args.k)
        print(f"      fold {f}: {len(va):3d} val, cluster mix {comp.tolist()}")

    out = args.preprocessed / "splits_final.json"
    out.write_text(json.dumps(splits, indent=4))
    print(f"\n[4/4] wrote {out}")

    np.save(args.preprocessed / "goat_cluster_labels.npy", labels)


if __name__ == "__main__":
    sys.exit(main())
