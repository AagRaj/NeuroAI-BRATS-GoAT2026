#!/usr/bin/env python3
"""
BraTS-GoAT 2026 (Task 3) -> nnU-Net v2 raw dataset.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_GoAT"

CHANNELS = {0: "t1n", 1: "t1c", 2: "t2w", 3: "t2f"}

NCR, ED, ET = 1, 2, 3

REGIONS = {
    "background": 0,
    "whole tumor": (NCR, ED, ET),
    "tumor core": (NCR, ET),
    "enhancing tumor": (ET,),
}
REGIONS_CLASS_ORDER = (ED, NCR, ET)


def case_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("BraTS-GoAT-"))


def link(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def build_split(cases, images_dir, labels_dir):
    ids = []
    for case in cases:
        cid = case.name
        for idx, suffix in CHANNELS.items():
            link(case / f"{cid}-{suffix}.nii.gz", images_dir / f"{cid}_{idx:04d}.nii.gz")
        if labels_dir is not None:
            link(case / f"{cid}-seg.nii.gz", labels_dir / f"{cid}.nii.gz")
        ids.append(cid)
    return ids


def seg_to_regions(seg):
    return np.stack([
        np.isin(seg, REGIONS["whole tumor"]),
        np.isin(seg, REGIONS["tumor core"]),
        np.isin(seg, REGIONS["enhancing tumor"]),
    ])


def regions_to_seg(regions):
    out = np.zeros(regions.shape[1:], dtype=np.uint8)
    for chan, label in enumerate(REGIONS_CLASS_ORDER):
        out[regions[chan]] = label
    return out


def verify(labels_dir, n, seed=0):
    files = sorted(labels_dir.glob("*.nii.gz"))
    if not files:
        print("  [verify] no labels found, skipping", file=sys.stderr)
        return
    rng = random.Random(seed)
    sample = rng.sample(files, min(n, len(files)))
    empty = {"NCR": 0, "ED": 0, "ET": 0}
    for f in sample:
        seg = np.asarray(nib.load(f).dataobj).astype(np.uint8)
        stray = set(np.unique(seg)) - {0, NCR, ED, ET}
        if stray:
            raise ValueError(f"{f.name}: unexpected label values {sorted(stray)}")
        rebuilt = regions_to_seg(seg_to_regions(seg))
        if not np.array_equal(seg, rebuilt):
            bad = int((seg != rebuilt).sum())
            raise AssertionError(f"{f.name}: round-trip mismatch on {bad} voxels.")
        for name, lab in (("NCR", NCR), ("ED", ED), ("ET", ET)):
            if not (seg == lab).any():
                empty[name] += 1
    k = len(sample)
    print(f"  [verify] round-trip OK on {k} cases")
    print(f"  [verify] absent: " + ", ".join(f"{n_}={c}/{k}" for n_, c in empty.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-gt", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify", type=int, default=20)
    args = ap.parse_args()

    root = args.out / DATASET_NAME
    imagesTr, labelsTr, imagesTs = root / "imagesTr", root / "labelsTr", root / "imagesTs"

    print(f"[1/3] linking training cases -> {imagesTr}")
    train_ids = build_split(case_dirs(args.train_gt), imagesTr, labelsTr)
    print(f"      {len(train_ids)} cases (expect 1351)")

    print(f"[2/3] linking validation cases -> {imagesTs}")
    val_ids = build_split(case_dirs(args.val), imagesTs, None)
    print(f"      {len(val_ids)} cases (expect 451)")

    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(f"train/val ID overlap: {sorted(overlap)[:5]}")

    dataset_json = {
        "channel_names": {str(i): s for i, s in CHANNELS.items()},
        "labels": {k: (list(v) if isinstance(v, tuple) else v) for k, v in REGIONS.items()},
        "regions_class_order": list(REGIONS_CLASS_ORDER),
        "numTraining": len(train_ids),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
        "name": DATASET_NAME,
        "description": "BraTS-GoAT 2026 Task 3. Labels 1=NCR 2=ED 3=ET. Region-based.",
    }
    (root / "dataset.json").write_text(json.dumps(dataset_json, indent=4))
    print(f"[3/3] wrote {root / 'dataset.json'}")

    if args.verify:
        verify(labelsTr, args.verify)

    print("\nnext:")
    print(f"  export nnUNet_raw={args.out}")


if __name__ == "__main__":
    sys.exit(main())
