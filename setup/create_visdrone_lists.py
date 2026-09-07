"""One-time dataset lock generation for the VisDrone trajectory.

The dataset itself is already converted to YOLO format at DATASET_ROOT
(train 6,471 / val 548 / test 1,610 — dir name `test` = VisDrone test-dev).
This script only renders the Tier-3 lock artifacts:

- visdrone_train.txt   : sorted absolute paths, images/train (6,471)
- visdrone_eval.txt    : FROZEN eval set = sorted(val) + sorted(test) (2,158)
                         (user decision 2026-08-21)
- visdrone_val500.txt  : seed-42 random 500-image subset of the eval set —
                         per-epoch training-time val only (P12 curves),
                         never headline numbers
- visdrone.yaml        : headline eval yaml  (val = visdrone_eval.txt)
- visdrone_s1.yaml     : training yaml       (val = visdrone_val500.txt)
- adapters/yolo26n_visdrone_scratch/configs/yolo26n-visdrone.yaml
                       : ultralytics yolo26.yaml with nc: 10 (filename keeps
                         the yolo26n prefix so scale 'n' resolves)
- visdrone_manifest.json : sha256 fingerprint of every lock artifact —
                         the provenance record each experiment JSON pins

Idempotent; run from the repo root:  python setup/create_visdrone_lists.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.environ.get("VISDRONE_ROOT", "/home/atuin/v134ce/v134ce15/datasets/VisDrone")
YOLO26_YAML = os.path.normpath(os.path.join(
    PROJECT, "..", "ultralytics_src", "ultralytics", "cfg", "models", "26", "yolo26.yaml"))
MODEL_YAML_OUT = os.path.join(
    PROJECT, "adapters", "yolo26n_visdrone_scratch", "configs", "yolo26n-visdrone.yaml")

EXPECTED = {"train": 6471, "val": 548, "test": 1610}
VAL500_SEED = 42
VAL500_N = 500

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
         "tricycle", "awning-tricycle", "bus", "motor"]


def image_list(split: str) -> list[str]:
    d = os.path.join(DATASET_ROOT, "images", split)
    paths = sorted(os.path.join(d, f) for f in os.listdir(d)
                   if f.lower().endswith((".jpg", ".png")))
    if len(paths) != EXPECTED[split]:
        sys.exit(f"FATAL: {split} has {len(paths)} images, expected {EXPECTED[split]}")
    return paths


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def labels_sha256(image_paths: list[str]) -> str:
    """sha256 over the concatenated label files, in list order. A missing
    label file (image with no objects) hashes as the empty string."""
    h = hashlib.sha256()
    for p in image_paths:
        lp = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if os.path.exists(lp):
            with open(lp, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def write_list(name: str, paths: list[str]) -> str:
    out = os.path.join(PROJECT, name)
    with open(out, "w") as f:
        f.write("\n".join(paths) + "\n")
    print(f"  {name}: {len(paths)} images, sha256 {sha256_file(out)[:12]}…")
    return out


def render_data_yaml(name: str, train_list: str, val_list: str, comment: str) -> str:
    out = os.path.join(PROJECT, name)
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
    with open(out, "w") as f:
        f.write(f"""# VisDrone-DET, YOLO format — {comment}
# Rendered by setup/create_visdrone_lists.py (absolute paths, machine-specific
# → git-ignored; visdrone_manifest.json is the tracked fingerprint).

path: {DATASET_ROOT}
train: {train_list}
val: {val_list}

names:
{names_block}
""")
    print(f"  {name}: rendered")
    return out


def render_model_yaml() -> str:
    with open(YOLO26_YAML) as f:
        text = f.read()
    text, n = re.subn(r"(?m)^nc:\s*\d+", "nc: 10", text)
    if n != 1:
        sys.exit(f"FATAL: expected exactly one 'nc:' line in {YOLO26_YAML}, found {n}")
    header = ("# yolo26.yaml with nc: 10 for VisDrone — rendered by\n"
              "# setup/create_visdrone_lists.py from ultralytics_src (Tier-3 lock).\n"
              "# Filename keeps the 'yolo26n' prefix so ultralytics resolves scale n.\n")
    os.makedirs(os.path.dirname(MODEL_YAML_OUT), exist_ok=True)
    with open(MODEL_YAML_OUT, "w") as f:
        f.write(header + text)
    print(f"  configs/yolo26n-visdrone.yaml: rendered from {YOLO26_YAML}")
    return MODEL_YAML_OUT


def main() -> None:
    print("Generating VisDrone Tier-3 lock artifacts…")
    train = image_list("train")
    ev = image_list("val") + image_list("test")   # frozen eval: val + test-dev
    val500 = sorted(random.Random(VAL500_SEED).sample(ev, VAL500_N))

    train_f = write_list("visdrone_train.txt", train)
    eval_f = write_list("visdrone_eval.txt", ev)
    val500_f = write_list("visdrone_val500.txt", val500)
    render_data_yaml("visdrone.yaml", train_f, eval_f,
                     "headline eval (FROZEN: val + test-dev, 2,158 images)")
    render_data_yaml("visdrone_s1.yaml", train_f, val500_f,
                     "training-time val = locked 500-image subset (P12 curves only)")
    model_yaml = render_model_yaml()

    manifest = {
        "dataset_root": DATASET_ROOT,
        "eval_set_decision": "frozen eval = val + test-dev (2,158 images), user 2026-08-21",
        "train_list": {"file": "visdrone_train.txt", "n_images": len(train),
                       "sha256": sha256_file(train_f),
                       "labels_sha256": labels_sha256(train)},
        "eval_list": {"file": "visdrone_eval.txt", "n_images": len(ev),
                      "sha256": sha256_file(eval_f),
                      "labels_sha256": labels_sha256(ev)},
        "val500_list": {"file": "visdrone_val500.txt", "n_images": len(val500),
                        "seed": VAL500_SEED, "sha256": sha256_file(val500_f),
                        "labels_sha256": labels_sha256(val500)},
        "model_yaml": {"file": "adapters/yolo26n_visdrone_scratch/configs/yolo26n-visdrone.yaml",
                       "nc": 10, "sha256": sha256_file(model_yaml)},
        "note": ("Every experiment JSON records sha256(visdrone_manifest.json) as "
                 "provenance.data_manifest_sha256. Rendered yamls and lists are "
                 "git-ignored — they contain machine-specific absolute paths."),
    }
    out = os.path.join(PROJECT, "visdrone_manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  visdrone_manifest.json: written (sha256 {sha256_file(out)[:12]}…)")
    print("Done.")


if __name__ == "__main__":
    main()
