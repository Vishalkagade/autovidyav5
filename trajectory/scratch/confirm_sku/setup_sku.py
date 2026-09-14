"""Render the SKU-110K confirmation-track locks (ultralytics conversion at ~/.cache/autoyolo/dataset/SKU-110K):
train = train.txt (8,219); FROZEN eval = test.txt (2,936); train-time val = val.txt (588; curves only, last.pt lock).
Writes absolute-path lists, two data yamls, manifest."""
import hashlib, json, os
ROOT = os.path.expanduser("~/.cache/autoyolo/dataset/SKU-110K"); OUT = os.path.dirname(os.path.abspath(__file__))
def lst(name): return sorted(os.path.join(ROOT, l.strip()[2:] if l.strip().startswith("./") else l.strip()) for l in open(os.path.join(ROOT, name)) if l.strip())
train, ev, val = lst("train.txt"), lst("test.txt"), lst("val.txt"); assert (len(train), len(ev), len(val)) == (8219, 2936, 588), (len(train), len(ev), len(val))
assert all(os.path.exists(p) for p in train[:50] + ev[:50] + val[:50])
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def lsha(paths):
    h = hashlib.sha256()
    for p in paths:
        lp = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if os.path.exists(lp): h.update(open(lp, "rb").read())
    return h.hexdigest()
files = {}
for name, l in (("sku_train.txt", train), ("sku_eval.txt", ev), ("sku_val.txt", val)):
    f = os.path.join(OUT, name); open(f, "w").write("\n".join(l) + "\n"); files[name] = f
for name, v in (("sku.yaml", "sku_eval.txt"), ("sku_s1.yaml", "sku_val.txt")):
    open(os.path.join(OUT, name), "w").write(f"path: {ROOT}\ntrain: {files['sku_train.txt']}\nval: {files[v]}\nnames:\n  0: object\n")
json.dump({"dataset_root": ROOT, "source": "SKU110K_fixed.tar.gz (trax-geometry, CVPR challenge), ultralytics SKU-110K.yaml conversion",
           "train": {"n": len(train), "sha256": sha(files["sku_train.txt"]), "labels_sha256": lsha(train)},
           "eval": {"n": len(ev), "sha256": sha(files["sku_eval.txt"]), "labels_sha256": lsha(ev), "definition": "SKU-110K test split (frozen; one pass on last.pt per run)"},
           "val": {"n": len(val), "sha256": sha(files["sku_val.txt"]), "definition": "SKU-110K val split, training-time curves only"},
           "regime_at_640": "median box side ~40 px (train 38.5 / test 41.5), 28-36% of boxes under 32 px, median 141 boxes per image (max 386)"},
          open(os.path.join(OUT, "sku_manifest.json"), "w"), indent=2)
print("SKU locks rendered:", len(train), len(ev), len(val))
