"""Render the VOC confirmation-track locks (after the ultralytics-style conversion into ~/datasets/VOC):
train = VOC07 trainval + VOC12 trainval (16,551); FROZEN eval = VOC07 test (4,952); val500 = seed-42 500-image
subset of the eval (training-time curves only; last.pt lock applies). Writes lists, two data yamls, manifest."""
import hashlib, json, os, random, sys
P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")); ROOT = os.path.expanduser("~/datasets/VOC"); OUT = os.path.dirname(os.path.abspath(__file__))
NAMES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
def imgs(split):
    d = os.path.join(ROOT, "images", split); return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg"))
train = imgs("train2007") + imgs("val2007") + imgs("train2012") + imgs("val2012"); ev = imgs("test2007")
assert len(train) == 16551 and len(ev) == 4952, (len(train), len(ev))
val500 = sorted(random.Random(42).sample(ev, 500))
def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
def lsha(paths):
    h = hashlib.sha256()
    for p in paths:
        lp = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if os.path.exists(lp): h.update(open(lp, "rb").read())
    return h.hexdigest()
files = {}
for name, lst in (("voc_train.txt", train), ("voc_eval.txt", ev), ("voc_val500.txt", val500)):
    f = os.path.join(OUT, name); open(f, "w").write("\n".join(lst) + "\n"); files[name] = f
names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
for name, val in (("voc.yaml", "voc_eval.txt"), ("voc_s1.yaml", "voc_val500.txt")):
    open(os.path.join(OUT, name), "w").write(f"path: {ROOT}\ntrain: {files['voc_train.txt']}\nval: {files[val]}\nnames:\n{names}\n")
json.dump({"dataset_root": ROOT, "train": {"n": len(train), "sha256": sha(files["voc_train.txt"]), "labels_sha256": lsha(train)},
           "eval": {"n": len(ev), "sha256": sha(files["voc_eval.txt"]), "labels_sha256": lsha(ev), "definition": "VOC07 test, difficult objects excluded (ultralytics conversion)"},
           "val500": {"n": 500, "seed": 42, "sha256": sha(files["voc_val500.txt"])}}, open(os.path.join(OUT, "voc_manifest.json"), "w"), indent=2)
print("VOC locks rendered:", len(train), len(ev))
