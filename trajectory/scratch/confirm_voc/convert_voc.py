"""Unzip + convert VOC to the ultralytics layout (images/{train,val}{2007,2012}, images/test2007 + labels/), same
conversion as ultralytics/cfg/datasets/VOC.yaml (difficult objects excluded)."""
import os, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(os.path.expanduser("~/datasets/VOC")); IM = ROOT / "images"; IM.mkdir(exist_ok=True)
NAMES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
for z in ("VOCtrainval_06-Nov-2007.zip", "VOCtest_06-Nov-2007.zip", "VOCtrainval_11-May-2012.zip"):
    if not (IM / "VOCdevkit").exists() or z.startswith("VOCtrainval_11") and not (IM / "VOCdevkit" / "VOC2012").exists() or z.startswith("VOCtest") and not (IM / "VOCdevkit" / "VOC2007" / "ImageSets" / "Main" / "test.txt").exists():
        print("unzip", z); zipfile.ZipFile(ROOT / z).extractall(IM)
def convert_label(path, lb_path, year, image_id):
    def box(size, b):
        dw, dh = 1.0 / size[0], 1.0 / size[1]
        x, y, w, h = (b[0] + b[1]) / 2.0 - 1, (b[2] + b[3]) / 2.0 - 1, b[1] - b[0], b[3] - b[2]
        return x * dw, y * dh, w * dw, h * dh
    root = ET.parse(path / f"VOC{year}/Annotations/{image_id}.xml").getroot(); size = root.find("size"); w, h = int(size.find("width").text), int(size.find("height").text)
    with open(lb_path, "w") as out:
        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls in NAMES and int(obj.find("difficult").text) != 1:
                bb = obj.find("bndbox"); b = box((w, h), [float(bb.find(x).text) for x in ("xmin", "xmax", "ymin", "ymax")])
                out.write(" ".join(str(a) for a in (NAMES.index(cls), *b)) + "\n")
path = IM / "VOCdevkit"
for year, image_set in ("2012", "train"), ("2012", "val"), ("2007", "train"), ("2007", "val"), ("2007", "test"):
    imgs_path = IM / f"{image_set}{year}"; lbs_path = ROOT / "labels" / f"{image_set}{year}"; imgs_path.mkdir(exist_ok=True, parents=True); lbs_path.mkdir(exist_ok=True, parents=True)
    ids = open(path / f"VOC{year}/ImageSets/Main/{image_set}.txt").read().strip().split(); n = 0
    for i in ids:
        f = path / f"VOC{year}/JPEGImages/{i}.jpg"; dst = imgs_path / f.name
        if f.exists(): f.rename(dst)
        if dst.exists(): convert_label(path, (lbs_path / f.name).with_suffix(".txt"), year, i); n += 1
    print(f"{image_set}{year}: {n} images")
