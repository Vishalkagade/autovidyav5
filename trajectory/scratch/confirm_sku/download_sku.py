"""Download + convert SKU-110K via the ultralytics dataset yaml (train 8,219 / val 588 / test 2,936; single class)."""
from ultralytics.data.utils import check_det_dataset
d = check_det_dataset("SKU-110K.yaml", autodownload=True)
print("SKU_DOWNLOAD_DONE", d["path"], d["train"], d["val"], d["test"])
