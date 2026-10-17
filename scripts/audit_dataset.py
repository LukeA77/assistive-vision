from pathlib import Path
from collections import Counter
import yaml

root = Path("data/sod_v1")
cfg = yaml.safe_load((root / "data.yaml").read_text())
names = cfg["names"]
print("Classes:", names, "\n")

for split in ["train", "valid", "test"]:
    imgs = list((root / split / "images").glob("*.*"))
    labels = list((root / split / "labels").glob("*.txt"))
    counts, empty = Counter(), 0
    for f in labels:
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        if not lines:
            empty += 1
        for l in lines:
            counts[int(l.split()[0])] += 1
    print(f"{split}: {len(imgs)} images, {len(labels)} label files, {empty} empty")
    for c, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"   {names[c]}: {n}")
    print()