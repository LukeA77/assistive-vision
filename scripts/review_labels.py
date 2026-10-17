"""Draw sample labels per class into contact sheets for a visual quality check.

For each requested class, picks N random images (fixed seed) from one split that
contain that class, draws that class's boxes in red and all other boxes in faint
grey, and tiles them into one JPEG per class.

Usage (from the project root):
    python scripts/review_labels.py --root data/sod_v1 --split train \
        --out data/sod_v1_review --classes person car pole
"""
import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

TILE = 320   # each image is shrunk to TILE x TILE in the sheet
COLS = 5


def load_class_names(root: Path) -> list[str]:
    """Read class names from the dataset's data.yaml."""
    return yaml.safe_load((root / "data.yaml").read_text())["names"]


def read_labels(label_file: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into (class_id, x_centre, y_centre, width, height), all normalised 0-1."""
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def index_split(root: Path, split: str) -> dict[Path, list]:
    """Map every image in a split to its parsed boxes."""
    index = {}
    for img in sorted((root / split / "images").glob("*.jpg")):
        index[img] = read_labels(root / split / "labels" / f"{img.stem}.txt")
    return index


def draw_tile(img_path: Path, boxes: list, target_id: int) -> Image.Image:
    """Return one resized tile with the target class in red, other classes in grey."""
    img = Image.open(img_path).convert("RGB").resize((TILE, TILE))
    draw = ImageDraw.Draw(img)
    # Draw non-target boxes first so the target boxes sit on top.
    for cls, xc, yc, w, h in sorted(boxes, key=lambda b: b[0] == target_id):
        x0, y0 = (xc - w / 2) * TILE, (yc - h / 2) * TILE
        x1, y1 = (xc + w / 2) * TILE, (yc + h / 2) * TILE
        is_target = cls == target_id
        draw.rectangle([x0, y0, x1, y1],
                       outline=(255, 0, 0) if is_target else (170, 170, 170),
                       width=3 if is_target else 1)
    return img


def make_sheet(index: dict, target_id: int, n: int, seed: int) -> Image.Image | None:
    """Tile n random images containing the target class into one contact sheet."""
    candidates = [p for p, boxes in index.items() if any(b[0] == target_id for b in boxes)]
    if not candidates:
        return None
    picks = random.Random(seed).sample(candidates, min(n, len(candidates)))
    rows = -(-len(picks) // COLS)  # ceiling division
    sheet = Image.new("RGB", (COLS * TILE, rows * TILE), "white")
    for i, path in enumerate(picks):
        sheet.paste(draw_tile(path, index[path], target_id), ((i % COLS) * TILE, (i // COLS) * TILE))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/sod_v1"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=Path("data/sod_v1_review"))
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = load_class_names(args.root)
    index = index_split(args.root, args.split)
    args.out.mkdir(parents=True, exist_ok=True)

    for name in args.classes:
        sheet = make_sheet(index, names.index(name), args.n, args.seed)
        if sheet is None:
            print(f"{name}: no images found")
            continue
        out_path = args.out / f"{args.split}_{name}.jpg"
        sheet.save(out_path, quality=90)
        print(f"{name}: saved {out_path}")


if __name__ == "__main__":
    main()
