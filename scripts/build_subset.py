"""Build a class subset dataset from a config file.

Usage (from the project root):
    python scripts/build_subset.py configs/sod_subset_v1.yaml
"""
import sys
from pathlib import Path

import yaml

from assistive_vision.data.subset import build_subset


def main() -> None:
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    stats = build_subset(
        source=Path(cfg["source"]),
        output=Path(cfg["output"]),
        classes=cfg["classes"],
        splits=cfg["splits"],
        keep_images_without_kept_classes=cfg.get("keep_images_without_kept_classes", False),
    )
    for split, s in stats.items():
        print(f"{split}: {s['images']} images kept, {s['skipped']} skipped, instances {s['instances']}")


if __name__ == "__main__":
    main()
