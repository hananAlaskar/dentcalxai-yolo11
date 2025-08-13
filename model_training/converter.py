from __future__ import annotations
"""yolo_project.converter
================================
Convert COCO JSON *instances_*.json* files into Ultralytics‑friendly YOLO labels.

The function below is a superset of the quick‑and‑dirty script you pasted:

* works on **Path** objects as well as strings
* creates a *labels/* directory automatically (and an optional *images/* one)
* can **remap COCO category IDs** to contiguous 0…N‑1 YOLO class IDs
* lets you **filter categories** (``category_whitelist={1,3}`` for example)
* links or copies images so the dataset is immediately consumable by
  ``train(data="yolo_dataset/data.yaml", …)``

Usage in a notebook
-------------------
```python
from pathlib import Path
from yolo_project.converter import coco_to_yolo

stats = coco_to_yolo(
    coco_json_path=Path("data/coco/instances_train2017.json"),
    images_dir=Path("data/coco/train2017"),
    output_dir=Path("data/yolo/train/labels"),  # any folder; it will be created
    category_whitelist=None,     # keep all categories
    remap_to_contiguous_ids=True,
)
print(stats)
```

Command‑line helper
~~~~~~~~~~~~~~~~~~~
```bash
python -m yolo_project.converter instances_val2017.json val_labels --images val2017
```

The CLI accepts ``--copy`` and ``--whitelist 1 3 5`` flags mirroring the Python
API.
"""
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

__all__ = [
    "coco_to_yolo",
    "YOLOConversionStats",
]

# -----------------------------------------------------------------------------
# Helper dataclass so callers get a quick summary programmatically
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class YOLOConversionStats:
    num_images: int
    num_annotations: int
    label_dir: Path
    image_dir: Path | None = None

    def __str__(self) -> str:  # pragma: no cover – cosmetic only
        out = [
            f"Images processed : {self.num_images}",
            f"Annotations total : {self.num_annotations}",
            f"Labels location   : {self.label_dir}",
        ]
        if self.image_dir:
            out.append(f"Images location  : {self.image_dir}")
        return "\n".join(out)


# -----------------------------------------------------------------------------
# Main public function
# -----------------------------------------------------------------------------

def coco_to_yolo(
    coco_json_path: str | Path,
    images_dir: str | Path | None,
    output_dir: str | Path,
    *,
    category_whitelist: Iterable[int] | None = None,
    remap_to_contiguous_ids: bool = True,
    copy_images: bool = False,
    verbose: bool = True,
) -> YOLOConversionStats:
    """Convert *coco_json_path* annotations into YOLO txt files.

    Parameters
    ----------
    coco_json_path : str | Path
        Path to the COCO ``instances_*.json`` file.
    images_dir : str | Path | None
        Directory that contains the corresponding images. If *None* no images
        are touched. Otherwise they will be *symlinked* into
        ``output_dir/../images`` (or copied when *copy_images=True*).
    output_dir : str | Path
        Destination directory for the generated ``labels/*.txt``.  If you pass
        a folder that does **not** end with ``labels`` the function will create
        the sub‑folder for you.
    category_whitelist : Iterable[int] | None, default *None*
        Convert *only* the specified ``category_id`` values.  By default all
        categories found in the JSON are kept.
    remap_to_contiguous_ids : bool, default *True*
        When *True* the COCO IDs get squashed to ``0…N-1`` (the order is the
        sorted order of the category IDs).  Set to *False* if you already have
        contiguously‑numbered categories starting at 0.
    copy_images : bool, default *False*
        Copy images instead of symlinking them; slower but works on Windows
        without admin rights.
    verbose : bool, default *True*
        Print a concise summary when finished.

    Returns
    -------
    YOLOConversionStats
        A dataclass with basic counts and destination paths, so you can assert
        things in your unit tests.
    """

    coco_json_path = Path(coco_json_path)
    output_dir = Path(output_dir)

    # Ensure we always write into …/labels/ even if the caller gave just a parent dir
    labels_dir = output_dir if output_dir.name == "labels" else output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Prepare images output if requested
    dest_images: Path | None = None
    if images_dir is not None:
        images_dir = Path(images_dir)
        dest_images = labels_dir.parent / "images"
        dest_images.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load COCO JSON as plain dict
    # ------------------------------------------------------------------
    with coco_json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    # Map category_id → YOLO id (maybe remapped)
    category_ids: list[int] = [cat["id"] for cat in coco["categories"]]
    if category_whitelist is not None:
        category_ids = [cid for cid in category_ids if cid in category_whitelist]
    category_ids.sort()
    cid2yolo: dict[int, int] = (
        {cid: idx for idx, cid in enumerate(category_ids)} if remap_to_contiguous_ids else {cid: cid for cid in category_ids}
    )

    # image_id → info
    images: dict[int, Mapping] = {im["id"]: im for im in coco["images"]}

    # Gather annotations by image → list[ann]
    anns_by_img: dict[int, list[Mapping]] = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid not in cid2yolo or ann.get("iscrowd", 0):
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # ------------------------------------------------------------------
    # Emit YOLO txt files and optionally link/copy images
    # ------------------------------------------------------------------
    total_anns = 0
    for img_id, anns in anns_by_img.items():
        im = images[img_id]
        w, h = im["width"], im["height"]
        base = Path(im["file_name"]).stem
        label_path = labels_dir / f"{base}.txt"

        with label_path.open("w", encoding="utf-8") as lf:
            for ann in anns:
                x, y, bw, bh = ann["bbox"]  # COCO xywh
                xc, yc = (x + bw / 2) / w, (y + bh / 2) / h
                lf.write(
                    f"{cid2yolo[ann['category_id']]} "
                    f"{xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}\n"
                )
        total_anns += len(anns)

        if dest_images is not None:
            src = images_dir / im["file_name"]
            dst = dest_images / im["file_name"]
            if not dst.exists():
                try:
                    dst.symlink_to(src)
                except (OSError, NotImplementedError):
                    if copy_images:
                        shutil.copy2(src, dst)

    stats = YOLOConversionStats(
        num_images=len(anns_by_img),
        num_annotations=total_anns,
        label_dir=labels_dir,
        image_dir=dest_images,
    )

    if verbose:
        print(stats)

    return stats

# -----------------------------------------------------------------------------
# Simple CLI (`python -m yolo_project.converter …`)
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover – keep it tiny
    import argparse

    ap = argparse.ArgumentParser(description="Convert COCO JSON to YOLO labels")
    ap.add_argument("coco_json", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--images", type=Path, help="directory that contains images")
    ap.add_argument("--whitelist", nargs="*", type=int, help="allowed category IDs")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    coco_to_yolo(
        coco_json_path=args.coco_json,
        images_dir=args.images,
        output_dir=args.output,
        category_whitelist=set(args.whitelist) if args.whitelist else None,
        copy_images=args.copy,
    )
