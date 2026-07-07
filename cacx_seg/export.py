"""Export predicted instances to a portable label format.

Per-instance binary PNG masks are written directly by cacx_seg.inference; this
module additionally builds a COCO-style JSON with polygon segmentations
(rather than RLE), so no extra dependency such as pycocotools is required.
The resulting file can be opened directly, imported into common annotation
tools (CVAT, Label Studio, Roboflow), or used to build a training set for
further model development.
"""
import json
import os

import cv2
import numpy as np

CATEGORY_ID = 1
CATEGORY_NAME = "lesion"


def mask_to_polygons(mask: np.ndarray, min_points: int = 6):
    """Convert a binary mask to a list of COCO-style flattened polygons:
    one polygon per contour, as [x1, y1, x2, y2, ...]."""
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        if len(c) < 3:
            continue
        poly = c.reshape(-1, 2).astype(float).flatten().tolist()
        if len(poly) >= min_points:
            polygons.append(poly)
    return polygons


def build_coco_dataset(records: list):
    """Build a COCO-format dict from per-image detections.

    `records` is a list of dicts, one per image:
        {"file_name": str, "width": int, "height": int,
         "masks": [HxW uint8, ...], "scores": [float, ...]}
    """
    images, annotations = [], []
    ann_id = 1
    for image_id, rec in enumerate(records, start=1):
        images.append({
            "id": image_id,
            "file_name": rec["file_name"],
            "width": rec["width"],
            "height": rec["height"],
        })
        for mask, score in zip(rec["masks"], rec["scores"]):
            polygons = mask_to_polygons(mask)
            if not polygons:
                continue
            ys, xs = np.nonzero(mask)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": CATEGORY_ID,
                "segmentation": polygons,
                "area": float(mask.sum()),
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                "iscrowd": 0,
                "score": float(score),
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME, "supercategory": "lesion"}],
    }


def save_coco_json(records: list, output_path: str):
    coco = build_coco_dataset(records)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
    return coco
