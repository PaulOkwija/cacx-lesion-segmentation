"""Test-time augmentation.

The model is run on the original image plus horizontal and vertical flips.
Detections that refer to the same underlying lesion (same location once
flipped back, high mask IoU) are merged into one instance with an averaged
confidence score; this is more robust to orientation-dependent artifacts than
a single forward pass.
"""
import cv2
import numpy as np
import torch

from .config import Config


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def merge_instances(views: list, iou_thresh: float = 0.4):
    """Merge instance masks predicted across TTA views (original + flips).

    Each entry in `views` is {"masks": [HxW uint8, ...], "scores": [float, ...]},
    with masks already realigned to the original image orientation. Instances
    that overlap across views (IoU >= iou_thresh) are treated as the same
    lesion: their masks are unioned and their scores averaged.
    """
    pooled_masks, pooled_scores = [], []
    for v in views:
        pooled_masks.extend(v["masks"])
        pooled_scores.extend(v["scores"])

    if not pooled_masks:
        return [], []

    order = np.argsort(pooled_scores)[::-1]
    used = [False] * len(pooled_masks)
    merged_masks, merged_scores = [], []

    for idx in order:
        if used[idx]:
            continue
        group_mask = pooled_masks[idx].astype(bool).copy()
        group_scores = [pooled_scores[idx]]
        used[idx] = True
        for j in order:
            if used[j]:
                continue
            if _mask_iou(group_mask, pooled_masks[j]) >= iou_thresh:
                group_mask = np.logical_or(group_mask, pooled_masks[j].astype(bool))
                group_scores.append(pooled_scores[j])
                used[j] = True
        merged_masks.append(group_mask.astype(np.uint8))
        merged_scores.append(float(np.mean(group_scores)))

    return merged_masks, merged_scores


def predict_tta_instances(model, img_rgb: np.ndarray, cfg: Config, device):
    variants = [("none", img_rgb)]
    if cfg.tta_flip:
        variants.append(("h", cv2.flip(img_rgb, 1)))
        variants.append(("v", cv2.flip(img_rgb, 0)))

    model.eval()
    views = []
    with torch.no_grad():
        for flip_type, im in variants:
            img_t = (torch.from_numpy(np.ascontiguousarray(im).transpose(2, 0, 1)).float() / 255.0).to(device)
            out = model([img_t])[0]
            scores = out["scores"].detach().cpu().numpy()
            if len(scores) == 0:
                views.append({"masks": [], "scores": []})
                continue
            masks = (out["masks"][:, 0].detach().cpu().numpy() >= cfg.mask_binarize_thresh).astype(np.uint8)
            masks_list = [masks[i] for i in range(masks.shape[0])]
            if flip_type == "h":
                masks_list = [cv2.flip(m, 1) for m in masks_list]
            elif flip_type == "v":
                masks_list = [cv2.flip(m, 0) for m in masks_list]
            views.append({"masks": masks_list, "scores": scores.tolist()})

    return merge_instances(views, cfg.tta_merge_iou_thresh)
