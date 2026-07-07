"""Configuration for the cervical lesion instance segmentation pipeline.

All tunable behavior lives in a single dataclass so that a run can be fully
described (and reproduced) by one object.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # --- Paths ---
    checkpoint_path: str = "cacx_seg_maskrcnn.pth"
    output_dir: str = "outputs"

    # --- Model / Mask R-CNN internal resize ---
    # These control the internal GeneralizedRCNNTransform, not the files on
    # disk -- images are resized automatically and predictions are mapped
    # back to the original resolution.
    image_size: int = 640       # min_size
    max_size: int = 960         # max_size
    pretrained: bool = True     # COCO-pretrained backbone; only relevant when (re)training from scratch
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 30   # generous cap; colposcopy exams rarely show more than 30 lesions

    # --- Inference / postprocessing ---
    score_thresh: float = 0.5          # minimum confidence to keep a predicted instance
    mask_binarize_thresh: float = 0.5  # soft mask -> binary
    min_object_size: int = 40          # pixels; smaller predicted instances are treated as noise
    tta_flip: bool = True              # test-time augmentation via horizontal + vertical flips
    tta_merge_iou_thresh: float = 0.4  # IoU threshold used to merge/dedup instances across TTA views

    # --- Overlap resolution ---
    # Two DISTINCT detected instances (not TTA views of the same one) can
    # still occupy overlapping image regions. overlap_min_fraction is the
    # minimum fraction of the smaller instance's area that must be covered by
    # the other instance for the pair to be treated as the same lesion and
    # collapsed to their intersection. See cacx_seg/overlap.py.
    overlap_min_fraction: float = 0.05
