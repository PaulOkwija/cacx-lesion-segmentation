"""Postprocessing of raw predicted masks: confidence filtering, small-object
removal, and hole filling.
"""
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects

from .config import Config


def _remove_small_objects_compat(mask: np.ndarray, min_size: int):
    # scikit-image has deprecated the `min_size` keyword in favor of
    # `max_size` (despite both meaning "the size threshold below which
    # objects are removed"); support both so this keeps working across
    # versions.
    try:
        return remove_small_objects(mask, min_size=min_size)
    except TypeError:
        return remove_small_objects(mask, max_size=min_size)


def postprocess_instances(masks: list, scores: list, cfg: Config):
    kept_masks, kept_scores = [], []
    for m, s in zip(masks, scores):
        if s < cfg.score_thresh:
            continue
        mb = _remove_small_objects_compat(m.astype(bool), min_size=cfg.min_object_size)
        mb = binary_fill_holes(mb)
        if mb.sum() == 0:
            continue
        kept_masks.append(mb.astype(np.uint8))
        kept_scores.append(float(s))
    return kept_masks, kept_scores
