"""Lightweight smoke tests that do not require downloading real model weights.

Run with:
    pytest tests/test_pipeline.py
or:
    python tests/test_pipeline.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cacx_seg import Config, resolve_overlaps
from cacx_seg.device import get_device
from cacx_seg.inference import LesionInstanceInference
from cacx_seg.model import build_model


def test_resolve_overlaps_intersection_not_union():
    m1 = np.zeros((100, 100), dtype=np.uint8)
    m1[10:40, 10:40] = 1
    m2 = np.zeros((100, 100), dtype=np.uint8)
    m2[25:55, 25:55] = 1  # overlaps m1
    m3 = np.zeros((100, 100), dtype=np.uint8)
    m3[70:90, 70:90] = 1  # does not overlap either
    scores = [0.5, 0.7, 0.8]

    final_masks, final_scores, groups = resolve_overlaps([m1, m2, m3], scores, min_overlap_fraction=0.05)

    assert len(final_masks) == 2
    merged = [fm for fm, g in zip(final_masks, groups) if len(g) > 1][0]
    assert merged.sum() < m1.sum()
    assert merged.sum() < m2.sum()
    assert any(len(g) == 1 for g in groups)


def test_end_to_end_with_untrained_model():
    device = get_device()
    cfg = Config(pretrained=False)
    model = build_model(cfg, device)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "ckpt.pth")
        torch.save({"state_dict": model.state_dict()}, ckpt_path)

        inferencer = LesionInstanceInference(ckpt_path, cfg, device=device)

        import cv2
        img = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)
        img_path = os.path.join(tmp, "test.png")
        cv2.imwrite(img_path, img)

        masks, scores = inferencer.predict(img_path)
        assert isinstance(masks, list)
        assert isinstance(scores, list)
        assert len(masks) == len(scores)


if __name__ == "__main__":
    test_resolve_overlaps_intersection_not_union()
    test_end_to_end_with_untrained_model()
    print("All smoke tests passed.")
