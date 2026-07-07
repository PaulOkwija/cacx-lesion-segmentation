"""Resolve spatially overlapping lesion instances.

Two detected instances can legitimately occupy overlapping regions -- for
example, when the model splits one irregular lesion into two adjacent
detections. Rather than keeping both as duplicates or naively taking their
union, overlapping instances are grouped and collapsed to the INTERSECTION
of their masks: the region every member of the group agrees is lesion.
Instances with no spatial overlap are left completely untouched.
"""
import numpy as np


def _mask_centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def resolve_overlaps(masks: list, scores: list, min_overlap_fraction: float = 0.05):
    """Group spatially overlapping lesion instances and collapse each group to
    the intersection of its members.

    Args:
        masks: list of HxW uint8 {0,1} arrays, one per detected instance.
        scores: matching list of confidence scores.
        min_overlap_fraction: minimum fraction of the SMALLER mask's area that
            must be covered by the other mask for the pair to be grouped.

    Returns:
        final_masks, final_scores: one entry per resolved lesion. A group's
            score is the mean of its members' scores.
        groups: list of lists of original instance indices that formed each
            final lesion (a group of length 1 means that instance had no
            overlap with any other instance).
    """
    n = len(masks)
    if n == 0:
        return [], [], []

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            inter = np.logical_and(masks[i].astype(bool), masks[j].astype(bool)).sum()
            if inter == 0:
                continue
            smaller = min(masks[i].sum(), masks[j].sum())
            if smaller > 0 and inter / smaller >= min_overlap_fraction:
                union(i, j)

    groups_map = {}
    for i in range(n):
        groups_map.setdefault(find(i), []).append(i)
    groups = list(groups_map.values())

    final_masks, final_scores = [], []
    for idxs in groups:
        if len(idxs) == 1:
            final_masks.append(masks[idxs[0]])
            final_scores.append(scores[idxs[0]])
        else:
            inter_mask = masks[idxs[0]].astype(bool).copy()
            for k in idxs[1:]:
                inter_mask = np.logical_and(inter_mask, masks[k].astype(bool))
            final_masks.append(inter_mask.astype(np.uint8))
            final_scores.append(float(np.mean([scores[k] for k in idxs])))

    return final_masks, final_scores, groups
