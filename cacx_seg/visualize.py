"""Visualization: colored overlays and the four-panel summary figure."""
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from .overlap import _mask_centroid, resolve_overlaps


def _get_cmap(name="tab10"):
    # matplotlib >=3.7 deprecated cm.get_cmap in favor of plt.colormaps[...]
    try:
        return plt.colormaps[name]
    except Exception:
        return cm.get_cmap(name)


def colorize_instances(img_rgb, masks, alpha=0.45):
    """Overlay each instance mask on the image in a distinct color."""
    overlay = img_rgb.copy()
    cmap = _get_cmap("tab10")
    for i, m in enumerate(masks):
        color = np.array(cmap(i % 10)[:3]) * 255
        region = m.astype(bool)
        overlay[region] = (1 - alpha) * overlay[region] + alpha * color
    return overlay.astype(np.uint8)


def visualize_lesions(img_rgb, masks, scores, min_overlap_fraction=0.05,
                       figsize=(22, 5.5), show=True, save_path=None):
    """Render the four-panel view:

        [1] input image
        [2] every individual detection, numbered and colored
        [3] final lesions after overlap resolution (overlapping instances
            collapsed to their intersection; non-overlapping ones untouched)
        [4] confidence key: which color/number maps to which score

    Returns (final_masks, final_scores, groups) -- see overlap.resolve_overlaps.
    """
    cmap = _get_cmap("tab10")
    colors = [np.array(cmap(i % 10)[:3]) for i in range(max(len(masks), 1))]

    overlay_all = img_rgb.copy()
    for i, m in enumerate(masks):
        region = m.astype(bool)
        overlay_all[region] = (0.55 * overlay_all[region] + 0.45 * (colors[i] * 255))

    final_masks, final_scores, groups = resolve_overlaps(masks, scores, min_overlap_fraction)

    overlay_final = img_rgb.copy()
    group_colors = [colors[idxs[0]] for idxs in groups]
    for m, c in zip(final_masks, group_colors):
        region = m.astype(bool)
        overlay_final[region] = (0.55 * overlay_final[region] + 0.45 * (c * 255))

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(img_rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(overlay_all.astype(np.uint8))
    axes[1].set_title(f"{len(masks)} individual detection(s)")
    axes[1].axis("off")
    for i, m in enumerate(masks):
        c = _mask_centroid(m)
        if c is not None:
            axes[1].text(c[0], c[1], str(i + 1), color="white", fontsize=11, fontweight="bold",
                         ha="center", va="center",
                         bbox=dict(boxstyle="circle,pad=0.25", facecolor=colors[i], edgecolor="none", alpha=0.9))

    axes[2].imshow(overlay_final.astype(np.uint8))
    n_merged_groups = sum(1 for idxs in groups if len(idxs) > 1)
    title = f"{len(final_masks)} final lesion(s)"
    if n_merged_groups:
        title += f" ({n_merged_groups} from overlap intersection)"
    axes[2].set_title(title)
    axes[2].axis("off")
    for k, (m, idxs) in enumerate(zip(final_masks, groups)):
        c = _mask_centroid(m)
        if c is not None:
            label = "+".join(str(i + 1) for i in idxs) if len(idxs) > 1 else str(idxs[0] + 1)
            axes[2].text(c[0], c[1], label, color="white", fontsize=11, fontweight="bold",
                         ha="center", va="center",
                         bbox=dict(boxstyle="circle,pad=0.25", facecolor=group_colors[k], edgecolor="none", alpha=0.9))

    axes[3].axis("off")
    axes[3].set_title("Confidence key")
    y = 0.95
    for i, s in enumerate(scores):
        axes[3].add_patch(plt.Rectangle((0.02, y - 0.03), 0.06, 0.05, transform=axes[3].transAxes,
                                         facecolor=colors[i], edgecolor="none"))
        axes[3].text(0.12, y, f"Lesion {i + 1}: {s:.3f}", transform=axes[3].transAxes,
                     fontsize=10, va="center")
        y -= 0.09

    if groups and any(len(idxs) > 1 for idxs in groups):
        y -= 0.05
        axes[3].text(0.0, y, "Final (overlaps intersected):", transform=axes[3].transAxes,
                     fontsize=10, fontweight="bold", va="center")
        y -= 0.09
        for k, (idxs, s) in enumerate(zip(groups, final_scores)):
            if len(idxs) == 1:
                continue
            axes[3].add_patch(plt.Rectangle((0.02, y - 0.03), 0.06, 0.05, transform=axes[3].transAxes,
                                             facecolor=group_colors[k], edgecolor="none"))
            label = "+".join(str(i + 1) for i in idxs)
            axes[3].text(0.12, y, f"Lesions {label} \u2192 {s:.3f}", transform=axes[3].transAxes,
                         fontsize=10, va="center")
            y -= 0.09

    if not scores:
        axes[3].text(0.0, 0.95, "No lesions detected", transform=axes[3].transAxes, fontsize=10)

    plt.tight_layout(pad=1.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return final_masks, final_scores, groups
