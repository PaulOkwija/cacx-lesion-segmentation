"""High-level inference API: load the trained checkpoint once, then run it on
a single image or an entire folder.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import Config
from .device import get_device
from .export import save_coco_json
from .model import build_model
from .overlap import resolve_overlaps
from .postprocess import postprocess_instances
from .tta import predict_tta_instances
from .visualize import colorize_instances, visualize_lesions

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class LesionInstanceInference:
    """Load the trained Mask R-CNN once, then call `.predict(image_path)` for
    a single image or `.run_on_folder(input_dir, output_dir)` for a batch."""

    def __init__(self, checkpoint_path: str, cfg: Config = None, device=None):
        self.cfg = cfg or Config(checkpoint_path=checkpoint_path)
        self.device = device or get_device()
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model = build_model(self.cfg, self.device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.label_source = ckpt.get("label_source", "unknown") if isinstance(ckpt, dict) else "unknown"
        self.internal_val_ap50 = (
            ckpt.get("internal_val_ap50", float("nan")) if isinstance(ckpt, dict) else float("nan")
        )
        print(
            f"Loaded checkpoint (label_source='{self.label_source}', "
            f"internal_val_ap50={self.internal_val_ap50:.4f})"
        )

    def predict(self, image_path: str, cfg_override: Config = None):
        """Run instance segmentation on a single image.

        Returns (masks, scores): masks is a list of HxW uint8 {0,1} arrays,
        scores their corresponding confidences (0-1). These are the raw
        individual detections; overlapping instances are NOT resolved here --
        see `resolve_overlaps` or `predict_and_visualize`.
        """
        cfg = cfg_override or self.cfg
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        masks, scores = predict_tta_instances(self.model, img_rgb, cfg, self.device)
        masks, scores = postprocess_instances(masks, scores, cfg)
        return masks, scores

    def predict_and_visualize(self, image_path, cfg_override: Config = None, figsize=(22, 5.5),
                               show: bool = True, save_path: str = None):
        """Run inference and render the four-panel view (input, individual
        detections, final overlap-resolved lesions, confidence key).

        Returns a dict with:
            raw_masks, raw_scores       -- every individual detection
            final_masks, final_scores   -- after resolving overlapping instances
            groups                     -- which raw instance indices (1-based
                                           in the plot) formed each final lesion
        """
        cfg = cfg_override or self.cfg
        img_rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        masks, scores = self.predict(image_path, cfg_override=cfg)
        final_masks, final_scores, groups = visualize_lesions(
            img_rgb, masks, scores,
            min_overlap_fraction=cfg.overlap_min_fraction,
            figsize=figsize, show=show, save_path=save_path,
        )
        return {
            "raw_masks": masks, "raw_scores": scores,
            "final_masks": final_masks, "final_scores": final_scores,
            "groups": groups,
        }

    def run_on_folder(self, input_dir: str, output_dir: str = None, save_overlay: bool = True,
                       save_full_visualization: bool = False, export_coco: bool = False):
        """Run inference over every image in `input_dir`, resolve overlapping
        instances, and save masks (plus optional overlays, four-panel
        visualizations, and COCO annotations) to `output_dir`.

        Returns the manifest DataFrame (also saved as inference_manifest.csv).
        """
        cfg = self.cfg
        output_dir = output_dir or cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)
        mask_dir = os.path.join(output_dir, "instance_masks")
        final_mask_dir = os.path.join(output_dir, "final_masks")
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(final_mask_dir, exist_ok=True)
        overlay_dir = os.path.join(output_dir, "overlays")
        if save_overlay:
            os.makedirs(overlay_dir, exist_ok=True)
        viz_dir = os.path.join(output_dir, "full_visualizations")
        if save_full_visualization:
            os.makedirs(viz_dir, exist_ok=True)

        files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(IMAGE_EXTENSIONS)]
        if not files:
            print(f"No images found in {input_dir} (looked for extensions {IMAGE_EXTENSIONS})")

        manifest = []
        coco_records = []
        for f in tqdm(files, desc="inference"):
            path = os.path.join(input_dir, f)
            img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            masks, scores = self.predict(path)
            final_masks, final_scores, groups = resolve_overlaps(masks, scores, cfg.overlap_min_fraction)
            stem = Path(f).stem

            inst_paths = []
            for k, m in enumerate(masks):
                p = os.path.join(mask_dir, f"{stem}_inst{k}.png")
                cv2.imwrite(p, m * 255)
                inst_paths.append(p)

            final_paths = []
            for k, m in enumerate(final_masks):
                p = os.path.join(final_mask_dir, f"{stem}_final{k}.png")
                cv2.imwrite(p, m * 255)
                final_paths.append(p)

            overlay_path = ""
            if save_overlay:
                overlay = colorize_instances(img_rgb, final_masks) if final_masks else img_rgb
                overlay_path = os.path.join(overlay_dir, f"{stem}_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            viz_path = ""
            if save_full_visualization:
                viz_path = os.path.join(viz_dir, f"{stem}_visualization.png")
                visualize_lesions(img_rgb, masks, scores, min_overlap_fraction=cfg.overlap_min_fraction,
                                   show=False, save_path=viz_path)

            if export_coco:
                h, w = img_rgb.shape[:2]
                coco_records.append({
                    "file_name": f, "width": w, "height": h,
                    "masks": final_masks, "scores": final_scores,
                })

            manifest.append({
                "image": f,
                "image_path": path,
                "n_lesions_raw": len(masks),
                "n_lesions_final": len(final_masks),
                "raw_scores": ";".join(f"{s:.3f}" for s in scores),
                "final_scores": ";".join(f"{s:.3f}" for s in final_scores),
                "mean_confidence": float(np.mean(final_scores)) if final_scores else 0.0,
                "instance_mask_paths": ";".join(inst_paths),
                "final_mask_paths": ";".join(final_paths),
                "overlay_path": overlay_path,
                "visualization_path": viz_path,
            })

        manifest_df = pd.DataFrame(manifest)
        manifest_csv = os.path.join(output_dir, "inference_manifest.csv")
        manifest_df.to_csv(manifest_csv, index=False)

        if export_coco:
            coco_path = os.path.join(output_dir, "coco_annotations.json")
            save_coco_json(coco_records, coco_path)
            print(f"COCO annotations saved to {coco_path}")

        n_with_lesions = int((manifest_df.n_lesions_final > 0).sum()) if len(manifest_df) else 0
        print(
            f"{n_with_lesions}/{len(manifest_df)} images have at least one final detected lesion; "
            f"manifest saved to {manifest_csv}"
        )
        return manifest_df


def segment(input_path: str, inferencer: "LesionInstanceInference", output_dir: str = None,
            save_overlay: bool = True, visualize: bool = True, export_coco: bool = False):
    """Single entry point: pass either the path to one image or the path to a
    folder of images and get back cervical lesion segmentation masks.

    - Single image: shows the four-panel view if visualize=True, returns a
      dict with raw and final (overlap-resolved) masks/scores.
    - Folder: returns a manifest DataFrame and writes masks/overlays (plus
      optional COCO annotations) to output_dir.
    """
    input_path = str(input_path)
    if os.path.isdir(input_path):
        return inferencer.run_on_folder(
            input_path, output_dir=output_dir, save_overlay=save_overlay, export_coco=export_coco
        )
    elif os.path.isfile(input_path):
        if visualize:
            return inferencer.predict_and_visualize(input_path)
        masks, scores = inferencer.predict(input_path)
        final_masks, final_scores, groups = resolve_overlaps(masks, scores, inferencer.cfg.overlap_min_fraction)
        return {
            "raw_masks": masks, "raw_scores": scores,
            "final_masks": final_masks, "final_scores": final_scores, "groups": groups,
        }
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
