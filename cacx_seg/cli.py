"""Command line interface for running inference.

Examples:
    python -m cacx_seg.cli --input image.jpg --checkpoint cacx_seg_maskrcnn.pth

    python -m cacx_seg.cli --input images/ --checkpoint cacx_seg_maskrcnn.pth \\
        --output-dir results/ --export-coco --save-visualizations
"""
import argparse
import os

from .config import Config
from .inference import LesionInstanceInference


def build_parser():
    p = argparse.ArgumentParser(
        description="Cervical cancer lesion instance segmentation inference."
    )
    p.add_argument("--input", required=True,
                   help="Path to a single image or a folder of images.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to the trained model checkpoint (.pth).")
    p.add_argument("--output-dir", default="outputs",
                   help="Directory to write masks, overlays, and manifests to. Default: outputs")
    p.add_argument("--score-thresh", type=float, default=0.5,
                   help="Minimum confidence to keep a predicted instance. Default: 0.5")
    p.add_argument("--overlap-min-fraction", type=float, default=0.05,
                   help="Minimum fractional overlap for two instances to be treated as the "
                        "same lesion and collapsed to their intersection. Default: 0.05")
    p.add_argument("--no-tta", action="store_true",
                   help="Disable test-time flip augmentation (faster, slightly less robust).")
    p.add_argument("--no-overlay", action="store_true",
                   help="Skip saving colored overlay images for folder input.")
    p.add_argument("--save-visualizations", action="store_true",
                   help="Also save the four-panel visualization PNG for every image in "
                        "folder mode (slower for large folders).")
    p.add_argument("--export-coco", action="store_true",
                   help="Also export a COCO-format JSON with polygon annotations for folder input.")
    p.add_argument("--no-show", action="store_true",
                   help="For single-image input, save the visualization instead of displaying it.")
    p.add_argument("--no-backbone-init", action="store_true",
                   help="Skip downloading COCO-pretrained backbone weights before loading the "
                        "checkpoint. The checkpoint's weights are loaded either way, so this only "
                        "matters if you have no internet access or want a faster cold start.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    cfg = Config(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        score_thresh=args.score_thresh,
        overlap_min_fraction=args.overlap_min_fraction,
        tta_flip=not args.no_tta,
        pretrained=not args.no_backbone_init,
    )

    inferencer = LesionInstanceInference(args.checkpoint, cfg)

    if os.path.isdir(args.input):
        inferencer.run_on_folder(
            args.input,
            output_dir=args.output_dir,
            save_overlay=not args.no_overlay,
            save_full_visualization=args.save_visualizations,
            export_coco=args.export_coco,
        )
    elif os.path.isfile(args.input):
        os.makedirs(args.output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.input))[0]
        save_path = os.path.join(args.output_dir, f"{stem}_visualization.png")
        result = inferencer.predict_and_visualize(
            args.input, show=not args.no_show, save_path=save_path,
        )
        print(
            f"Detected {len(result['final_masks'])} final lesion(s). "
            f"Visualization saved to {save_path}"
        )
    else:
        raise FileNotFoundError(f"Input path does not exist: {args.input}")


if __name__ == "__main__":
    main()
