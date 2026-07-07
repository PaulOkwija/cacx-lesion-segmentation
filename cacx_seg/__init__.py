"""Cervical cancer lesion instance segmentation: inference package.

See README.md for usage. Quick reference:

    from cacx_seg import Config, LesionInstanceInference, segment

    inferencer = LesionInstanceInference("cacx_seg_maskrcnn.pth")
    result = segment("path/to/image_or_folder", inferencer)
"""
from .config import Config
from .export import build_coco_dataset, save_coco_json
from .inference import LesionInstanceInference, segment
from .overlap import resolve_overlaps
from .visualize import colorize_instances, visualize_lesions

__version__ = "1.0.0"

__all__ = [
    "Config",
    "LesionInstanceInference",
    "segment",
    "resolve_overlaps",
    "visualize_lesions",
    "colorize_instances",
    "save_coco_json",
    "build_coco_dataset",
]
