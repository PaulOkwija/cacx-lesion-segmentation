"""Mask R-CNN model construction.

Two output classes: background and lesion. The backbone is a
ResNet-50-FPN-v2, matching the architecture the released checkpoint was
trained with.
"""
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .config import Config

NUM_CLASSES = 2  # background, lesion


def build_model(cfg: Config, device):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if cfg.pretrained else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=cfg.image_size,
        max_size=cfg.max_size,
        box_nms_thresh=cfg.box_nms_thresh,
        box_detections_per_img=cfg.box_detections_per_img,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)

    return model.to(device)
