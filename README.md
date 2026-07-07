# Cervical Cancer Lesion Instance Segmentation

Instance segmentation of cervical lesions in colposcopy images, using a Mask
R-CNN model fine-tuned for this task. Given an image or a folder of images,
the model locates each visible lesion and produces a separate segmentation
mask and confidence score for it.

This repository contains the inference code and trained weights. It does not
include training code or the original annotated dataset.

## Notice

This tool is a research artifact, released for experimentation, teaching, and
further research. It has not been validated for clinical use, has not
received regulatory clearance of any kind, and must not be used to make
decisions about a patient's diagnosis, treatment, or care. Any use in a
clinical or clinical-adjacent setting requires independent validation
appropriate to that setting.

**Disclaimer.** This model is not intended to diagnose, treat, or otherwise guide the care of any patient. Its output is provided for research purposes only and must not be relied upon as a substitute for professional medical judgment. Any output produced by this model must be independently reviewed and verified by a licensed and experienced healthcare professional before it informs any clinical decision.

## Contents

```
cacx-lesion-segmentation/
├── cacx_seg/                  inference package
│   ├── config.py              all tunable parameters (Config dataclass)
│   ├── model.py                Mask R-CNN construction
│   ├── tta.py                 test-time augmentation (flips + merging)
│   ├── postprocess.py         confidence filtering, small-object removal, hole filling
│   ├── overlap.py             overlap resolution between distinct instances
│   ├── visualize.py           colored overlays and the four-panel figure
│   ├── export.py              COCO-style label export
│   ├── inference.py           the LesionInstanceInference class and segment() entry point
│   └── cli.py                 command line interface
├── scripts/
│   └── download_weights.py    downloads the released checkpoint
├── notebooks/
│   └── inference_demo.ipynb   worked example: single image, folder, label export
├── tests/
│   └── test_pipeline.py       smoke tests (no download required)
├── assets/
│   └── example_output.png     illustrative figure referenced below
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Installation

Requires Python 3.9 or later.

```bash
git clone https://github.com/paulokwija/cacx-lesion-segmentation.git
cd cacx-lesion-segmentation

python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

A CUDA-capable GPU is used automatically if available; otherwise the code
falls back to CPU. CPU inference works but is considerably slower.

## Downloading the model weights

The trained checkpoint is hosted separately from this repository (it is a
binary file, not suited to git). Download it with:

```bash
python scripts/download_weights.py --output cacx_seg_maskrcnn.pth
```

This requires the `gdown` package (included in `requirements.txt`). If you
prefer, download the file manually from the same source and place it
anywhere on disk -- you pass its path explicitly wherever it is needed.

The checkpoint stores the model weights together with two pieces of
metadata, `label_source` and `internal_val_ap50`, which are printed when the
model is loaded so you can confirm which checkpoint you are running.

## Quick start (Python)

```python
from cacx_seg import Config, LesionInstanceInference, segment

inferencer = LesionInstanceInference("cacx_seg_maskrcnn.pth")

# Single image: displays the four-panel view described below
result = segment("path/to/image.jpg", inferencer)
result["raw_masks"], result["raw_scores"]        # every individual detection
result["final_masks"], result["final_scores"]    # after resolving overlapping instances

# Folder of images: writes masks, overlays, and a manifest to disk
manifest_df = segment("path/to/image_folder", inferencer, export_coco=True)
```

`segment()` is the single entry point: pass it either a path to one image or
a path to a folder, and it dispatches accordingly.

## Command line usage

```bash
# Single image
python -m cacx_seg.cli --input image.jpg --checkpoint cacx_seg_maskrcnn.pth

# Folder of images, with COCO label export and per-image visualization
python -m cacx_seg.cli \
    --input images/ \
    --checkpoint cacx_seg_maskrcnn.pth \
    --output-dir results/ \
    --export-coco \
    --save-visualizations
```

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Path to a single image or a folder of images |
| `--checkpoint` | required | Path to the trained checkpoint (`.pth`) |
| `--output-dir` | `outputs` | Where masks, overlays, and manifests are written |
| `--score-thresh` | `0.5` | Minimum confidence to keep a predicted instance |
| `--overlap-min-fraction` | `0.05` | Minimum fractional overlap for two instances to be merged into one final lesion |
| `--no-tta` | off | Disable test-time flip augmentation (faster, slightly less robust) |
| `--no-overlay` | off | Skip saving colored overlay images (folder mode) |
| `--save-visualizations` | off | Also save the four-panel figure for every image (folder mode; slower) |
| `--export-coco` | off | Also write a COCO-format JSON with polygon annotations (folder mode) |
| `--no-show` | off | Single-image mode: save the figure instead of displaying it |
| `--no-backbone-init` | off | Skip downloading COCO-pretrained backbone weights before loading the checkpoint; only affects startup, not the final result |

## Understanding the output

Every image is run through the model once at its original orientation and
twice more with horizontal and vertical flips (test-time augmentation).
Detections that correspond to the same lesion across these three views are
merged, and any surviving instance below the confidence threshold, or
smaller than a minimum pixel size, is discarded. What remains after this
stage is the set of **individual detections**.

Individual detections can still overlap each other in the image -- this
happens when the model represents one irregular lesion as two adjacent
instances. Detections that overlap by at least `overlap_min_fraction` of the
smaller one's area are grouped and collapsed to their **intersection**: the
region every member of the group agrees is lesion. Detections with no
overlap are left exactly as they are. This second, smaller set is the
**final lesions** -- the result you should generally use.

Calling `segment()` on a single image renders a four-panel figure:

1. **Input** -- the original image.
2. **Individual detections** -- every detection from the step above, each
   numbered and drawn in its own color.
3. **Final lesions** -- after overlap resolution. A merged group is labeled
   with the numbers of the detections that formed it, for example `1+2`.
4. **Confidence key** -- a legend mapping each numbered detection's color to
   its confidence score, plus the combined score for any merged group.

The figure below is a synthetic illustration (plain colored shapes, not a
real image) included only to show the layout:

![Illustrative example of the four-panel output](assets/example_output.png)

## Exporting labels

Running `segment()` or the CLI on a folder writes the following to the
output directory:

- **`instance_masks/`** -- one binary PNG per individual detection
  (`<image_stem>_inst<k>.png`, values 0 or 255).
- **`final_masks/`** -- one binary PNG per final, overlap-resolved lesion
  (`<image_stem>_final<k>.png`).
- **`overlays/`** -- the input image with final lesions drawn in color, one
  file per image (skipped with `--no-overlay`).
- **`full_visualizations/`** -- the four-panel figure per image, only if
  `--save-visualizations` / `save_full_visualization=True` was requested.
- **`inference_manifest.csv`** -- one row per image, with columns:
  `image`, `image_path`, `n_lesions_raw`, `n_lesions_final`, `raw_scores`,
  `final_scores`, `mean_confidence`, `instance_mask_paths`,
  `final_mask_paths`, `overlay_path`, `visualization_path`.
- **`coco_annotations.json`** -- written only with `--export-coco` /
  `export_coco=True`. A COCO-format dataset (`images`, `annotations`,
  `categories`) describing the final lesions, with polygon segmentation
  (not RLE), so it can be read with the standard `json` module and does not
  require `pycocotools`. Each annotation also carries a `score` field with
  the detection's confidence. This file can be imported into common
  annotation tools such as CVAT or Label Studio, or used directly to build a
  training set for further model development.

## Configuration reference

All parameters live in `cacx_seg.config.Config`. Pass an instance to
`LesionInstanceInference` to override any of them.

| Field | Default | Description |
|---|---|---|
| `checkpoint_path` | `"cacx_seg_maskrcnn.pth"` | Path to the trained checkpoint |
| `output_dir` | `"outputs"` | Default output directory |
| `image_size` | `640` | Minimum internal resize dimension |
| `max_size` | `960` | Maximum internal resize dimension |
| `pretrained` | `True` | Initialize the backbone from COCO weights before loading the checkpoint (irrelevant to the final result; only affects cold-start time and internet requirement) |
| `box_nms_thresh` | `0.5` | Box-level non-maximum suppression threshold |
| `box_detections_per_img` | `30` | Maximum instances kept per image |
| `score_thresh` | `0.5` | Minimum confidence to keep a predicted instance |
| `mask_binarize_thresh` | `0.5` | Threshold used to binarize the soft predicted mask |
| `min_object_size` | `40` | Minimum instance area in pixels; smaller instances are discarded as noise |
| `tta_flip` | `True` | Run test-time augmentation using horizontal and vertical flips |
| `tta_merge_iou_thresh` | `0.4` | IoU threshold used to merge detections across TTA views |
| `overlap_min_fraction` | `0.05` | Minimum fractional overlap for two distinct instances to be merged into one final lesion |

## Model details

- Architecture: Mask R-CNN with a ResNet-50-FPN-v2 backbone
  (`torchvision.models.detection.maskrcnn_resnet50_fpn_v2`), fine-tuned for
  two classes: background and lesion.
- Test-time augmentation: the model is run on the image and on its
  horizontal and vertical flips; detections referring to the same lesion are
  merged and their scores averaged.
- Postprocessing: instances below the confidence threshold are dropped,
  small connected components are removed, and remaining holes in each mask
  are filled.
- Overlap resolution: distinct instances that spatially overlap are
  collapsed to their intersection, as described above.

## Running the tests

The test suite does not require the released checkpoint; it builds an
untrained model of the same architecture to check that the pipeline runs
end to end, and checks the overlap-resolution logic directly.

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

Released under the MIT License. See `LICENSE`.

## Citation

If you use this model or code in your work, please reference this
repository.
