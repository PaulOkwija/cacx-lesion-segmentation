"""Download the pretrained model checkpoint used for inference.

Usage:
    python scripts/download_weights.py [--output cacx_seg_maskrcnn.pth]
"""
import argparse
import sys

DRIVE_FILE_ID = "15qRFKCiDazpL2cLuD7Bw3SUH-dWI68VS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="cacx_seg_maskrcnn.pth",
                         help="Where to save the checkpoint. Default: cacx_seg_maskrcnn.pth")
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        print("The 'gdown' package is required for this script. Install it with: pip install gdown")
        sys.exit(1)

    gdown.download(id=DRIVE_FILE_ID, output=args.output, quiet=False)


if __name__ == "__main__":
    main()
