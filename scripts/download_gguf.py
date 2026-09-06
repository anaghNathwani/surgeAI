#!/usr/bin/env python3
"""
Download a quantized GGUF version of Phi-4-mini-instruct.

Usage:
    python scripts/download_gguf.py               # downloads Q4_K_M (recommended)
    python scripts/download_gguf.py --quant Q5_K_M
    python scripts/download_gguf.py --quant Q8_0  # highest quality, largest file

Quantisation guide:
    Q4_K_M  ~2.5 GB  good quality, fast          ← default
    Q5_K_M  ~3.1 GB  better quality
    Q6_K    ~3.6 GB  near-lossless
    Q8_0    ~4.7 GB  almost identical to full precision
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = ROOT / "weights"
REPO = "unsloth/Phi-4-mini-instruct-GGUF"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Phi-4-mini GGUF")
    parser.add_argument(
        "--quant", default="Q4_K_M",
        choices=["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        help="Quantisation level (default: Q4_K_M)",
    )
    parser.add_argument(
        "--out-dir", default=str(WEIGHTS_DIR),
        help="Directory to save the file (default: weights/)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("  pip install huggingface_hub")
        sys.exit(1)

    filename = f"Phi-4-mini-instruct-{args.quant}.gguf"
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / filename

    if dest.exists():
        print(f"Already downloaded: {dest}")
        print(f"\nRun the GUI with:\n  python GUI/app.py --model {dest}")
        return

    print(f"Downloading {filename} from {REPO} …")
    path = hf_hub_download(
        repo_id=REPO,
        filename=filename,
        local_dir=str(out_dir),
    )
    print(f"\nSaved to: {path}")
    print(f"\nRun the GUI with:\n  python GUI/app.py --model {path}")


if __name__ == "__main__":
    main()
