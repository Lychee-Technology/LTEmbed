#!/usr/bin/env python3
"""Convert a float32 safetensors model to float16.

Usage:
    python3 scripts/convert_to_fp16.py \
        --input  assets/model.safetensors \
        --output assets/model_fp16.safetensors
"""
import argparse
import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description="Convert FP32 safetensors to FP16")
    parser.add_argument("--input", required=True, help="Path to input FP32 .safetensors file")
    parser.add_argument("--output", required=True, help="Path for output FP16 .safetensors file")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    tensors = load_file(args.input)

    fp16 = {k: v.to(dtype=torch.float16) for k, v in tensors.items()}

    print(f"Saving {len(fp16)} tensors → {args.output}")
    save_file(fp16, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
