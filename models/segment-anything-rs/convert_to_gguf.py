#!/usr/bin/env python3
"""Convert SAM safetensors to GGUF format for fusor-ml.

Handles both:
  - sam_vit_b_01ec64.safetensors (ViT-B SAM)
  - mobile_sam-tiny-vitt.safetensors (MobileSAM / TinyViT)

For TinyViT, BatchNorm layers are fused into the preceding Conv2d weights
at conversion time, so the runtime model only needs Conv2d (no BN).

Usage:
    pip install safetensors gguf numpy
    python convert_to_gguf.py mobile_sam-tiny-vitt.safetensors --out mobile_sam-tiny-vitt.gguf
    python convert_to_gguf.py sam_vit_b_01ec64.safetensors --out sam_vit_b_01ec64.gguf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from gguf import GGUFWriter


def fuse_bn_into_conv(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fuse BatchNorm parameters into Conv2d weight and produce a bias.

    conv_weight: (out_ch, in_ch/groups, kH, kW)
    bn_weight:   (out_ch,) aka gamma
    bn_bias:     (out_ch,) aka beta
    bn_mean:     (out_ch,)
    bn_var:      (out_ch,)

    Returns: (fused_weight, fused_bias)
    """
    out_ch = conv_weight.shape[0]
    # scale = gamma / sqrt(var + eps)
    scale = bn_weight / np.sqrt(bn_var + eps)
    # Reshape scale for broadcasting with conv weight dims
    scale_shape = [out_ch] + [1] * (conv_weight.ndim - 1)
    fused_weight = conv_weight * scale.reshape(scale_shape)
    fused_bias = bn_bias - bn_mean * scale
    return fused_weight, fused_bias


def find_bn_groups(tensors):
    """Find all Conv2dBN groups (prefixes that have both .c.weight and .bn.weight)."""
    bn_prefixes = set()
    for key in tensors:
        if key.endswith(".bn.weight"):
            prefix = key[: -len(".bn.weight")]
            # Check that corresponding conv weight exists
            if f"{prefix}.c.weight" in tensors:
                bn_prefixes.add(prefix)
    return sorted(bn_prefixes)


def convert(input_path, output_path, is_tiny=None):
    """Convert a SAM safetensors file to GGUF."""
    print(f"Loading {input_path}...")
    model = safe_open(str(input_path), framework="numpy")
    keys = list(model.keys())

    # Auto-detect model variant
    if is_tiny is None:
        is_tiny = any("patch_embed.seq" in k for k in keys)
    arch = "sam_tiny" if is_tiny else "sam"
    print(f"Detected architecture: {arch}")

    # Find BatchNorm groups to fuse
    tensors = {k: model.get_tensor(k) for k in keys}
    bn_groups = find_bn_groups(tensors)
    if bn_groups:
        print(f"Found {len(bn_groups)} Conv2dBN groups to fuse")

    # Track which keys have been consumed by BN fusion
    consumed = set()
    fused = {}

    for prefix in bn_groups:
        conv_w = tensors[f"{prefix}.c.weight"]
        bn_w = tensors[f"{prefix}.bn.weight"]
        bn_b = tensors[f"{prefix}.bn.bias"]
        bn_mean = tensors[f"{prefix}.bn.running_mean"]
        bn_var = tensors[f"{prefix}.bn.running_var"]

        fused_weight, fused_bias = fuse_bn_into_conv(conv_w, bn_w, bn_b, bn_mean, bn_var)
        fused[f"{prefix}.c.weight"] = fused_weight
        fused[f"{prefix}.c.bias"] = fused_bias

        consumed.add(f"{prefix}.c.weight")
        consumed.add(f"{prefix}.bn.weight")
        consumed.add(f"{prefix}.bn.bias")
        consumed.add(f"{prefix}.bn.running_mean")
        consumed.add(f"{prefix}.bn.running_var")
        # num_batches_tracked is not needed
        if f"{prefix}.bn.num_batches_tracked" in tensors:
            consumed.add(f"{prefix}.bn.num_batches_tracked")

    # Build final tensor dict
    output_tensors = {}

    # Add fused tensors
    for k, v in fused.items():
        output_tensors[k] = v

    # Add remaining tensors (skip consumed ones and num_batches_tracked)
    for k, v in tensors.items():
        if k in consumed:
            continue
        if k.endswith(".num_batches_tracked"):
            continue
        output_tensors[k] = v

    # Write GGUF
    print(f"Writing {output_path} with {len(output_tensors)} tensors...")
    writer = GGUFWriter(str(output_path), arch)

    # Add metadata
    writer.add_name(f"SAM {'TinyViT' if is_tiny else 'ViT-B'}")

    # Sort keys for deterministic output
    for key in sorted(output_tensors.keys()):
        tensor = output_tensors[key]
        tensor = tensor.astype(np.float32)
        writer.add_tensor(key, tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Done! Wrote {output_path}")
    print(f"  Architecture: {arch}")
    print(f"  Tensors: {len(output_tensors)}")
    if bn_groups:
        print(f"  Fused BN groups: {len(bn_groups)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SAM safetensors to GGUF for fusor-ml"
    )
    parser.add_argument("input", type=Path, help="Input safetensors file")
    parser.add_argument("--out", type=Path, required=True, help="Output GGUF file")
    parser.add_argument(
        "--tiny",
        action="store_true",
        default=None,
        help="Force TinyViT (MobileSAM) variant detection",
    )
    parser.add_argument(
        "--vit-b",
        action="store_true",
        default=None,
        help="Force ViT-B variant detection",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    is_tiny = None
    if args.tiny:
        is_tiny = True
    elif args.vit_b:
        is_tiny = False

    convert(args.input, args.out, is_tiny)


if __name__ == "__main__":
    main()
