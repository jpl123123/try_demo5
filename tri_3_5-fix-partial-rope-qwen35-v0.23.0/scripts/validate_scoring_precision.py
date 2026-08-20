#!/usr/bin/env python3
"""Validate TriAttention PyTorch scoring precision across devices.

This is a small standalone check for the vLLM-Ascend path. It compares the
PyTorch/torch_npu scoring backend against a CPU fp32 baseline using the same
quantized key tensor and statistics.
"""

from __future__ import annotations

import argparse
import sys

import torch

from triattention.vllm.core.config import TriAttentionConfig
from triattention.vllm.core.scoring import compute_scores_pytorch


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _resolve_device(name: str) -> torch.device:
    normalized = name.strip().lower()
    if normalized == "auto":
        if hasattr(torch, "npu"):
            try:
                if torch.npu.is_available():
                    return torch.device("npu")
            except Exception:
                pass
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if normalized == "npu":
        try:
            import torch_npu  # noqa: F401
        except Exception as exc:
            raise RuntimeError("torch_npu is required for --device npu") from exc
    return torch.device(normalized)


def _build_inputs(args: argparse.Namespace, tensor_dtype: torch.dtype):
    torch.manual_seed(args.seed)
    freq_count = args.head_dim // 2
    key_fp32 = torch.randn(
        1,
        args.heads,
        args.seq_len,
        args.head_dim,
        dtype=torch.float32,
    )
    key_quant = key_fp32.to(dtype=tensor_dtype).to(dtype=torch.float32)

    q_mean_complex = torch.randn(
        args.heads,
        freq_count,
        2,
        dtype=torch.float32,
    ) * 0.05
    q_abs_mean = q_mean_complex.norm(dim=-1) + torch.rand(
        args.heads,
        freq_count,
        dtype=torch.float32,
    ) * 0.02
    freq_scale_sq = torch.ones(args.heads, freq_count, dtype=torch.float32)
    omega = 1.0 / (
        args.rope_theta
        ** (torch.arange(0, args.head_dim, 2, dtype=torch.float32) / args.head_dim)
    )
    offsets = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.float32)
    head_stats = {
        "q_mean_complex": q_mean_complex,
        "q_abs_mean": q_abs_mean,
    }
    return key_quant, head_stats, omega, offsets, freq_scale_sq


def _score(
    *,
    key_states: torch.Tensor,
    head_stats: dict[str, torch.Tensor],
    omega: torch.Tensor,
    offsets: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    device: torch.device,
    config: TriAttentionConfig,
    round_start: int,
) -> torch.Tensor:
    key_on_device = key_states.to(device=device, dtype=key_states.dtype)
    stats_on_device = {
        name: tensor.to(device=device)
        for name, tensor in head_stats.items()
    }
    scores = compute_scores_pytorch(
        key_states=key_on_device,
        cache_positions=None,
        head_stats=stats_on_device,
        omega=omega.to(device=device),
        offsets=offsets.to(device=device),
        freq_scale_sq=freq_scale_sq.to(device=device),
        config=config,
        round_start=round_start,
    )
    return scores.detach().to(device="cpu", dtype=torch.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--round-start", type=int, default=8192)
    parser.add_argument("--rope-theta", type=float, default=1000000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-abs-threshold", type=float, default=5e-3)
    parser.add_argument("--topk-overlap-threshold", type=float, default=0.99)
    args = parser.parse_args()

    if args.head_dim % 2 != 0:
        raise ValueError("--head-dim must be even")
    tensor_dtype = _dtype_from_name(args.dtype)
    target_device = _resolve_device(args.device)
    key, head_stats, omega, offsets, freq_scale_sq = _build_inputs(args, tensor_dtype)

    cpu_cfg = TriAttentionConfig(
        kv_budget=max(args.topk, 1),
        divide_length=128,
        use_triton_scoring=False,
        compute_dtype=torch.float32,
        topk_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    target_cfg = TriAttentionConfig(
        kv_budget=max(args.topk, 1),
        divide_length=128,
        use_triton_scoring=False,
        compute_dtype=torch.float32,
        topk_dtype=torch.float32,
        device=target_device,
    )

    baseline = _score(
        key_states=key,
        head_stats=head_stats,
        omega=omega,
        offsets=offsets,
        freq_scale_sq=freq_scale_sq,
        device=torch.device("cpu"),
        config=cpu_cfg,
        round_start=args.round_start,
    )
    candidate = _score(
        key_states=key.to(dtype=tensor_dtype),
        head_stats=head_stats,
        omega=omega,
        offsets=offsets,
        freq_scale_sq=freq_scale_sq,
        device=target_device,
        config=target_cfg,
        round_start=args.round_start,
    )

    diff = (candidate - baseline).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    k = min(args.topk, args.seq_len)
    base_topk = torch.topk(baseline[0], k=k, dim=-1).indices
    cand_topk = torch.topk(candidate[0], k=k, dim=-1).indices
    overlaps = []
    for head_idx in range(args.heads):
        base_set = set(int(x) for x in base_topk[head_idx].tolist())
        cand_set = set(int(x) for x in cand_topk[head_idx].tolist())
        overlaps.append(len(base_set & cand_set) / float(k))
    min_overlap = min(overlaps)
    mean_overlap = sum(overlaps) / len(overlaps)

    print(f"device={target_device} dtype={tensor_dtype}")
    print(f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")
    print(f"topk_overlap_min={min_overlap:.6g} topk_overlap_mean={mean_overlap:.6g}")

    if max_abs > args.max_abs_threshold:
        print(
            f"FAIL: max_abs {max_abs:.6g} > threshold {args.max_abs_threshold:.6g}",
            file=sys.stderr,
        )
        return 1
    if min_overlap < args.topk_overlap_threshold:
        print(
            "FAIL: topk_overlap_min "
            f"{min_overlap:.6g} < threshold {args.topk_overlap_threshold:.6g}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
