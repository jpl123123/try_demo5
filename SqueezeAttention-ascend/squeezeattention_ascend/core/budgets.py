"""SqueezeAttention layer-wise budget learning.

Port of the SqueezeAttention mechanism (utils_hh/modify_llama_drop.py):

1. during the request's prefill, record per-layer mean cosine similarity
   between the layer input and the post-self-attention output (the paper's
   ``hidd_data``);
2. cluster the per-layer means with KMeans (3 classes by default);
3. assign per-layer KV budgets preserving the total budget invariant
   ``num_layers * ini_size``: class1/class2 layers get ``a * prompt_len``,
   class3 (highest importance) layers get ``percent * prompt_len`` where
   ``a = (num_layers * ini_size - n3 * percent) / (n1 + n2)``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def kmeans_labels(values: list[float], n_clusters: int, seed: Optional[int]) -> list[int]:
    """Cluster per-layer importance values with sklearn (or a torch fallback).

    Returns the cluster label per value (0..n_clusters-1). Label order is
    arbitrary; callers sort by cluster center to recover the paper's
    class1/class2/class3 ordering.
    """
    if not values:
        return []
    try:
        import numpy as np
        from sklearn.cluster import KMeans

        data = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(data)
        return [int(x) for x in labels]
    except Exception:
        return _torch_kmeans_labels(values, n_clusters, seed)


def _torch_kmeans_labels(values: list[float], n_clusters: int, seed: Optional[int]) -> list[int]:
    x = torch.as_tensor(values, dtype=torch.float64)
    if x.numel() < n_clusters:
        return [0] * int(x.numel())
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    perm = torch.randperm(int(x.numel()), generator=generator)
    centers = x[perm[:n_clusters]].clone()
    labels = torch.zeros(int(x.numel()), dtype=torch.long)
    for _ in range(100):
        dists = (x.unsqueeze(1) - centers.unsqueeze(0)).abs()
        new_labels = dists.argmin(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(n_clusters):
            members = x[labels == c]
            if members.numel() > 0:
                centers[c] = members.mean()
    return [int(v) for v in labels.tolist()]


def compute_layer_budgets(
    *,
    layer_importance: list[float],
    num_layers: int,
    ini_size: float,
    class3_size: float,
    prompt_len: int,
    n_clusters: int = 3,
    seed: Optional[int] = None,
) -> tuple[list[int], dict[str, Any]]:
    """Compute per-layer KV budgets (tokens) and cluster diagnostics.

    Mirrors SqueezeAttention's budget math exactly:
    ``a = (num_layers * ini_size - len(class3) * percent) / (len(class1) +
    len(class2))``; class1/class2 windows = ``int(a * prompt_len)``; class3
    windows = ``int(percent * prompt_len)``.
    """
    num_layers = max(1, int(num_layers))
    prompt_len = max(1, int(prompt_len))
    if len(layer_importance) < num_layers:
        layer_importance = list(layer_importance) + [0.0] * (num_layers - len(layer_importance))
    layer_importance = layer_importance[:num_layers]

    n_clusters = min(max(2, int(n_clusters)), num_layers)
    labels = kmeans_labels(layer_importance, n_clusters, seed)
    centers = _cluster_centers(layer_importance, labels, n_clusters)
    # Sort classes by ascending center: class1 = lowest importance (smallest
    # budget share), class3 = highest importance (percent share).
    order = sorted(range(n_clusters), key=lambda c: centers[c])
    class_of = {cluster: rank for rank, cluster in enumerate(order)}
    class_ids = [class_of[label] for label in labels]

    class_sizes = {c: class_ids.count(c) for c in range(n_clusters)}
    if n_clusters == 3:
        n3 = class_sizes.get(2, 0)
        n12 = class_sizes.get(0, 0) + class_sizes.get(1, 0)
        percent = max(0.0, min(1.0, float(class3_size)))
        total_fraction = max(0.0, float(num_layers) * ini_size)
        if n12 > 0:
            a = (total_fraction - n3 * percent) / n12
        else:
            # Degenerate labeling (all layers in one class): fall back to the
            # uniform initial budget so the total is conserved.
            a = total_fraction / max(1, num_layers)
            percent = a
        a = max(0.0, a)
        budgets: list[int] = []
        for c in class_ids:
            share = percent if c == 2 else a
            budgets.append(max(0, int(share * prompt_len)))
    else:
        # Generic n-cluster fallback: linear share interpolation normalized
        # over the USED classes so the total budget is conserved.
        total_budget = float(num_layers) * max(0.0, ini_size)
        shares = [
            (0.5 + 0.5 * rank / max(1, n_clusters - 1)) for rank in range(n_clusters)
        ]
        used_shares = [shares[c] for c in set(class_ids)]
        used_total = sum(used_shares) or 1.0
        budgets = [
            max(
                0,
                int((total_budget * shares[class_ids[i]] / used_total) * prompt_len),
            )
            for i in range(num_layers)
        ]

    diagnostics = {
        "num_layers": num_layers,
        "n_clusters": n_clusters,
        "centers": [float(c) for c in centers],
        "class_of_cluster": {str(k): v for k, v in class_of.items()},
        "class_ids": class_ids,
        "class_sizes": class_sizes,
        "prompt_len": prompt_len,
        "ini_size": ini_size,
        "class3_size": class3_size,
        "total_budget_fraction": float(num_layers) * ini_size,
    }
    return budgets, diagnostics


def _cluster_centers(values: list[float], labels: list[int], n_clusters: int) -> list[float]:
    centers: list[float] = []
    for c in range(n_clusters):
        members = [v for v, label in zip(values, labels) if label == c]
        centers.append(sum(members) / len(members) if members else 0.0)
    return centers


class LayerImportanceAccumulator:
    """Per-request per-layer cosine-similarity accumulation.

    The attention hooks report per-token similarities per layer per step; the
    runner proxy slices them per request and feeds this accumulator.
    """

    def __init__(self) -> None:
        self._sums: dict[int, float] = {}
        self._counts: dict[int, int] = {}

    def add(self, layer_idx: int, similarities: torch.Tensor, start: int, end: int) -> None:
        if start >= end:
            return
        try:
            values = similarities[start:end].detach().float()
            total = float(values.sum().item())
            count = int(values.numel())
        except Exception:
            return
        self._sums[layer_idx] = self._sums.get(layer_idx, 0.0) + total
        self._counts[layer_idx] = self._counts.get(layer_idx, 0) + count

    def means(self, num_layers: int) -> list[float]:
        out: list[float] = []
        for layer_idx in range(num_layers):
            count = self._counts.get(layer_idx, 0)
            out.append(self._sums.get(layer_idx, 0.0) / count if count > 0 else 0.0)
        return out

    def clear(self) -> None:
        self._sums = {}
        self._counts = {}

    def __bool__(self) -> bool:
        return bool(self._counts)
