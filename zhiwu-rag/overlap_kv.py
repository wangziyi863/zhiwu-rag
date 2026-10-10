"""Overlap-aware KV compression for GLA-CLIP sliding-window inference.

Each sliding window is encoded independently and every window's VFM tokens are
concatenated into one global key/value set.  Because neighbouring windows
overlap, the *same original-image location* is often encoded several times.
This module detects those duplicates and merges only the ones whose DINO
features are consistent, while keeping every retained token's *representative
count* so the dynamic-normalization statistics of GLA-CLIP stay calibrated.

Three compression modes (all gated by ``model_cfg.overlap_compression``):

* ``location``: merge every token that maps to the same location (aggressive).
* ``sim``     : merge tokens at the same location only when their normalised
                DINO cosine similarity exceeds ``model_cfg.merge_tau``
                (the proposed overlap-aware strategy).
* ``uniform`` : keep an evenly spaced subset of the original KV set (the
                naive baseline to compare against).

The module is pure-functional: it never mutates the model and returns new,
shorter tensors plus a per-token representative count.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# location grouping
# ---------------------------------------------------------------------------
def _location_cluster_ids(indices: torch.Tensor, img_w: int, patch: int) -> Tuple[torch.Tensor, int]:
    """Group token ids by the image location they cover.

    ``indices`` holds the flattened original-image pixel index of the top-left
    corner of every token (GLA-CLIP already builds this mapping).  We snap each
    token to a global grid using its centre so that windows whose pixel grids
    are not perfectly aligned (ragged last window) still pair up.
    """
    y = torch.div(indices, img_w, rounding_mode='floor')
    x = indices - y * img_w
    cy = torch.div(y + patch // 2, patch, rounding_mode='floor')
    cx = torch.div(x + patch // 2, patch, rounding_mode='floor')
    gx_max = int(cx.max().item()) + 1 if cx.numel() else 0
    gk = cy * gx_max + cx
    uniq, assign = torch.unique(gk, return_inverse=True)
    return assign, int(uniq.numel())


def _greedy_labels_padded(feats: torch.Tensor, lengths: torch.Tensor,
                          valid: torch.Tensor, tau: float) -> torch.Tensor:
    """Vectorised greedy first-fit clustering for a batch of small groups.

    ``feats``  : [G, m, C]  L2-normalised features (row t of group g)
    ``lengths``: [G]        number of valid members per group
    ``valid``  : [G, m]     bool mask of valid slots
    Returns ``labels`` : [G, m] int64 cluster label; a label equals the index of
    the group member that acts as the cluster representative.
    """
    G, m, _ = feats.shape
    device = feats.device
    labels = torch.zeros((G, m), dtype=torch.long, device=device)
    is_rep = torch.zeros((G, m), dtype=torch.bool, device=device)
    labels[:, 0] = 0
    is_rep[:, 0] = valid[:, 0]
    for t in range(1, m):
        sim = torch.einsum('gc,ghc->gh', feats[:, t], feats[:, :t])  # [G, t]
        sim = sim.masked_fill(~is_rep[:, :t], float('-inf'))
        best_sim, best_idx = sim.max(dim=1)
        join = best_sim > tau
        labels[:, t] = torch.where(join, best_idx, torch.full_like(best_idx, t))
        is_rep[:, t] = (~join) & valid[:, t]
    labels = torch.where(valid, labels, torch.zeros_like(labels))
    return labels


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _aggregate(key: torch.Tensor, v_ext: torch.Tensor, cluster_ids: torch.Tensor, M: int):
    """Average members of each cluster (keys re-normalised afterwards)."""
    device = key.device
    N = key.shape[1]
    one = torch.ones(N, device=device, dtype=key.dtype)
    rep_count = torch.zeros(M, device=device, dtype=key.dtype).index_add_(0, cluster_ids, one)  # [M]

    key_sum = torch.zeros(M, key.shape[-1], device=device, dtype=key.dtype)
    key_sum.index_add_(0, cluster_ids, key[0])
    key_c = F.normalize(key_sum / rep_count.unsqueeze(1), dim=-1).unsqueeze(0)  # [1, M, C]

    v_sum = torch.zeros(v_ext.shape[0], M, v_ext.shape[-1], device=device, dtype=v_ext.dtype)
    v_sum.index_add_(1, cluster_ids, v_ext)
    v_ext_c = v_sum / rep_count.view(1, -1, 1)  # [h, M, Dh]
    return key_c, v_ext_c, rep_count.unsqueeze(0)  # rep_count [1, M]


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def compress_kv(key: torch.Tensor, v_ext: torch.Tensor, indices: torch.Tensor,
                model_cfg, grid_h: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(key_c, v_ext_c, rep_count)`` for the requested compression mode.

    key       : [1, N, C]   L2-normalised DINO keys (global KV set)
    v_ext     : [h, N, Dh]  values aligned with ``key``
    indices   : [N]         flattened original-image pixel index per token
    """
    mode = getattr(model_cfg, 'compression_mode', 'sim')
    device = key.device
    N = key.shape[1]

    if mode == 'uniform':
        ratio = float(getattr(model_cfg, 'uniform_keep_ratio', 0.5))
        M = max(1, int(round(N * ratio)))
        keep = torch.linspace(0, N - 1, M, device=device).round().long().unique()
        key_c = key[:, keep]
        v_ext_c = v_ext[:, keep]
        rep = torch.ones(keep.numel(), device=device, dtype=key.dtype).unsqueeze(0)
        _record(model_cfg, N, int(keep.numel()), mode)
        return key_c, v_ext_c, rep

    patch = max(1, int(getattr(model_cfg, 'slide_crop', 224))) // max(1, int(grid_h))
    img_w = int(getattr(model_cfg, 'img_w', 0)) or (int(indices.max().item()) + 1)
    assign, n_loc = _location_cluster_ids(indices, img_w, patch)

    if mode == 'location':
        counts = torch.bincount(assign, minlength=n_loc)
        rep = counts[assign].to(key.dtype)
        key_c, v_ext_c, rep_count = _aggregate(key, v_ext, assign, n_loc)
        _record(model_cfg, N, n_loc, mode)
        return key_c, v_ext_c, rep_count

    if mode != 'sim':
        raise ValueError(f'unknown compression_mode={mode!r}')

    # ---- location + similarity merge --------------------------------------
    tau = float(getattr(model_cfg, 'merge_tau', 0.9))
    counts = torch.bincount(assign, minlength=n_loc)
    is_multi = counts[assign] > 1
    cluster_ids = torch.full((N,), -1, dtype=torch.long, device=device)

    sidx = torch.nonzero(~is_multi).flatten()
    cluster_ids[sidx] = torch.arange(sidx.numel(), device=device)
    next_id = int(sidx.numel())

    midx = torch.nonzero(is_multi).flatten()
    if midx.numel():
        g = assign[midx]
        order = torch.argsort(g, stable=True)
        midx_s = midx[order]
        g_s = g[order]
        change = torch.nonzero(g_s[1:] != g_s[:-1]).flatten() + 1
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), change])
        ends = torch.cat([change, torch.tensor([midx_s.numel()], dtype=torch.long, device=device)])

        G = starts.numel()
        lengths = ends - starts
        m_max = int(lengths.max().item())
        slots = torch.arange(m_max, device=device)
        pos = starts.unsqueeze(1) + slots.unsqueeze(0)          # [G, m]
        valid = slots.unsqueeze(0) < lengths.unsqueeze(1)       # [G, m]
        tok = midx_s[pos.clamp(max=midx_s.numel() - 1)]          # [G, m]
        feats = key[0, tok]                                      # [G, m, C]
        labels = _greedy_labels_padded(feats, lengths, valid, tau)

        gidx = torch.arange(G, device=device).unsqueeze(1).expand(G, m_max)
        keys = (gidx * m_max + labels)[valid]
        toks = tok[valid]
        uniqk, inv = torch.unique(keys, return_inverse=True)
        cluster_ids[toks] = inv + next_id
        next_id += int(uniqk.numel())

    M = next_id
    key_c, v_ext_c, rep_count = _aggregate(key, v_ext, cluster_ids, M)
    _record(model_cfg, N, M, mode)
    return key_c, v_ext_c, rep_count


def _record(model_cfg, N: int, M: int, mode: str) -> None:
    model_cfg._kv_stats = {
        'N': int(N),
        'M': int(M),
        'retention': float(M) / float(max(N, 1)),
        'mode': mode,
    }
