"""Probe: how much cross-window KV redundancy exists, how similar are the
duplicate tokens, and how much does the global KV attention cost?

This is a *read-only* analysis tool. It hooks `myutils.KV_Extension.forward`
without changing any model behaviour, runs a few validation images through the
real GLA-CLIP inference path, and reports:

  1. overlap redundancy  : total KV tokens vs unique original-image locations
  2. duplicate similarity: cosine similarity between tokens sharing a location
  3. compressibility     : retained KV ratio for a similarity threshold tau
  4. attention cost      : time spent inside KV_Extension vs whole forward

Example:
    python tools/analyze_overlap.py \
        --config ./data/_gla_configs/cfg_voc21.py \
        --num-images 4
"""
import argparse
import os
import sys
import time
import json

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import gla_clip_segmentor  # noqa: F401  (registers the segmentor)
import custom_datasets      # noqa: F401  (registers the datasets)
import mmseg.datasets.transforms  # noqa: F401  (registers PackSegInputs etc.)
import myutils

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import DefaultScope
from mmseg.registry import MODELS, DATASETS
from cfg import ModelConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='./data/_gla_configs/cfg_voc21.py')
    p.add_argument('--num-images', type=int, default=4)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--taus', type=float, nargs='+', default=[0.9, 0.95, 0.98, 0.99])
    p.add_argument('--out', default='reports/overlap_probe.json')
    p.add_argument('--save-tensors', default=None, help='optional .pt to dump captured key/indices')
    return p.parse_args()


def build_model_and_cfg(cfg_path):
    cfg = Config.fromfile(cfg_path)
    model_cfg = ModelConfig(**cfg.model)
    model_cfg.CLIP_type = 'ProxyCLIP'
    model_cfg.token_norm = True
    model_cfg.KV_token_extension = True
    model_cfg.proxy_sim = True
    model_cfg.mini_iters = 2
    model_cfg.initial_crit_pos = 0.6
    model_cfg.dynamic_beta = True
    model_cfg.beta_alpha = 0.3
    model_cfg.dynamic_gamma = True
    model_cfg.gamma_alpha = 30
    cfg.model.model_cfg = model_cfg
    model = MODELS.build(cfg.model)
    model.eval()
    return cfg, model


def make_capture():
    """Monkeypatch KV_Extension.forward to record its inputs + runtime."""
    cap = {}
    original_forward = myutils.KV_Extension.forward

    def wrapped(self, ex_feats_grid, num_heads=12, scale=1, beta=1.2,
                gamma=3.0, indices=None, v_ext=None, model_cfg=None):
        torch.cuda.synchronize()
        t0 = time.time()
        out = original_forward(self, ex_feats_grid, num_heads=num_heads, scale=scale,
                               beta=beta, gamma=gamma, indices=indices,
                               v_ext=v_ext, model_cfg=model_cfg)
        torch.cuda.synchronize()
        t1 = time.time()

        # Reconstruct the exact normalised key / value ordering used inside KV_Extension.
        h_grids, w_grids, C, H, W = ex_feats_grid.shape
        B = h_grids * w_grids
        S = H * W
        key = ex_feats_grid.permute(0, 1, 3, 4, 2).reshape(1, B * S, C)
        key = torch.nn.functional.normalize(key, dim=-1)

        cap['key'] = key.detach().float().cpu()
        cap['indices'] = None if indices is None else indices.flatten().detach().cpu()
        cap['h_grids'] = h_grids
        cap['w_grids'] = w_grids
        cap['H'] = H
        cap['W'] = W
        cap['img_h'] = getattr(model_cfg, 'img_h', None)
        cap['img_w'] = getattr(model_cfg, 'img_w', None)
        cap['kv_time'] = t1 - t0
        return out

    myutils.KV_Extension.forward = wrapped
    return cap, original_forward


def greedy_group_size(key, tau):
    """Size of the representative set after greedily merging tokens whose
    cosine similarity to a cluster representative exceeds tau.

    key: [N, C] L2-normalised.
    """
    key = key
    N = key.shape[0]
    assigned = torch.zeros(N, dtype=torch.bool)
    reps = []
    for i in range(N):
        if assigned[i]:
            continue
        reps.append(i)
        assigned[i] = True
        sims = key[i + 1:] @ key[i]
        cand = (~assigned[i + 1:]) & (sims > tau)
        assigned[i + 1:][cand] = True
    return len(reps)


def overlap_stats(key, indices, img_h, img_w, taus):
    """key [N,C] normalised, indices [N] flattened original-image pixel ids."""
    N = key.shape[0]
    valid = indices > 0  # 0 == padding sentinel
    idx = indices.clone()
    # group by absolute pixel id
    uniq, inverse, counts = torch.unique(idx, return_inverse=True, return_counts=True)
    n_unique = uniq.numel()
    dup_mask = counts[inverse] > 1

    # pairwise similarity among tokens that share a location
    dup_sims = []
    n_pairs = 0
    # sort by index so equal ids are contiguous
    order = torch.argsort(idx)
    sorted_idx = idx[order]
    sorted_key = key[order]
    start = 0
    loc_groups = 0
    for j in range(1, N + 1):
        if j == N or sorted_idx[j] != sorted_idx[start]:
            g = j - start
            if g > 1:
                loc_groups += 1
                sub = sorted_key[start:j]
                sim = sub @ sub.t()
                iu = torch.triu_indices(g, g, offset=1)
                dup_sims.append(sim[iu[0], iu[1]])
                n_pairs += iu.shape[1]
            start = j
    dup_sims = torch.cat(dup_sims) if dup_sims else torch.zeros(0)

    stats = {
        'N_total': int(N),
        'N_unique_locations': int(n_unique),
        'N_duplicate_tokens': int(dup_mask.sum()),
        'redundancy_ratio': float(1.0 - n_unique / N),
        'n_multi_location_groups': int(loc_groups),
        'n_duplicate_pairs': int(n_pairs),
        'dup_sim_mean': float(dup_sims.mean()) if n_pairs else None,
        'dup_sim_median': float(dup_sims.median()) if n_pairs else None,
        'dup_sim_p10': float(dup_sims.quantile(0.10)) if n_pairs else None,
        'dup_sim_frac_above': {
            str(t): float((dup_sims > t).float().mean()) if n_pairs else None for t in taus
        },
    }
    return stats


def compressibility_stats(key, indices, taus):
    """Compressibility under location + similarity grouping.

    Unique locations are always kept (this is the location-aware ceiling).
    Within each multi-token location group, tokens are greedily merged at tau.
    """
    idx = indices
    N = key.shape[0]
    order = torch.argsort(idx)
    sorted_idx = idx[order]
    sorted_key = key[order]

    # ceiling: unique locations
    uniq = torch.unique(idx)
    ceiling = uniq.numel()

    out = {'location_ceiling_retention': float(ceiling / N)}
    start = 0
    for tau in taus:
        kept = 0
        start = 0
        for j in range(1, N + 1):
            if j == N or sorted_idx[j] != sorted_idx[start]:
                g = j - start
                if g == 1:
                    kept += 1
                else:
                    kept += greedy_group_size(sorted_key[start:j], tau)
                start = j
        out[f'retention_tau_{tau}'] = float(kept / N)
    return out


def main():
    args = parse_args()
    DefaultScope.get_instance('gla_probe', scope_name='mmseg')
    cfg, model = build_model_and_cfg(args.config)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    n = min(args.num_images, len(dataset))
    print(f'dataset={cfg.dataset_type}  size={len(dataset)}  probing {n} images')

    cap, original_forward = make_capture()
    per_image = []
    try:
        for i in range(args.start, args.start + n):
            data = dataset[i]
            processed = model.data_preprocessor(pseudo_collate([data]), training=False)
            inputs = processed['inputs'].cuda()
            data_samples = processed['data_samples']

            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                model.predict(inputs, data_samples)
            torch.cuda.synchronize()
            t1 = time.time()

            key = cap['key'][0]
            indices = cap['indices']
            stats = overlap_stats(key, indices, cap['img_h'], cap['img_w'], args.taus)
            stats.update(compressibility_stats(key, indices, args.taus))
            stats['kv_time_s'] = cap['kv_time']
            stats['forward_time_s'] = t1 - t0
            stats['kv_time_fraction'] = cap['kv_time'] / max(t1 - t0, 1e-9)
            stats['img_h'] = cap['img_h']
            stats['img_w'] = cap['img_w']
            stats['h_grids'] = cap['h_grids']
            stats['w_grids'] = cap['w_grids']
            per_image.append(stats)

            print(f'[{i}] N={stats["N_total"]} unique={stats["N_unique_locations"]} '
                  f'redundancy={stats["redundancy_ratio"]:.3f} '
                  f'dup_sim_mean={stats["dup_sim_mean"]} '
                  f'kv_frac={stats["kv_time_fraction"]:.3f}')
    finally:
        myutils.KV_Extension.forward = original_forward

    if args.save_tensors:
        torch.save({'key': cap.get('key'), 'indices': cap.get('indices')}, args.save_tensors)

    # aggregate
    def agg(field):
        vals = [s[field] for s in per_image if s.get(field) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        'config': args.config,
        'n_images': len(per_image),
        'mean_redundancy_ratio': agg('redundancy_ratio'),
        'mean_dup_sim': agg('dup_sim_mean'),
        'mean_dup_sim_p10': agg('dup_sim_p10'),
        'mean_location_ceiling_retention': agg('location_ceiling_retention'),
        'mean_kv_time_fraction': agg('kv_time_fraction'),
        'mean_retention': {str(t): agg(f'retention_tau_{t}') for t in args.taus},
        'per_image': per_image,
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n=== SUMMARY ===')
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_image'}, indent=2))
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
