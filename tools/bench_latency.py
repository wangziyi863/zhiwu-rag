"""Interleaved latency / memory benchmark for the compression modes.

Separate process invocations are unreliable on laptops (thermal & power
throttling makes a later run look ~2x slower than an earlier one).  This tool
loads the model once and interleaves the modes across repeats so that every mode
sees the same thermal state.
"""
import argparse
import json
import os
import statistics as st
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import gla_clip_segmentor  # noqa: F401
import custom_datasets      # noqa: F401
import mmseg.datasets.transforms  # noqa: F401

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import DefaultScope
from mmseg.registry import MODELS, DATASETS
from cfg import ModelConfig


MODES = [
    ('baseline', dict(mode='none', tau=0.9, adapt=True)),
    ('uniform50', dict(mode='uniform', tau=0.9, adapt=True, ratio=0.5)),
    ('location', dict(mode='location', tau=0.9, adapt=True)),
    ('sim80', dict(mode='sim', tau=0.8, adapt=True)),
    ('sim90', dict(mode='sim', tau=0.9, adapt=True)),
    ('sim80_nocount', dict(mode='sim', tau=0.8, adapt=False)),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--num-images', type=int, default=40)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--out', default='reports/bench_voc21.json')
    return p.parse_args()


def set_mode(model_cfg, spec):
    model_cfg.overlap_compression = spec['mode'] != 'none'
    model_cfg.compression_mode = spec['mode']
    model_cfg.merge_tau = spec['tau']
    model_cfg.count_adapt = spec['adapt']
    model_cfg.uniform_keep_ratio = spec.get('ratio', 0.5)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
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
    DefaultScope.get_instance('gla_bench', scope_name='mmseg')
    model = MODELS.build(cfg.model)
    model.eval()
    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    inputs_list, samples_list = [], []
    for i in range(min(args.num_images, len(dataset))):
        data = dataset[i]
        processed = model.data_preprocessor(pseudo_collate([data]), training=False)
        inputs_list.append(processed['inputs'].cuda())
        samples_list.append(processed['data_samples'])

    times = {name: [] for name, _ in MODES}
    mems = {name: [] for name, _ in MODES}
    rets = {name: [] for name, _ in MODES}

    # warmup
    set_mode(model_cfg, dict(mode='none', tau=0.9, adapt=True))
    for i in range(min(args.warmup, len(inputs_list))):
        with torch.no_grad():
            model.predict(inputs_list[i], samples_list[i])
    for name, spec in MODES:
        set_mode(model_cfg, spec)
        with torch.no_grad():
            model.predict(inputs_list[0], samples_list[0])

    for r in range(args.repeats):
        for name, spec in MODES:
            set_mode(model_cfg, spec)
            for i in range(len(inputs_list)):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                with torch.no_grad():
                    model.predict(inputs_list[i], samples_list[i])
                torch.cuda.synchronize()
                times[name].append(time.time() - t0)
                mems[name].append(torch.cuda.max_memory_allocated() / (1024 ** 2))
                stats = getattr(model_cfg, '_kv_stats', None)
                if stats is not None and model_cfg.compression_mode != 'none':
                    rets[name].append(stats['retention'])
                else:
                    rets[name].append(1.0)
        print(f'repeat {r + 1}/{args.repeats} done')

    result = {}
    for name, _ in MODES:
        t = sorted(times[name])
        result[name] = {
            'latency_median_s': float(st.median(t)),
            'latency_mean_s': float(np.mean(t)),
            'latency_p10_s': float(t[int(0.1 * len(t))]),
            'peak_mem_mb': float(np.max(mems[name])),
            'peak_mem_median_mb': float(st.median(mems[name])),
            'retention': float(np.mean(rets[name])),
        }

    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
