"""Subset evaluator + ablation harness for overlap-aware KV compression.

Runs the *real* GLA-CLIP inference path on the first N validation images and
reports mIoU / pixel accuracy, end-to-end latency, peak GPU memory and the mean
KV retention ratio.  Intended for fast, same-subset ablation studies.

Example:
    python tools/subset_eval.py \
        --config ./data/_gla_configs/cfg_voc21.py \
        --num-images 100 --tag baseline \
        --out reports/abl_voc21_baseline.json
"""
import argparse
import json
import os
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--num-images', type=int, default=100)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--tag', default='run')
    p.add_argument('--out', default=None)
    p.add_argument('--num-classes', type=int, default=21)
    p.add_argument('--ignore-index', type=int, default=255)
    # compression controls
    p.add_argument('--overlap_compression', action='store_true')
    p.add_argument('--compression_mode', default='none',
                   choices=['none', 'sim', 'location', 'uniform'])
    p.add_argument('--merge_tau', type=float, default=0.9)
    p.add_argument('--no_count_adapt', dest='count_adapt', action='store_false', default=True)
    p.add_argument('--uniform_keep_ratio', type=float, default=0.5)
    return p.parse_args()


def build_model(cfg_path, args):
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
    model_cfg.overlap_compression = args.overlap_compression
    model_cfg.compression_mode = args.compression_mode if args.overlap_compression else 'none'
    model_cfg.merge_tau = args.merge_tau
    model_cfg.count_adapt = args.count_adapt
    model_cfg.uniform_keep_ratio = args.uniform_keep_ratio
    cfg.model.model_cfg = model_cfg
    DefaultScope.get_instance('gla_subset', scope_name='mmseg')
    model = MODELS.build(cfg.model)
    model.eval()
    return cfg, model, model_cfg


class Confusion:
    def __init__(self, num_classes, ignore):
        self.n = num_classes
        self.ignore = ignore
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, gt):
        pred = pred.reshape(-1)
        gt = gt.reshape(-1)
        valid = gt != self.ignore
        pred, gt = pred[valid], gt[valid]
        valid = (gt >= 0) & (gt < self.n)
        pred, gt = pred[valid], gt[valid]
        k = (gt * self.n + pred).astype(np.int64)
        self.mat += np.bincount(k, minlength=self.n * self.n).reshape(self.n, self.n)

    def metrics(self):
        mat = self.mat
        tp = np.diag(mat).astype(np.float64)
        fp = mat.sum(0) - tp
        fn = mat.sum(1) - tp
        denom = tp + fp + fn
        iou = np.full(self.n, np.nan)
        acc = np.full(self.n, np.nan)
        nz = denom > 0
        iou[nz] = tp[nz] / denom[nz]
        acc[nz] = tp[nz] / np.maximum(mat.sum(1)[nz].astype(np.float64), 1)
        return {
            'mIoU': float(np.nanmean(iou) * 100.0),
            'aAcc': float(tp.sum() / max(mat.sum(), 1) * 100.0),
            'mAcc': float(np.nanmean(acc) * 100.0),
        }


def main():
    args = parse_args()
    cfg, model, model_cfg = build_model(args.config, args)

    DefaultScope.get_instance('gla_subset', scope_name='mmseg')
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    n = min(args.num_images, len(dataset) - args.start)
    print(f'[{args.tag}] dataset={cfg.dataset_type} evaluating {n} images '
          f'({args.start}..{args.start + n}) mode={model_cfg.compression_mode} '
          f'tau={args.merge_tau} count_adapt={args.count_adapt}')

    conf = Confusion(args.num_classes, args.ignore_index)
    times, mems, retentions = [], [], []

    for k in range(n):
        i = args.start + k
        data = dataset[i]
        processed = model.data_preprocessor(pseudo_collate([data]), training=False)
        inputs = processed['inputs'].cuda()
        data_samples = processed['data_samples']
        gt = data_samples[0].gt_sem_seg.data.cpu().numpy()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            model.predict(inputs, data_samples)
        torch.cuda.synchronize()
        t1 = time.time()

        times.append(t1 - t0)
        mems.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
        stats = getattr(model_cfg, '_kv_stats', None)
        if stats is not None and model_cfg.compression_mode != 'none':
            retentions.append(stats['retention'])

        pred = data_samples[0].pred_sem_seg.data.cpu().numpy()
        conf.update(pred, gt)

        if (k + 1) % 10 == 0 or k == 0:
            m = conf.metrics()
            print(f'  [{args.tag}] {k + 1}/{n} mIoU={m["mIoU"]:.2f} '
                  f't={np.mean(times):.3f}s mem={np.max(mems):.0f}MB '
                  f'ret={np.mean(retentions) if retentions else 1.0:.3f}')

    metrics = conf.metrics()
    # warmup-robust timing: drop the first sample
    t_steady = times[1:] if len(times) > 1 else times
    result = {
        'tag': args.tag,
        'config': args.config,
        'dataset': cfg.dataset_type,
        'num_images': n,
        'start': args.start,
        'compression_mode': model_cfg.compression_mode,
        'merge_tau': args.merge_tau,
        'count_adapt': args.count_adapt,
        'uniform_keep_ratio': args.uniform_keep_ratio,
        'mIoU': metrics['mIoU'],
        'aAcc': metrics['aAcc'],
        'mAcc': metrics['mAcc'],
        'mean_latency_s': float(np.mean(t_steady)),
        'peak_mem_mb': float(np.max(mems)),
        'mean_kv_retention': float(np.mean(retentions)) if retentions else 1.0,
    }
    print(f'[{args.tag}] RESULT ' + json.dumps({k: v for k, v in result.items()
                                                 if k not in ('config',)}))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
