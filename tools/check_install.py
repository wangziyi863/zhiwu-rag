"""Validate the portable source, evaluated policies, and optional local runtime."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',action='store_true');args=p.parse_args()
    from region_rag.e_rerank import load_e_policy
    from region_rag.dpo_rerank import load_policy
    e=load_e_policy(ROOT/'models/clean_e');dpo=load_policy(ROOT/'models/dpo')
    assert sha(ROOT/'vendor/dino/vision_transformer.py')==e['preparation']['dino_code']
    print('PASS: clean E + DPO adapter identities, validation calibration and evaluated code hashes')
    print('E threshold:',repr(e['threshold']),'DPO threshold:',repr(dpo['threshold']))
    if args.runtime:
        import torch
        from mmseg.models.segmentors import BaseSegmentor
        assert torch.cuda.is_available(),'CUDA unavailable'
        assert torch.cuda.is_bf16_supported(),'Qwen runtime requires BF16 support'
        weights=Path(os.environ.get('DINO_WEIGHTS',ROOT/'models/base/dino_vitbase8_pretrain.pth')).expanduser()
        assert sha(weights)==e['preparation']['weights'],'DINO weights mismatch'
        manifest=json.loads((ROOT/'models/base_manifest.json').read_text())
        assert sha(Path.home()/'.cache/clip/ViT-B-16.pt')==manifest['clip']['sha256'],'CLIP weights mismatch'
        cache=Path(os.environ.get('HF_HUB_CACHE',Path.home()/'.cache/huggingface/hub'))
        for model in manifest['huggingface']:
            folder=cache/('models--'+model['repo_id'].replace('/','--'))
            assert (folder/'refs/main').read_text().strip()==model['revision'],'HF revision mismatch: '+model['repo_id']
            snapshot=folder/'snapshots'/model['revision']
            assert (snapshot/'config.json').is_file()
            indices=list(snapshot.glob('*.safetensors.index.json'))
            if indices:
                for index in indices:
                    for shard in set(json.loads(index.read_text())['weight_map'].values()):
                        assert (snapshot/shard).is_file(),'Missing weight shard: '+shard
            else: assert list(snapshot.glob('*.safetensors')),'Missing HF weights'
        probes=[('RAG_PYTHON','.venv-rag','from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Sam2Model, AutoModelForZeroShotObjectDetection'),
                ('DPO_PYTHON','.venv-posttrain','import peft; from region_rag.detail_ablation import DetailPairModel')]
        for variable,folder,code in probes:
            python=os.environ.get(variable,str(ROOT/folder/'bin/python'))
            subprocess.run([python,'-c',code],cwd=ROOT,check=True)
        print('PASS: CUDA, BF16, MMSegmentation, model caches and worker imports')
        print('GPU:',torch.cuda.get_device_name())
    print('No GPU inference was performed by this check. Build an index and run a query for end-to-end acceptance.')

if __name__=='__main__':main()
