"""Online DINO correspondence, identical to the E experiment's offline path."""
import argparse
import os
import time
from pathlib import Path
from .posttrain_data import read_json,write_json,sha
from .detail_core import bind_config,pair_key,mutual_region
from .detail_prepare import DinoFeatures,crop_original
from .e_rerank import load_e_policy


def prepare(request):
    start=time.perf_counter(); policy=load_e_policy(request['policy_root'])
    hub=Path(os.environ.get('DINO_HUB','~/.cache/torch/hub/facebookresearch_dino_main')).expanduser()
    weights=Path(os.environ.get('DINO_WEIGHTS','~/.cache/torch/hub/checkpoints/dino_vitbase8_pretrain.pth')).expanduser()
    expected=policy['preparation']
    if sha(weights)!=expected['weights'] or sha(hub/'vision_transformer.py')!=expected['dino_code']:
        raise ValueError('Online DINO weights/code differ from E evaluation')
    pairs={pair_key([request['crop'],h['image_path']]):[request['crop'],h['image_path']] for h in request['hits']}
    out=Path(request['output'])/'e_local'; out.mkdir(exist_ok=True)
    binding={'adapter_sha256':policy['adapter_sha256'],
             'image_hashes':{p:sha(p) for p in sorted({p for paths in pairs.values() for p in paths})},
             'preparation':expected}
    bind_config(out,binding); write_json(out/'binding.json',binding)
    features=DinoFeatures(hub,weights,out/'features'); crops=out/'crops'; crops.mkdir(exist_ok=True)
    local={}
    for key,paths in pairs.items():
        q,qmask=features.features(paths[0]); k,kmask=features.features(paths[1])
        match=mutual_region(q,k,qmask,kmask,tau=expected['tau'],gap=expected['gap'])
        item={'paths':paths,'crops':None,'match':match}
        if match:
            cp=[str(crops/f'{key}_{j}.png') for j in range(2)]
            boxes=[crop_original(p,b,Path(c)) for p,b,c in zip(paths,[match['query_box'],match['candidate_box']],cp)]
            if all(b is not None for b in boxes): item.update(crops=cp,original_boxes=boxes)
        local[key]=item
    path=out/'locals.json'; write_json(path,local)
    return {'local_map':str(path),'pairs':len(local),'matched':sum(bool(v['crops']) for v in local.values()),
            'seconds':time.perf_counter()-start,'adapter_sha256':policy['adapter_sha256']}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--request',required=True)
    request=read_json(p.parse_args().request)
    write_json(Path(request['output'])/'prepare_e_result.json',prepare(request))


if __name__=='__main__': main()
