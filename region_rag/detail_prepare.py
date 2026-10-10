"""Offline DINO region pairs; no query labels are used for local matching."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from .posttrain_data import read_json,write_json,sha,verify_manifest,signature
from .detail_core import bind_config,pair_key,mutual_region
from .posttrain_run import check_images


def training_pool(data,limit=0):
    cases=read_json(data/'train.json')
    positives={p['query_id']:p for p in read_json(data/'train_pairs.json') if p['label']}
    membership={p['identity']:p for p in read_json(data/'splits.json')}
    pool=[]; signatures={}
    def sig(path):
        if path not in signatures: signatures[path]=signature(path)
        return signatures[path]
    count=0
    for case in cases:
        if case['scenario']!='closed': continue
        if limit and count>=limit: break
        count+=1; qid=case['query_id']; pos=positives[qid]
        candidates={h['identity']:h for h in case['candidates']}
        for pair in read_json(data/'train_pairs.json'):
            if pair['query_id']==qid and not pair['label']:
                candidates.setdefault(pair['candidate_identity'],{'identity':pair['candidate_identity'],'image_path':pair['images'][1]})
        pool.append(dict(pos,ambiguous=False))
        psha,pdh,_=sig(pos['images'][1])
        for identity,hit in candidates.items():
            if identity==case['identity']: continue
            member=membership[identity]
            if member['split']!='train' or member['group']==case['group']: continue
            h,dh,_=sig(hit['image_path'])
            ambiguous=h==psha or (dh^pdh).bit_count()<=6
            pool.append({'query_id':qid,'candidate_identity':identity,'label':False,
                         'images':[case['image_path'],hit['image_path']], 'ambiguous':ambiguous,
                         'exclusion_reason':'near_duplicate_positive_image' if ambiguous else None})
    return pool


class DinoFeatures:
    def __init__(self,hub,weights,cache):
        import torch
        self.torch=torch; self.cache=cache; cache.mkdir(exist_ok=True)
        self.model=torch.hub.load(str(hub),'dino_vitb8',source='local',pretrained=False)
        self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.model=self.model.cuda().eval()

    def features(self,path):
        torch=self.torch; dest=self.cache/(pair_key([path])+'.npz')
        if dest.exists():
            with np.load(dest) as saved: return saved['features'],saved['mask']
        with Image.open(path) as raw:
            im=ImageOps.exif_transpose(raw).convert('RGB'); w,h=im.size
            canvas=ImageOps.pad(im,(224,224),color=(255,255,255))
        a=np.asarray(canvas,dtype=np.float32)/255.
        x=torch.from_numpy(a).permute(2,0,1).unsqueeze(0).cuda()
        x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        with torch.inference_mode():
            feat=self.model.get_intermediate_layers(x)[0][0,1:].float().cpu().numpy()
            att=self.model.get_last_selfattention(x)[0,:,0,1:].mean(0).float().cpu().numpy()
        # DINO class-attention foreground heuristic; padding excluded separately.
        order=np.argsort(-att); cumulative=np.cumsum(att[order])/max(att.sum(),1e-8)
        keep=order[:max(3,int(np.searchsorted(cumulative,.65))+1)]
        mask=np.zeros(28*28,bool); mask[keep]=True
        scale=min(224/w,224/h); rw=round(w*scale); rh=round(h*scale)
        ox=(224-rw)//2; oy=(224-rh)//2
        yy,xx=np.meshgrid(np.arange(28)*8+4,np.arange(28)*8+4,indexing='ij')
        mask &= ((xx>=ox)&(xx<ox+rw)&(yy>=oy)&(yy<oy+rh)).ravel()
        np.savez_compressed(dest,features=feat,mask=mask)
        return feat,mask


def crop_original(path,box,dest):
    with Image.open(path) as raw:
        image=ImageOps.exif_transpose(raw).convert('RGB'); w,h=image.size
        scale=min(224/w,224/h); rw=round(w*scale); rh=round(h*scale)
        ox=(224-rw)//2; oy=(224-rh)//2
        coords=[max(0,min(w,round((box[0]*224-ox)*w/rw))),max(0,min(h,round((box[1]*224-oy)*h/rh))),
                max(0,min(w,round((box[2]*224-ox)*w/rw))),max(0,min(h,round((box[3]*224-oy)*h/rh)))]
        if coords[2]-coords[0]<16 or coords[3]-coords[1]<16: return None
        image.crop(coords).save(dest)
        return coords


def prepare(args):
    data=Path(args.data); out=Path(args.out); verify_manifest(data); check_images(data)
    hub=Path(args.dino_hub).expanduser(); weights=Path(args.dino_weights).expanduser()
    if not hub.is_dir() or not weights.is_file(): raise FileNotFoundError('Local DINO hub/weights missing; set DINO_HUB and DINO_WEIGHTS')
    config={'manifest':sha(data/'manifest.json'),'weights':sha(weights),'dino_code':sha(hub/'vision_transformer.py'),
            'limit':args.limit,'version':1,'tau':.65,'gap':.03,'foreground_mass':.65,
            'code':{n:sha(Path(__file__).parent/n) for n in ('detail_prepare.py','detail_core.py')}}
    bind_config(out,config)
    pool=training_pool(data,args.limit); write_json(out/'pool.json',pool)
    pairs={pair_key(p['images']):p['images'] for p in pool}
    for split in ('val','test'):
        cases=read_json(data/f'{split}.json')
        if args.limit: cases=cases[:args.limit*2]
        for case in cases:
            for hit in case['candidates']:
                paths=[case['image_path'],hit['image_path']]; pairs[pair_key(paths)]=paths
    features=DinoFeatures(hub,weights,out/'features')
    crops=out/'crops'; crops.mkdir(exist_ok=True)
    path=out/'locals.json'; cache=read_json(path) if path.exists() else {}
    start=time.perf_counter()
    for n,(key,paths) in enumerate(pairs.items()):
        if key in cache: continue
        q,qmask=features.features(paths[0]); k,kmask=features.features(paths[1])
        match=mutual_region(q,k,qmask,kmask)
        item={'paths':paths,'crops':None,'match':match}
        if match:
            cp=[str(crops/f'{key}_{j}.png') for j in range(2)]
            boxes=[crop_original(p,b,Path(c)) for p,b,c in zip(paths,[match['query_box'],match['candidate_box']],cp)]
            if all(b is not None for b in boxes): item.update(crops=cp,original_boxes=boxes)
        cache[key]=item
        if n%20==0:
            write_json(path,cache); print('local matching',n+1,len(pairs),flush=True)
    write_json(path,cache)
    write_json(out/'complete.json',{'pairs':len(pairs),'matched':sum(bool(cache[k]['crops']) for k in pairs),
                                  'seconds_this_run':time.perf_counter()-start,'smoke':bool(args.limit)})


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--out',required=True)
    p.add_argument('--dino-hub',default='~/.cache/torch/hub/facebookresearch_dino_main')
    p.add_argument('--dino-weights',default='~/.cache/torch/hub/checkpoints/dino_vitbase8_pretrain.pth')
    p.add_argument('--limit',type=int,default=0); prepare(p.parse_args())

if __name__=='__main__': main()
