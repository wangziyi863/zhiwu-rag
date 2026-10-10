"""Train, score and compare frozen ABO same-product experiments."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

from .posttrain_data import read_json,write_json,sha,verify_manifest,metrics,calibrate


def adapter_hash(path):
    if not path: return 'base'
    path=Path(path)
    files=[path/'adapter_config.json',path/'adapter_model.safetensors']
    return hashlib.sha256(''.join(sha(p) for p in files).encode()).hexdigest()


def check_images(data):
    for p,digest in read_json(Path(data)/'image_hashes.json').items():
        if sha(p)!=digest: raise ValueError('Image changed: '+p)


def validate_lineage(arms):
    if arms['baseline']['adapter_sha256']!='base': raise ValueError('Baseline must be untrained')
    for mode in ('sft','dpo'):
        if arms[mode].get('smoke_checkpoint') or arms[mode].get('training_mode')!=mode:
            raise ValueError('Wrong training arm or smoke checkpoint')
    if arms['dpo'].get('reference_adapter_sha256')!=arms['sft']['adapter_sha256']:
        raise ValueError('DPO must start from the evaluated SFT checkpoint')


def training_parent(mode,sft):
    if mode in ('dpo','continue_sft'):
        if not sft: raise ValueError('This mode requires the exact SFT checkpoint')
        return sft
    if mode!='sft' or sft: raise ValueError('Fresh SFT must not specify an ignored parent')
    return None


def train(args):
    import torch
    from .posttrain_model import PairModel,dpo_loss
    verify_manifest(args.data); check_images(args.data)
    out=Path(args.out)
    if out.exists(): raise ValueError('Training output exists; choose a fresh directory')
    out.mkdir(parents=True)
    parent=training_parent(args.mode,args.sft)
    if args.sft:
        previous=read_json(Path(args.sft)/'run.json')
        if previous['mode']!='sft' or previous['manifest_sha256']!=sha(Path(args.data)/'manifest.json'):
            raise ValueError('SFT checkpoint did not use this frozen dataset')
        if previous['pixels']!=args.pixels: raise ValueError('SFT and DPO must use the same image resolution')
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    pairs=read_json(Path(args.data)/'train_pairs.json')
    if args.limit_pairs: pairs=pairs[:args.limit_pairs]
    if not pairs: raise ValueError('Empty training set')
    runtime=PairModel(parent,training=True,max_pixels=args.pixels*28*28)
    model=runtime.model
    params=[p for p in model.parameters() if p.requires_grad]
    before=[p.detach().cpu().clone() for p in params]
    optim=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.)
    start=time.perf_counter(); torch.cuda.reset_peak_memory_stats(); reference={}
    if args.mode=='dpo':
        model.eval()
        with torch.no_grad():
            for n,pair in enumerate(pairs):
                reference[pair['pair_id']]=runtime.logps(pair['images']).cpu().tolist()
                if n%50==0: print('Frozen SFT reference',n,len(pairs),flush=True)
        write_json(out/'reference_logps.json',{'adapter_sha256':adapter_hash(args.sft),'pairs':reference})
        model.train()
    log=[]; order=list(range(len(pairs))); random.shuffle(order); cursor=0
    for step in range(args.steps):
        optim.zero_grad(set_to_none=True); losses=[]
        for _ in range(args.accum):
            if cursor>=len(order): random.shuffle(order); cursor=0
            pair=pairs[order[cursor]]; cursor+=1
            lp=runtime.logps(pair['images']); chosen=0 if pair['label'] else 1; rejected=1-chosen
            if args.mode in ('sft','continue_sft'): loss=-lp[chosen]
            else:
                ref=torch.tensor(reference[pair['pair_id']],device=lp.device)
                loss=dpo_loss(lp[chosen],lp[rejected],ref[chosen],ref[rejected],args.beta)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            (loss/args.accum).backward(); losses.append(float(loss.detach()))
        grad=float(torch.nn.utils.clip_grad_norm_(params,1.))
        if not torch.isfinite(torch.tensor(grad)): raise RuntimeError('Nonfinite gradients')
        optim.step()
        row={'step':step+1,'loss':sum(losses)/len(losses),'grad_norm':grad}
        log.append(row); print(json.dumps(row),flush=True)
        write_json(out/'training_log.json',log)
    delta=sum(float((p.detach().cpu()-b).abs().sum()) for p,b in zip(params,before))
    if delta<=0: raise RuntimeError('No adapter weights changed')
    model.save_pretrained(out,safe_serialization=True)
    config={'mode':args.mode,'manifest_sha256':sha(Path(args.data)/'manifest.json'),
            'initial_adapter_sha256':adapter_hash(parent) if parent else None,
            'reference_adapter_sha256':adapter_hash(args.sft) if args.mode=='dpo' else None,
            'pixels':args.pixels,'steps':args.steps,'accum':args.accum,'seed':args.seed,'lr':args.lr,
            'beta':args.beta,'training_pairs':len(pairs),'adapter_l1_delta':delta,
            'trainable_parameters':sum(p.numel() for p in params),'wall_seconds':time.perf_counter()-start,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'torch':torch.__version__,'smoke_only':bool(args.limit_pairs)}
    write_json(out/'run.json',config); print(json.dumps(config,indent=2),flush=True)


def score(args):
    import torch
    from .posttrain_model import PairModel
    verify_manifest(args.data); check_images(args.data)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    dataset_hash=sha(Path(args.data)/'manifest.json')
    run={}
    if args.adapter:
        run=read_json(Path(args.adapter)/'run.json')
        if run['manifest_sha256']!=dataset_hash or run['pixels']!=args.pixels: raise ValueError('Checkpoint/data configuration mismatch')
    config={'manifest_sha256':dataset_hash,'adapter_sha256':adapter_hash(args.adapter),
            'split':args.split,'pixels':args.pixels,'limit':args.limit,'task':'single-token same-product policy v1',
            'training_mode':run.get('mode'),'smoke_checkpoint':run.get('smoke_only',False),
            'reference_adapter_sha256':run.get('reference_adapter_sha256')}
    if run.get('mode')=='continue_sft': config['initial_adapter_sha256']=run['initial_adapter_sha256']
    if (out/'config.json').exists() and read_json(out/'config.json')!=config:
        raise ValueError('Existing scoring output belongs to another model/data/config')
    write_json(out/'config.json',config)
    cases=read_json(Path(args.data)/f'{args.split}.json')
    if args.limit: cases=cases[:args.limit]
    # Resume at image-pair granularity. Interrupted runs cannot silently become complete reports.
    cache=read_json(out/'pairs.json') if (out/'pairs.json').exists() else {}
    runtime=PairModel(args.adapter,max_pixels=args.pixels*28*28)
    rows=[]
    with torch.inference_mode():
        for n,case in enumerate(cases):
            assessed=[]; pair_times=[]; peaks=[]
            for hit in case['candidates']:
                key=case['query_id']+'|'+hit['identity']
                if key not in cache:
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter()
                    lp=runtime.logps([case['image_path'],hit['image_path']])
                    torch.cuda.synchronize()
                    cache[key]={'score':float(lp[0]-lp[1]),'seconds':time.perf_counter()-begin,
                                'peak_mib':torch.cuda.max_memory_allocated()/1024**2}
                    write_json(out/'pairs.json',cache)
                result=cache[key]
                assessed.append((result['score'],hit['similarity'],hit['identity']))
                pair_times.append(result['seconds']); peaks.append(result['peak_mib'])
            assessed.sort(reverse=True)
            top=assessed[0] if assessed else None
            rows.append({'case_id':case['case_id'],'query_id':case['query_id'],'scenario':case['scenario'],
                         'has_positive':case['has_positive'],'top_correct':bool(top and top[2]==case['identity']),
                         'score':top[0] if top else None,'top_identity':top[2] if top else None,
                         'pair_seconds_sum':sum(pair_times),'peak_mib':max(peaks,default=0),
                         'candidate_count':len(assessed)})
            write_json(out/'rows.partial.json',rows)
            if n%10==0: print(args.split,n,len(cases),flush=True)
    write_json(out/'rows.json',rows)
    write_json(out/'complete.json',dict(config,cases=len(rows),smoke_only=bool(args.limit)))


def report(args):
    root=Path(args.root); manifest=verify_manifest(args.data); results=[]
    configs=[]; arms={}
    for arm in ('baseline','sft','dpo'):
        config={}
        for split in ('val','test'):
            folder=root/arm/split; done=read_json(folder/'complete.json')
            if done['smoke_only']: raise ValueError('Limited smoke scores cannot produce a full ablation report')
            if done['manifest_sha256']!=sha(Path(args.data)/'manifest.json'): raise ValueError('Frozen benchmark mismatch')
            rows=read_json(folder/'rows.json'); expected=read_json(Path(args.data)/f'{split}.json')
            if len(rows)!=len(expected) or {r['case_id'] for r in rows}!={r['case_id'] for r in expected}:
                raise ValueError('Incomplete or duplicate evaluation cases')
            config[split]=done
        if config['val']['adapter_sha256']!=config['test']['adapter_sha256']: raise ValueError('Validation/test model mismatch')
        configs.extend(config.values())
        arms[arm]=config['test']
        val=read_json(root/arm/'val/rows.json'); test=read_json(root/arm/'test/rows.json')
        threshold=calibrate(val); m=metrics(test,threshold)
        results.append(dict(arm=arm,k=manifest['k'],threshold=threshold,**m,
            mean_rerank_seconds=sum(r['pair_seconds_sum'] for r in test)/len(test),
            peak_mib=max(r['peak_mib'] for r in test)))
    validate_lineage(arms)
    if len({c['pixels'] for c in configs})!=1: raise ValueError('Image resolution differs across arms')
    with (root/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    text='# Same-product reranking ablation\n\n'
    text+='Fractions are 0–1. Conditional top-1 excludes recall misses; retrieval top-1 includes them. This is crop/product-image retrieval, not full-scene end-to-end accuracy. Thresholds use validation only. No-match cases remove the target group from the split gallery. Timing sums pair forwards, excludes model loading/CLIP and is reference only.\n\n'
    keys=list(results[0]); text+='| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'
    for r in results: text+='| '+' | '.join(f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]) for k in keys)+' |\n'
    (root/'summary.md').write_text(text,encoding='utf-8')
    print(text)


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train'); t.add_argument('--mode',choices=['sft','dpo','continue_sft'],required=True)
    t.add_argument('--data',required=True); t.add_argument('--out',required=True); t.add_argument('--sft')
    t.add_argument('--steps',type=int,default=90); t.add_argument('--accum',type=int,default=8)
    t.add_argument('--lr',type=float,default=2e-5); t.add_argument('--beta',type=float,default=.1)
    t.add_argument('--seed',type=int,default=20261008); t.add_argument('--limit-pairs',type=int,default=0)
    t.add_argument('--pixels',type=int,default=128)
    s=sub.add_parser('score'); s.add_argument('--data',required=True); s.add_argument('--out',required=True)
    s.add_argument('--adapter'); s.add_argument('--split',choices=['val','test'],required=True)
    s.add_argument('--limit',type=int,default=0); s.add_argument('--pixels',type=int,default=128)
    r=sub.add_parser('report'); r.add_argument('--data',required=True); r.add_argument('--root',required=True)
    args=p.parse_args()
    if args.command=='train':
        if min(args.steps,args.accum,args.pixels,args.lr,args.beta)<=0: p.error('Training parameters must be positive')
        train(args)
    elif args.command=='score': score(args)
    else: report(args)


if __name__=='__main__': main()
