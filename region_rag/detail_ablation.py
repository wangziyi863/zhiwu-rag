"""A-E resolution, DINO local-view and pairwise ranking experiments."""
import argparse
import csv
import random
import time
from pathlib import Path
import torch
from PIL import Image,ImageOps
from .posttrain_model import PairModel,messages
from .posttrain_data import read_json,write_json,sha,verify_manifest,metrics
from .posttrain_run import adapter_hash
from .posttrain_followup import far_threshold,paired_counts,table
from .dpo_rerank import load_policy
from .detail_core import pair_key,bind_config,choose_triplets,rank_objective
from .training_quality import load_exclusions,filter_training

ARMS={'A':{'pixels':128,'local':False,'trained':False},
      'B':{'pixels':256,'local':False,'trained':False},
      'C':{'pixels':128,'local':True,'trained':False},
      'D':{'pixels':128,'local':False,'trained':True},
      'E':{'pixels':128,'local':True,'trained':True}}


class DetailPairModel(PairModel):
    def __init__(self,*args,local=False,local_map=None,pixels=128,**kwargs):
        self.local=local; self.local_map=local_map or {}; self.pixels=pixels
        self.last_visual_tokens=0; self.last_local=False
        super().__init__(*args,max_pixels=pixels*28*28,**kwargs)

    def inputs(self,paths):
        crop_paths=None
        if self.local:
            entry=self.local_map.get(pair_key(paths))
            if entry is None: raise ValueError('Pair missing from prepared local cache')
            crop_paths=entry['crops']
        all_paths=list(paths)+(crop_paths or [])
        images=[]
        for p in all_paths:
            with Image.open(p) as im: images.append(ImageOps.exif_transpose(im).convert('RGB'))
        prompt=messages()
        if crop_paths:
            prompt[1]['content']=[{'type':'image'} for _ in range(4)]+[{'type':'text','text':
                'Images 1 and 2 show the whole query object and catalog product. Images 3 and 4 are '
                'automatically matched local regions of images 1 and 2. Do the whole objects show the same '
                'product design, possibly from different viewpoints? Compare visible shape, connections, '
                'layout and repeated structures. Local matches may be imperfect; use whole objects for context. '
                'Occluded parts are unknown, not absent. Similar category alone is insufficient. '
                'Answer exactly Yes for a same-product match, otherwise No.'}]
        pixels=256 if self.local and not crop_paths else self.pixels
        # Small ABO sources otherwise stay at native resolution when only the
        # upper bound is raised. B/C/E explicitly allocate their target budget;
        # A/D retain the exact deployed baseline's min=64, max=128 processing.
        minimum=pixels if self.local or self.pixels==256 else 64
        text=self.processor.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],images=images,return_tensors='pt',
                              min_pixels=minimum*28*28,max_pixels=pixels*28*28).to('cuda')
        if inputs.input_ids.shape[1]>1600: raise ValueError('Unexpected input length exceeds experiment budget')
        pad=self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        self.last_visual_tokens=int((inputs.input_ids==pad).sum())
        self.last_local=bool(crop_paths)
        return inputs


def save_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def setup(args):
    data=Path(args.data); out=Path(args.out); root=Path(args.root)
    verify_manifest(data); policy=load_policy(root)
    if read_json(root/'dpo_adapter/run.json')['manifest_sha256']!=sha(data/'manifest.json'):
        raise ValueError('DPO checkpoint trained on another dataset')
    # Exact source fingerprints prevent stale inference/training caches after edits.
    code={name:sha(Path(__file__).parent/name) for name in ('detail_ablation.py','detail_core.py','detail_prepare.py','posttrain_model.py')}
    config={'manifest':sha(data/'manifest.json'),'parent':policy['adapter_sha256'],
            'limit':args.limit,'steps':args.steps,'accum':args.accum,'lr':args.lr,'seed':args.seed,
            'margin':args.margin,'ce_weight':args.ce_weight,'code':code,'version':1}
    if getattr(args,'quality_review',None):
        excluded=load_exclusions(data,args.quality_review)
        config['quality_review']={'sha256':sha(args.quality_review),'excluded_queries':sorted(excluded),
                                  'code_sha256':sha(Path(__file__).parent/'training_quality.py')}
    bind_config(out/'experiment',config)
    return data,out,root,policy,config


def get_locals(out):
    if not (out/'prepare/complete.json').is_file(): raise ValueError('Run prepare first')
    return read_json(out/'prepare/locals.json')


def mine(args):
    data,out,root,policy,config=setup(args)
    pool=read_json(out/'prepare/pool.json')
    if 'quality_review' in config:
        pool,_=filter_training(pool,set(config['quality_review']['excluded_queries']))
    prep=read_json(out/'prepare/config.json')
    if prep['manifest']!=config['manifest'] or prep['limit']!=args.limit: raise ValueError('Prepared data mismatch')
    folder=out/'mining'; bind_config(folder,{'experiment':config,'pool':sha(out/'prepare/pool.json')})
    path=folder/'scores.json'; cache=read_json(path) if path.exists() else {}
    runtime=DetailPairModel(policy['adapter'])
    with torch.inference_mode():
        for n,pair in enumerate(pool):
            key=pair['query_id']+'|'+pair['candidate_identity']
            if not pair['query_id'].startswith('train_'): raise ValueError('Nontraining mining pair')
            if key not in cache:
                lp=runtime.logps(pair['images']); cache[key]=float(lp[0]-lp[1])
                if n%20==0: write_json(path,cache); print('mine',n,len(pool),flush=True)
    write_json(path,cache)
    triplets=choose_triplets(pool,cache); write_json(folder/'triplets.json',triplets)
    write_json(folder/'complete.json',{'triplets':len(triplets),'pool_pairs':len(pool),
              'excluded_ambiguous':sum(bool(p.get('ambiguous')) for p in pool),'smoke':bool(args.limit)})


def train(args):
    data,out,root,policy,config=setup(args); spec=ARMS[args.arm]
    if not spec['trained']: raise ValueError('Only D/E require training')
    triplets=read_json(out/'mining/triplets.json')
    if 'quality_review' in config:
        triplets,_=filter_training(triplets,set(config['quality_review']['excluded_queries']))
    if not triplets: raise ValueError('No usable training triplets')
    folder=out/f'{args.arm}_adapter'
    bind_config(folder,{'experiment':config,'arm':args.arm,'triplets':sha(out/'mining/triplets.json'),
                        'locals':sha(out/'prepare/locals.json') if spec['local'] else None})
    if (folder/'run.json').exists():
        if not (folder/'adapter_model.safetensors').exists(): raise ValueError('Missing trained weights')
        print('Completed adapter reused:',args.arm); return
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    runtime=DetailPairModel(policy['adapter'],training=True,local=spec['local'],
                            local_map=get_locals(out) if spec['local'] else None,pixels=spec['pixels'])
    params=[p for p in runtime.model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.)
    before=[p.detach().cpu().clone() for p in params]
    rng=random.Random(args.seed); order=list(range(len(triplets))); rng.shuffle(order); cursor=0; log=[]
    torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter()
    # An interrupted training restarts from the fixed parent, never partially saved weights.
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True); losses=[]
        for _ in range(args.accum):
            if cursor>=len(order): rng.shuffle(order); cursor=0
            triple=triplets[order[cursor]]; cursor+=1
            pos=runtime.logps(triple['positive']); neg=runtime.logps(triple['negative'])
            loss=rank_objective(pos,neg,args.margin,args.ce_weight)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite ranking loss')
            (loss/args.accum).backward(); losses.append(float(loss.detach()))
        grad=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradients')
        opt.step(); log.append({'step':step+1,'loss':sum(losses)/len(losses),'grad_norm':float(grad)})
        write_json(folder/'training_log.json',log); print(args.arm,log[-1],flush=True)
    delta=sum(float((p.detach().cpu()-b).abs().sum()) for p,b in zip(params,before))
    if delta<=0: raise RuntimeError('Adapter did not change')
    runtime.model.save_pretrained(folder,safe_serialization=True)
    write_json(folder/'run.json',{'mode':'pairwise_rank_plus_ce','parent':policy['adapter_sha256'],
               'config':config,'arm':args.arm,'adapter_l1_delta':delta,'smoke':bool(args.limit),
               'seconds':time.perf_counter()-begin,'peak_mib':torch.cuda.max_memory_allocated()/1024**2})


def score(args):
    data,out,root,policy,config=setup(args); spec=ARMS[args.arm]
    adapter=out/f'{args.arm}_adapter' if spec['trained'] else Path(policy['adapter'])
    if spec['trained']:
        run=read_json(adapter/'run.json')
        if run['config']!=config or run['arm']!=args.arm: raise ValueError('Training configuration mismatch')
    folder=out/args.arm/args.split
    scoring={'experiment':config,'arm':args.arm,'split':args.split,'adapter':adapter_hash(adapter),
             'locals':sha(out/'prepare/locals.json') if spec['local'] else None}
    bind_config(folder,scoring)
    path=folder/'pairs.json'; cache=read_json(path) if path.exists() else {}
    cases=read_json(data/f'{args.split}.json')
    if args.limit: cases=cases[:args.limit*2]
    runtime=DetailPairModel(adapter,local=spec['local'],local_map=get_locals(out) if spec['local'] else None,pixels=spec['pixels'])
    rows=[]
    with torch.inference_mode():
        for n,case in enumerate(cases):
            ranked=[]; times=[]; tokens=[]; peaks=[]; local_count=0
            for hit in case['candidates']:
                key=case['query_id']+'|'+hit['identity']
                if key not in cache:
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
                    lp=runtime.logps([case['image_path'],hit['image_path']]); torch.cuda.synchronize()
                    cache[key]={'score':float(lp[0]-lp[1]),'seconds':time.perf_counter()-start,
                                'visual_tokens':runtime.last_visual_tokens,'used_local':runtime.last_local,
                                'peak_mib':torch.cuda.max_memory_allocated()/1024**2}
                    write_json(path,cache)
                pair=cache[key]
                ranked.append((pair['score'],hit['similarity'],hit['identity']))
                times.append(pair['seconds']); tokens.append(pair['visual_tokens']); peaks.append(pair['peak_mib']); local_count+=int(pair['used_local'])
            ranked.sort(reverse=True); top=ranked[0] if ranked else None
            rows.append({'case_id':case['case_id'],'scenario':case['scenario'],'has_positive':case['has_positive'],
                         'top_correct':bool(top and top[2]==case['identity']),'score':top[0] if top else None,
                         'top_identity':top[2] if top else None,'pair_seconds_sum':sum(times),'peak_mib':max(peaks,default=0),
                         'visual_tokens_sum':sum(tokens),'local_pairs':local_count,'candidate_count':len(ranked)})
            write_json(folder/'rows.partial.json',rows)
            if n%10==0: print(args.arm,args.split,n,len(cases),flush=True)
    write_json(folder/'rows.json',rows)
    write_json(folder/'complete.json',dict(scoring,cases=len(rows),smoke=bool(args.limit)))


def report(args):
    data,out,root,policy,config=setup(args)
    summaries=[]; results={}
    for arm in ARMS:
        splits={}
        for split in ('val','test'):
            folder=out/arm/split; done=read_json(folder/'complete.json'); rows=read_json(folder/'rows.json')
            if done['experiment']!=config or done['arm']!=arm or done['split']!=split: raise ValueError('Mismatched experiment')
            expected=read_json(data/f'{split}.json')
            if args.limit: expected=expected[:args.limit*2]
            if len(rows)!=len(expected) or {r['case_id'] for r in rows}!={r['case_id'] for r in expected}: raise ValueError('Incomplete scores')
            splits[split]=rows
        threshold=far_threshold(splits['val'],.1); test=splits['test']; count=sum(r['candidate_count'] for r in test)
        summaries.append({'arm':arm,'threshold':threshold,**metrics(test,threshold),
                          'validation_far':metrics(splits['val'],threshold)['false_accept_rate'],
                          'mean_visual_tokens_per_pair':sum(r['visual_tokens_sum'] for r in test)/max(1,count),
                          'local_pair_fraction':sum(r['local_pairs'] for r in test)/max(1,count),
                          'mean_rerank_seconds':sum(r['pair_seconds_sum'] for r in test)/len(test),
                          'peak_mib':max(r['peak_mib'] for r in test)})
        results[arm]=test
    paired=[dict(before=a,after=b,**paired_counts(results[a],results[b])) for a,b in [('A','B'),('B','C'),('A','D'),('C','E'),('D','E')]]
    save_csv(out/'summary.csv',summaries); save_csv(out/'paired.csv',paired)
    write_json(out/'analysis.json',{'summary':summaries,'paired':paired,'smoke':bool(args.limit),'config':config})
    title='SMOKE ONLY: pipeline check, not accuracy evidence' if args.limit else 'A–E detail and ranking ablation'
    text='# '+title+'\n\n'+table(summaries,list(summaries[0]))+'\n\n'+table(paired,list(paired[0]))
    text+='\nAll arms share frozen candidates. Thresholds use validation only (FAR target 10%; actual test FAR can differ). '
    text+='C/E add correspondence crops AND a contextual prompt; B/C do not isolate prompt effects. '
    text+='D/E share mined training triplets, seed, optimizer steps and CE weight, not equal FLOPs. '
    text+='D versus A includes extra training; attribution solely to ranking loss needs an additional equal-budget CE-only control. '
    text+='Local matching uses DINO saliency, not ground-truth masks/semantic part labels. '
    text+='Timings exclude offline DINO preparation, model loading and CLIP; offline time is recorded in prepare/complete.json. '
    text+='The existing repeatedly examined test set is regression data, not a new independent generalization test.\n'
    (out/'REPORT.md').write_text(text,encoding='utf-8'); print(text,flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['mine','train','score','report'])
    p.add_argument('--data',required=True); p.add_argument('--root',required=True); p.add_argument('--out',required=True)
    p.add_argument('--arm',choices=list(ARMS)); p.add_argument('--split',choices=['val','test'])
    p.add_argument('--steps',type=int,default=90); p.add_argument('--accum',type=int,default=4)
    p.add_argument('--lr',type=float,default=5e-6); p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--margin',type=float,default=.5); p.add_argument('--ce-weight',type=float,default=.2)
    p.add_argument('--quality-review',help='Training-only reviewed query exclusions; frozen eval data is unchanged')
    p.add_argument('--limit',type=int,default=0); args=p.parse_args()
    if min(args.steps,args.accum,args.lr)<=0 or args.margin<0 or args.ce_weight<0 or args.limit<0: p.error('Invalid training settings')
    if args.action in ('train','score') and not args.arm: p.error('--arm required')
    if args.action=='score' and not args.split: p.error('--split required')
    globals()[args.action](args)

if __name__=='__main__': main()
