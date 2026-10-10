"""Frozen ABO cross-view benchmark. Never mutates the live retrieval index."""
import argparse
import csv
import gzip
import hashlib
import json
import random
import os
import tempfile
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path=Path(path)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity_groups(rows):
    parent={r['identity']:r['identity'] for r in rows}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    tokens={}
    for row in rows:
        for t in row['tokens']:
            if t in tokens:
                a,b=find(row['identity']),find(tokens[t]); parent[max(a,b)]=min(a,b)
            else: tokens[t]=row['identity']
    return {x:find(x) for x in parent}


def signature(path):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        exact=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        a=np.asarray(im.convert('L').resize((9,8)))
        dh=int.from_bytes(np.packbits((a[:,1:]>a[:,:-1]).reshape(-1)).tobytes(),'big')
        return exact,dh,min(im.size)


def eligible_gallery(gallery,query,scenario):
    return [j for j,g in enumerate(gallery) if g['group']!=query['group'] or
            (scenario=='closed' and g['identity']==query['identity'])]


def ratio(n,d):
    return n/d if d else None


def metrics(rows,threshold):
    closed=[r for r in rows if r['scenario']=='closed']
    opened=[r for r in rows if r['scenario']=='open']
    recalled=[r for r in closed if r['has_positive']]
    return {'closed_queries':len(closed),'no_match_queries':len(opened),
            'recall_at_k':ratio(len(recalled),len(closed)),
            'conditional_top1':ratio(sum(r['top_correct'] for r in recalled),len(recalled)),
            'retrieval_top1':ratio(sum(r['top_correct'] for r in closed),len(closed)),
            'accepted_correct_rate':ratio(sum(r['top_correct'] and r['score'] is not None and r['score']>=threshold for r in closed),len(closed)),
            'false_accept_rate':ratio(sum(r['score'] is not None and r['score']>=threshold for r in opened),len(opened))}


def calibrate(rows):
    if not any(r['scenario']=='closed' for r in rows) or not any(r['scenario']=='open' for r in rows):
        raise ValueError('Validation requires closed and no-match scenarios')
    vals=sorted(set(r['score'] for r in rows if r['score'] is not None))
    if not vals: return 0.
    thresholds=[vals[0]-1]+[(a+b)/2 for a,b in zip(vals,vals[1:])]+[vals[-1]+1]
    # Equal weight to correct acceptance and correct rejection; ties prefer fewer false accepts.
    def objective(t):
        m=metrics(rows,t)
        return ((m['accepted_correct_rate']+1-m['false_accept_rate'])/2,-m['false_accept_rate'],t)
    return max(thresholds,key=objective)


def verify_manifest(directory):
    directory=Path(directory); manifest=read_json(directory/'manifest.json')
    for name,digest in manifest['files'].items():
        if sha(directory/name)!=digest: raise ValueError(f'Frozen data changed: {name}')
    return manifest


def prepare(args):
    from .catalog import read_index
    from .recall import select_candidates,resolve_categories
    from .core import TARGETS
    from .vision import load_clip,standard_clip_batch,rgb_image
    from open_clip import tokenize
    import torch
    import torch.nn.functional as F
    out=Path(args.out).resolve()
    if out.exists(): raise ValueError('Output already exists. Use a new directory; frozen data is never overwritten.')
    out.mkdir(parents=True)
    items,vectors=read_index(args.index)
    old=read_json(args.exclude_queries)
    excluded={q['product_id'] for q in old}
    root=Path(args.abo)
    with gzip.open(root/'images/metadata/images.csv.gz','rt') as f:
        paths={r['image_id']:root/'images/small'/r['path'] for r in csv.DictReader(f)}
    aliases=defaultdict(list)
    for j,item in enumerate(items): aliases[item['provenance']['item_id']].append(j)
    records={}; image_ids=defaultdict(set)
    for shard in sorted((root/'listings/metadata').glob('*.json.gz')):
        with gzip.open(shard,'rt') as f:
            for line in f:
                row=json.loads(line); pid=row.get('item_id')
                if pid not in aliases: continue
                image_ids[pid].update(row.get('other_image_id',[]))
                if row.get('main_image_id'): image_ids[pid].add(row['main_image_id'])
                records[pid]=row
    cache={}
    def sig(path):
        key=str(path)
        if key not in cache: cache[key]=signature(path)
        return cache[key]
    print('Grouping ABO identities and checking image duplicates...',flush=True)
    gallery_hashes=set(); gallery_dhash=set()
    for item in items:
        h,d,_=sig(item['image_path']); gallery_hashes.add(h); gallery_dhash.add(d)
    rows=[]; queries={}; stats=Counter(); rng=random.Random(args.seed)
    for pid,indices in sorted(aliases.items()):
        tokens=['id:'+x for x in image_ids[pid]]
        tokens+=['pixel:'+sig(items[j]['image_path'])[0] for j in indices]
        alternatives=sorted(image_ids[pid]); rng.shuffle(alternatives)
        for image_id in alternatives:
            path=paths.get(image_id)
            if not path or not path.is_file(): continue
            h,d,size=sig(path)
            if size<100 or h in gallery_hashes or any((d^g).bit_count()<=4 for g in gallery_dhash):
                stats['excluded_duplicate_or_small']+=1; continue
            tokens.append('pixel:'+h)
            queries[pid]={'identity':pid,'image_path':str(path),'image_id':image_id,'sha256':sha(path),
                          'category':items[indices[0]]['category']}
            break
        rows.append({'identity':pid,'tokens':tokens})
    groups=identity_groups(rows)
    grouped=defaultdict(list)
    for pid,group in groups.items(): grouped[group].append(pid)
    blocked={groups[p] for p in excluded if p in groups}
    buckets=defaultdict(list)
    for group,pids in sorted(grouped.items()):
        cats={items[j]['category'] for p in pids for j in aliases[p]}
        if group in blocked or len(cats)!=1:
            stats['excluded_development_or_conflicting_groups']+=1; continue
        buckets[next(iter(cats))].append(group)
    assignment={}
    for category,gs in sorted(buckets.items()):
        rng.shuffle(gs); a=int(len(gs)*.7); b=int(len(gs)*.85)
        for k,g in enumerate(gs): assignment[g]='train' if k<a else 'val' if k<b else 'test'
    products=[]
    # Grouping prevents leakage, but shared images do NOT establish same-product labels.
    # Deduplicate cross-market records by canonical product ID only.
    for group,split in sorted(assignment.items()):
        for pid in grouped[group]:
            j=min(aliases[pid]); item=dict(items[j],identity=pid,group=group)
            products.append((split,j,item))
    selected={}; wanted={'train':args.train_queries,'val':args.val_queries,'test':args.test_queries}
    for split in wanted:
        pool=defaultdict(list)
        for group in sorted(g for g,s in assignment.items() if s==split):
            options=[queries[p] for p in grouped[group] if p in queries]
            if options: pool[options[0]['category']].append(dict(options[0],group=group))
        for values in pool.values(): rng.shuffle(values)
        chosen=[]
        while len(chosen)<wanted[split] and any(pool.values()):
            for category in sorted(pool):
                if pool[category] and len(chosen)<wanted[split]: chosen.append(pool[category].pop())
        if len(chosen)<wanted[split]: raise ValueError(f'{split}: only {len(chosen)} usable queries, requested {wanted[split]}')
        selected[split]=chosen
    model=load_clip(); names=list(TARGETS)
    with torch.inference_mode():
        text_vectors=F.normalize(model.encode_text(tokenize([f'a photo of a {c}.' for c in names]).cuda()).float(),dim=-1).cpu().numpy()
    train_pairs=[]; counts={}; manifest_items=[]
    for split,chosen in selected.items():
        entries=[(j,i) for s,j,i in products if s==split]; gallery=[i for j,i in entries]
        gv=vectors[[j for j,i in entries]]; cases=[]
        for qi,q in enumerate(chosen):
            qv=standard_clip_batch(model,[rgb_image(q['image_path'])])[0]
            scores=gv@qv; cs=dict(zip(names,(text_vectors@qv).astype(float).tolist()))
            cats=resolve_categories(max(cs,key=cs.get),cs)
            qid=f'{split}_{qi:04d}'
            for scenario in ('closed','open'):
                eligible=eligible_gallery(gallery,q,scenario)
                hits=select_candidates([gallery[j] for j in eligible],scores[eligible],cats,
                                       k=args.k,threshold=.2,dual=True)
                rng.shuffle(hits)
                case=dict(q,query_id=qid,case_id=qid+'_'+scenario,scenario=scenario,candidates=hits,
                          predicted_categories=cats,has_positive=any(h['identity']==q['identity'] for h in hits))
                cases.append(case)
            if split=='train':
                positive=next(g for g in gallery if g['identity']==q['identity'])
                negatives=[gallery[int(j)] for j in np.argsort(-scores,kind='stable') if gallery[int(j)]['group']!=q['group']][:args.negatives]
                for j,candidate in enumerate([positive]+negatives):
                    label=j==0
                    train_pairs.append({'pair_id':qid+f'_{j}','query_id':qid,'images':[q['image_path'],candidate['image_path']],
                                        'label':label,'chosen':'Yes' if label else 'No','rejected':'No' if label else 'Yes',
                                        'query_identity':q['identity'],'candidate_identity':candidate['identity']})
            if qi%25==0: print(split,qi,len(chosen),flush=True)
        write_json(out/f'{split}.json',cases)
        counts[split]={'queries':len(chosen),'gallery_products':len(gallery),
                       'gallery_groups':len(set(i['group'] for i in gallery)),'cases':len(cases),
                       'closed_recall':sum(c['has_positive'] for c in cases if c['scenario']=='closed')/len(chosen)}
        manifest_items.extend({'identity':i['identity'],'group':i['group'],'split':split,'image_path':i['image_path']} for i in gallery)
    write_json(out/'train_pairs.json',train_pairs)
    write_json(out/'splits.json',manifest_items)
    # Reserve actual user-scene evaluation schema, never invent its correctness labels.
    write_json(out/'real_scenes_template.json',{'status':'pending_ground_truth','examples':[],
        'schema':{'query_id':'unique','image_path':'absolute scene path','target_description':'front chair',
                  'bbox':[0,0,100,100],'acceptable_product_ids':[],'verified_no_match':False}})
    file_names=['train.json','val.json','test.json','train_pairs.json','splits.json']
    image_paths={p for r in train_pairs for p in r['images']}
    for split in selected:
        for case in read_json(out/f'{split}.json'):
            image_paths.add(case['image_path']); image_paths.update(h['image_path'] for h in case['candidates'])
    write_json(out/'image_hashes.json',{p:sha(p) for p in sorted(image_paths)})
    file_names.append('image_hashes.json')
    manifest={'version':2,'seed':args.seed,'k':args.k,'clip_threshold':.2,'dual_recall':True,
              'counts':counts,'training_pairs':len(train_pairs),'index_sha256':sha(Path(args.index)/'index.npz'),
              'development_exclusions_sha256':sha(args.exclude_queries),'stats':dict(stats),
              'task':'ABO metadata-derived same-product image-pair ranking; NOT general similarity or scene localization',
              'split_policy':'identity/shared image IDs/selected pixel hashes; per-split galleries; old development groups removed',
              'label_policy':'Same canonical product ID only; other products in the shared-image group excluded from both positives and negatives',
              'no_match_policy':'same query with all ground-truth group gallery entries removed; synthetic',
              'files':{n:sha(out/n) for n in file_names}}
    write_json(out/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--abo',required=True); p.add_argument('--index',required=True)
    p.add_argument('--exclude-queries',required=True); p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=20261008); p.add_argument('--k',type=int,default=12)
    p.add_argument('--train-queries',type=int,default=240); p.add_argument('--val-queries',type=int,default=60)
    p.add_argument('--test-queries',type=int,default=120); p.add_argument('--negatives',type=int,default=2)
    args=p.parse_args()
    if min(args.train_queries,args.val_queries,args.test_queries,args.negatives)<1: p.error('Counts must be positive')
    prepare(args)


if __name__=='__main__': main()
