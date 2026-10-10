"""Change only recall K on existing frozen test queries; optional fixed DPO rerank."""
import argparse,time,random
from pathlib import Path
import numpy as np
from .posttrain_data import read_json,write_json,sha,verify_manifest,eligible_gallery
from .posttrain_followup import save_csv,table
from .posttrain_run import adapter_hash,check_images


def recall(args):
    from .catalog import read_index
    from .vision import load_clip,standard_clip_batch,rgb_image
    from .recall import select_candidates
    import torch
    data=Path(args.data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'recall.json').exists(): raise ValueError('Recall output exists; use rerank/report or a fresh directory')
    manifest=verify_manifest(data); check_images(data)
    if manifest['k']!=12 or manifest['index_sha256']!=sha(Path(args.index)/'index.npz'):
        raise ValueError('Requires original K12 benchmark and unchanged index')
    items,vectors=read_index(args.index)
    by_key={(i['provenance']['item_id'],i['image_path']):j for j,i in enumerate(items)}
    products=[r for r in read_json(data/'splits.json') if r['split']=='test']
    indices=[by_key[(r['identity'],r['image_path'])] for r in products]
    gallery=[dict(items[j],identity=r['identity'],group=r['group']) for j,r in zip(indices,products)]
    gv=vectors[indices]; cases=[r for r in read_json(data/'test.json') if r['scenario']=='closed']
    model=load_clip(); rows=[]; frozen=[]; rng=random.Random(20261009)
    # Warm up encoder separately from timed single-query encoding.
    standard_clip_batch(model,[rgb_image(cases[0]['image_path'])])
    for n,case in enumerate(cases):
        im=rgb_image(case['image_path']); torch.cuda.synchronize(); start=time.perf_counter()
        qv=standard_clip_batch(model,[im])[0]; torch.cuda.synchronize(); encoding=time.perf_counter()-start
        start=time.perf_counter(); scores=gv@qv; similarity_seconds=time.perf_counter()-start
        eligible=eligible_gallery(gallery,case,'closed'); candidates=[gallery[j] for j in eligible]; ss=scores[eligible]
        hits={}; timings={12:[],30:[]}
        for repetition in range(6):
            order=[12,30]; rng.shuffle(order)
            for k in order:
                start=time.perf_counter()
                hits[k]=select_candidates(candidates,ss,case['predicted_categories'],k=k,threshold=.2,dual=True)
                elapsed=time.perf_counter()-start
                if repetition: timings[k].append(elapsed)
        old_ids={h['identity'] for h in case['candidates']}
        if old_ids!={h['identity'] for h in hits[12]}:
            raise ValueError(f'K12 does not reproduce frozen candidates: {case["case_id"]}')
        found={k:any(h['identity']==case['identity'] for h in hits[k]) for k in hits}
        if found[12]!=case['has_positive']: raise ValueError('Frozen recall label mismatch')
        row={'query_id':case['query_id'],'category':case['category'],'identity':case['identity'],
             'hit12':found[12],'hit30':found[30],'rescued':not found[12] and found[30],
             'lost':found[12] and not found[30],'encoding_ms':encoding*1000,'similarity_ms':similarity_seconds*1000,
             'select12_ms':float(np.median(timings[12]))*1000,'select30_ms':float(np.median(timings[30]))*1000,
             'candidates12':len(hits[12]),'candidates30':len(hits[30])}
        rows.append(row); frozen.append(dict(case,candidates12=hits[12],candidates30=hits[30]))
        if n%20==0: print('Recall',n,len(cases),flush=True)
    write_json(out/'cases.json',frozen); write_json(out/'recall.json',rows)
    image_paths={c['image_path'] for c in frozen}|{h['image_path'] for c in frozen for h in c['candidates30']+c['candidates12']}
    write_json(out/'input_hashes.json',{p:sha(p) for p in sorted(image_paths)})
    write_json(out/'config.json',{'parent_manifest_sha256':sha(data/'manifest.json'),'cases_sha256':sha(out/'cases.json'),
                                 'input_hashes_sha256':sha(out/'input_hashes.json'),'index_sha256':manifest['index_sha256'],
                                 'ks':[12,30],'categories':'frozen per query','threshold':.2,'dual':True})
    save_csv(out/'recall.csv',rows)
    report(args)


def rerank(args):
    from .posttrain_model import PairModel
    import torch
    out=Path(args.out); config=read_json(out/'config.json')
    if sha(Path(args.data)/'manifest.json')!=config['parent_manifest_sha256'] or sha(out/'cases.json')!=config['cases_sha256']:
        raise ValueError('Frozen query/candidate data changed')
    if sha(out/'input_hashes.json')!=config['input_hashes_sha256']: raise ValueError('Image manifest changed')
    for p,digest in read_json(out/'input_hashes.json').items():
        if sha(p)!=digest: raise ValueError('Image changed: '+p)
    run=read_json(Path(args.adapter)/'run.json')
    if run['manifest_sha256']!=config['parent_manifest_sha256'] or run['mode']!='dpo' or run['smoke_only']:
        raise ValueError('Expected original full DPO checkpoint')
    meta={'adapter_sha256':adapter_hash(args.adapter),'config_sha256':sha(out/'config.json'),'pixels':run['pixels']}
    if (out/'rerank_config.json').exists() and read_json(out/'rerank_config.json')!=meta:
        raise ValueError('Score cache configuration mismatch')
    write_json(out/'rerank_config.json',meta)
    cache=read_json(out/'pairs.json') if (out/'pairs.json').exists() else {}
    model=PairModel(args.adapter,max_pixels=run['pixels']*28*28); cases=read_json(out/'cases.json'); rows=[]
    rng=random.Random(20261009)
    with torch.inference_mode():
        for n,c in enumerate(cases):
            union={h['identity']:h for h in c['candidates12']+c['candidates30']}
            order=list(union);rng.shuffle(order)
            for identity in order:
                hit=union[identity]; key=c['query_id']+'|'+identity
                if key not in cache:
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    lp=model.logps([c['image_path'],hit['image_path']]);torch.cuda.synchronize()
                    cache[key]={'margin':float(lp[0]-lp[1]),'seconds':time.perf_counter()-start,
                                'peak_mib':torch.cuda.max_memory_allocated()/1024**2}
                    write_json(out/'pairs.json',cache)
            for k in (12,30):
                hits=c[f'candidates{k}']; ranked=sorted(hits,key=lambda h:(cache[c['query_id']+'|'+h['identity']]['margin'],h['similarity'],h['identity']),reverse=True)
                rows.append({'query_id':c['query_id'],'k':k,'recalled':any(h['identity']==c['identity'] for h in hits),
                    'correct':bool(ranked and ranked[0]['identity']==c['identity']),
                    'top_identity':ranked[0]['identity'] if ranked else None,
                    'rerank_seconds':sum(cache[c['query_id']+'|'+h['identity']]['seconds'] for h in hits),
                    'peak_mib':max((cache[c['query_id']+'|'+h['identity']]['peak_mib'] for h in hits),default=0)})
            if n%10==0: print('DPO rerank',n,len(cases),flush=True)
    write_json(out/'rerank.json',rows);save_csv(out/'rerank.csv',rows);report(args)


def report(args):
    out=Path(args.out); rows=read_json(out/'recall.json'); summary=[]
    for k in (12,30):
        summary.append({'k':k,'queries':len(rows),'hits':sum(r[f'hit{k}'] for r in rows),
          'recall':sum(r[f'hit{k}'] for r in rows)/len(rows),
          'mean_candidates':float(np.mean([r[f'candidates{k}'] for r in rows])),
          'mean_select_ms':float(np.mean([r[f'select{k}_ms'] for r in rows])),
          'mean_encode_similarity_select_ms':float(np.mean([r['encoding_ms']+r['similarity_ms']+r[f'select{k}_ms'] for r in rows]))})
    text='# 固定查询 K=12 / K=30 实验\n\n固定原120个测试查询、785个测试图库商品、类别路由、0.20阈值、双路策略。K12候选逐查询精确复现。未重新训练。\n\n'
    text+=table(summary,list(summary[0]))
    text+=f"\n原44个漏召回中找回 {sum(r['rescued'] for r in rows)} 个；原已召回中丢失 {sum(r['lost'] for r in rows)} 个；K30剩余漏召回 {sum(not r['hit30'] for r in rows)} 个。\n"
    stats={'recall':summary,'rescued':sum(r['rescued'] for r in rows),'lost':sum(r['lost'] for r in rows)}
    if (out/'rerank.json').exists():
        rr=read_json(out/'rerank.json'); rs=[]
        for k in (12,30):
            selected=[r for r in rr if r['k']==k]
            if len(selected)!=len(rows): raise ValueError('Incomplete reranking')
            correct=sum(r['correct'] for r in selected);recalled=sum(r['recalled'] for r in selected)
            rs.append({'k':k,'correct':correct,'retrieval_top1':correct/len(rows),'conditional_top1':correct/recalled,
                       'mean_rerank_seconds':float(np.mean([r['rerank_seconds'] for r in selected])),
                       'peak_mib':max(r['peak_mib'] for r in selected)})
        before={r['query_id']:r for r in rr if r['k']==12};after={r['query_id']:r for r in rr if r['k']==30}
        stats['rerank']=rs;stats['top1_gained']=sum(not before[q]['correct'] and after[q]['correct'] for q in before)
        stats['top1_lost']=sum(before[q]['correct'] and not after[q]['correct'] for q in before)
        text+='\n## 同一个 DPO 模型重排\n\n'+table(rs,list(rs[0]))
        text+=f"\nTop-1 新增正确 {stats['top1_gained']} 个，退步 {stats['top1_lost']} 个。\n"
    text+='\n耗时为本次机器上的参考值。编码/相似度共享；候选筛选交错重复5次取各查询中位数。重排对每个图像对只实际评分一次，两种K对同一分数复用并累加各自候选对耗时；不是两次独立端到端请求。不含定位、加载模型或答案生成。只评估有同款的120查询，未重校准拒绝阈值或测无匹配错误接受率，不能据此自动上线K30。\n'
    write_json(out/'summary.json',stats);(out/'REPORT.md').write_text(text,encoding='utf-8');print(text,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['recall','rerank','report']);p.add_argument('--data',required=True)
    p.add_argument('--out',required=True);p.add_argument('--index');p.add_argument('--adapter');args=p.parse_args()
    globals()[args.action](args)

if __name__=='__main__': main()
