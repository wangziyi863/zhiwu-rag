"""Serving adapter for the evaluated DPO same-product policy."""
import argparse
import math
import time
from pathlib import Path
from .posttrain_data import read_json, write_json
from .posttrain_run import adapter_hash
from .posttrain_followup import far_threshold


def load_policy(root):
    root=Path(root).resolve()
    adapter=root/'dpo_adapter'
    run=read_json(adapter/'run.json')
    meta=read_json(root/'dpo/val/complete.json')
    digest=adapter_hash(adapter)
    if (run.get('mode')!='dpo' or run.get('smoke_only') or meta.get('smoke_only')
            or meta.get('smoke_checkpoint') or meta.get('training_mode')!='dpo'
            or meta['adapter_sha256']!=digest or run['pixels']!=meta['pixels']
            or run['manifest_sha256']!=meta['manifest_sha256']):
        raise ValueError('DPO checkpoint does not match the full validation evaluation')
    rows=read_json(root/'dpo/val/rows.json')
    if len(rows)!=meta['cases'] or len({r['case_id'] for r in rows})!=len(rows):
        raise ValueError('Incomplete DPO validation results')
    return {'adapter':str(adapter),'adapter_sha256':digest,'pixels':run['pixels'],
            'threshold':far_threshold(rows,.10),'validation_far_target':.10,
            'calibrated_recall_k':12,'task':'same_product','method':'dpo_yes_no'}


def rank_pairs(hits,scores,threshold):
    if len(hits)!=len(scores) or not math.isfinite(threshold) or any(not math.isfinite(s) for s in scores):
        raise ValueError('Invalid DPO scores')
    # Same tie-break as the frozen scorer: score, CLIP similarity, canonical ID.
    def identity(h):
        return h.get('identity') or h.get('provenance',{}).get('item_id') or h['item_id']
    ordered=sorted(range(len(hits)),key=lambda i:(scores[i],hits[i]['similarity'],identity(hits[i])),reverse=True)
    audit=[{'item_id':hits[i]['item_id'],'evidence_id':hits[i].get('evidence_id'),
            'score':scores[i],'recall_rank':i+1,'above_threshold':scores[i]>=threshold} for i in ordered]
    kept=[]
    if ordered and scores[ordered[0]]>=threshold:
        i=ordered[0]
        kept=[dict(hits[i],recall_rank=i+1,rerank={'method':'dpo_yes_no','score':scores[i],
                     'accepted':True,'threshold':threshold,'score_type':'log_p_yes_minus_log_p_no'})]
    return {'status':'ranked' if kept else 'no_match','hits':kept,'audit':audit,
            'method':'dpo_yes_no','threshold':threshold,'identity_confirmed':False,
            'acceptance_policy':'validation-calibrated top-1 only'}


def rerank(request):
    import torch
    from .posttrain_model import PairModel
    policy=load_policy(request['policy_root'])
    runtime=PairModel(adapter=policy['adapter'],max_pixels=policy['pixels']*28*28)
    scores=[]; seconds=0.; peak=0.
    with torch.inference_mode():
        for hit in request['hits']:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
            lp=runtime.logps([request['crop'],hit['image_path']])
            torch.cuda.synchronize()
            seconds+=time.perf_counter()-start
            peak=max(peak,torch.cuda.max_memory_allocated()/1024**2)
            scores.append(float(lp[0]-lp[1]))
    result=rank_pairs(request['hits'],scores,policy['threshold'])
    result.update(policy=policy,stats={'pairs':len(scores),'pair_seconds_sum':seconds,'peak_allocated_mib':peak},
                  calibration_matches_recall_k=request.get('recall_k',12)==policy['calibrated_recall_k'])
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--request',required=True)
    request=read_json(parser.parse_args().request)
    result=rerank(request)
    out=Path(request['output'])
    write_json(out/'rerank_audit.json',result['audit'])
    write_json(out/'rerank_result.json',result)


if __name__=='__main__': main()
