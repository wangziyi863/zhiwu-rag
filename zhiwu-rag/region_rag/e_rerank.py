"""Serving for the evaluated clean E adapter; exact experiment preprocessing."""
import argparse
import math
import time
from pathlib import Path
from .posttrain_data import read_json,write_json,sha
from .posttrain_run import adapter_hash
from .posttrain_followup import far_threshold
from .detail_core import pair_key
from .dpo_rerank import rank_pairs


def validate_e_metadata(run,meta,rows,digest):
    config=run.get('config',{})
    if (run.get('arm')!='E' or run.get('mode')!='pairwise_rank_plus_ce' or run.get('smoke') is not False
            or config.get('limit')!=0 or not config.get('quality_review',{}).get('sha256')
            or run.get('parent')!=config.get('parent') or meta.get('experiment')!=config
            or meta.get('arm')!='E' or meta.get('split')!='val' or meta.get('smoke') is not False
            or meta.get('adapter')!=digest):
        raise ValueError('Clean E weights do not match the full validation evaluation')
    if (len(rows)!=meta.get('cases') or len({r['case_id'] for r in rows})!=len(rows)
            or {r['scenario'] for r in rows}!={'closed','open'}
            or any(r['score'] is not None and not math.isfinite(r['score']) for r in rows)):
        raise ValueError('Incomplete or invalid E validation scores')


def load_e_policy(root):
    root=Path(root).resolve(); adapter=root/'E_adapter'
    run=read_json(adapter/'run.json'); meta=read_json(root/'E/val/complete.json')
    rows=read_json(root/'E/val/rows.json'); digest=adapter_hash(adapter)
    validate_e_metadata(run,meta,rows,digest)
    # Reuse the exact evaluated class; refuse silent preprocessing changes.
    for name in ('detail_ablation.py','detail_prepare.py','detail_core.py','posttrain_model.py'):
        if sha(Path(__file__).parent/name)!=run['config']['code'][name]:
            raise ValueError('Evaluated E implementation changed: '+name)
    prep=read_json(root/'prepare/config.json')
    if prep['manifest']!=run['config']['manifest'] or prep['limit']!=0:
        raise ValueError('E local preparation metadata mismatch')
    return {'adapter':str(adapter),'adapter_sha256':digest,'threshold':far_threshold(rows,.1),
            'calibrated_recall_k':12,'validation_far_target':.1,'method':'e_pairwise_rank',
            'task':'same_product','pixels':128,'local':True,'preparation':prep,
            'quality_review_sha256':run['config']['quality_review']['sha256']}


def validate_local_inputs(pairs,local):
    for paths in pairs:
        row=local.get(pair_key(paths))
        if not isinstance(row,dict) or row.get('paths')!=paths or 'crops' not in row:
            raise ValueError('Online local map does not match requested image pair')
        crops=row['crops']
        if crops is not None and (not isinstance(crops,list) or len(crops)!=2
                                  or not all(Path(p).is_file() for p in crops)):
            raise ValueError('Missing E local crop images')


def rerank(request):
    import torch
    from .detail_ablation import DetailPairModel
    policy=load_e_policy(request['policy_root'])
    path=Path(request['local_map']); local=read_json(path); binding=read_json(path.parent/'binding.json')
    if binding['adapter_sha256']!=policy['adapter_sha256']:
        raise ValueError('Local preparation belongs to another E checkpoint')
    pairs=[[request['crop'],h['image_path']] for h in request['hits']]
    validate_local_inputs(pairs,local)
    for image in {p for pair in pairs for p in pair}:
        if binding['image_hashes'].get(image)!=sha(image): raise ValueError('Prepared image contents changed')
    runtime=DetailPairModel(policy['adapter'],local=True,local_map=local,pixels=128)
    scores=[]; observations=[]; seconds=0.; peak=0.
    with torch.inference_mode():
        for paths in pairs:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
            lp=runtime.logps(paths); torch.cuda.synchronize()
            seconds+=time.perf_counter()-start; peak=max(peak,torch.cuda.max_memory_allocated()/1024**2)
            scores.append(float(lp[0]-lp[1]))
            observations.append({'used_local':runtime.last_local,'visual_tokens':runtime.last_visual_tokens})
    result=rank_pairs(request['hits'],scores,policy['threshold'])
    for row in result['audit']: row.update(observations[row['recall_rank']-1])
    for hit in result['hits']: hit['rerank']['method']='e_pairwise_rank'
    result.update(method='e_pairwise_rank',policy=policy,
        stats={'pairs':len(scores),'pair_seconds_sum':seconds,'peak_allocated_mib':peak,
               'local_pairs':sum(r['used_local'] for r in observations)},
        calibration_matches_recall_k=request.get('recall_k',12)==12)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--request',required=True)
    request=read_json(p.parse_args().request); result=rerank(request); out=Path(request['output'])
    write_json(out/'rerank_audit.json',result['audit']); write_json(out/'rerank_result.json',result)


if __name__=='__main__': main()
