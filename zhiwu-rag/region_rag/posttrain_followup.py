"""Paired query analysis and validation-only FAR calibration; optional SFT control."""
import argparse
import csv
import math
from pathlib import Path
from .posttrain_data import read_json,write_json,sha,verify_manifest,metrics,calibrate
from .posttrain_run import adapter_hash,validate_lineage


def far_threshold(validation_rows,target):
    if not 0<=target<1: raise ValueError('FAR target must be in [0,1)')
    opened=[r for r in validation_rows if r['scenario']=='open']
    if not opened: raise ValueError('Validation has no no-match cases')
    scores=sorted(set(float(r['score']) for r in opened if r['score'] is not None))
    if not scores: return 0.
    thresholds=[math.nextafter(scores[0],-math.inf)]+[math.nextafter(s,math.inf) for s in scores]
    for t in thresholds:
        count=sum(r['score'] is not None and r['score']>=t for r in opened)
        if count/len(opened)<=target+1e-12: return t
    raise AssertionError('Reject-all threshold must satisfy budget')


def paired_counts(before,after):
    a={r['case_id']:r for r in before if r['scenario']=='closed'}
    b={r['case_id']:r for r in after if r['scenario']=='closed'}
    if set(a)!=set(b) or len(a)!=sum(r['scenario']=='closed' for r in before):
        raise ValueError('Paired query IDs differ or duplicate')
    counts={'gained':0,'lost':0,'both_correct':0,'both_wrong':0}
    for key in a:
        x,y=a[key]['top_correct'],b[key]['top_correct']
        counts['both_correct' if x and y else 'lost' if x else 'gained' if y else 'both_wrong']+=1
    n=counts['gained']+counts['lost']; k=min(counts['gained'],counts['lost'])
    counts['mcnemar_exact_p']=min(1.,2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1.
    counts['queries']=len(a)
    return counts


def save_csv(path,rows):
    if not rows: return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def validate_scored_adapters(metadata,sft_hash,dpo_hash):
    if metadata['sft']['adapter_sha256']!=sft_hash or metadata['dpo']['adapter_sha256']!=dpo_hash:
        raise ValueError('On-disk SFT/DPO checkpoints differ from the scored models')


def table(rows,keys):
    text='| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'
    for row in rows:
        text+='| '+' | '.join(f'{row[k]:.4f}' if isinstance(row[k],float) else str(row[k]) for k in keys)+' |\n'
    return text


def load_arm(folder,data):
    result={}; metadata={}
    for split in ('val','test'):
        meta=read_json(folder/split/'complete.json'); rows=read_json(folder/split/'rows.json')
        cases=read_json(data/f'{split}.json'); truth={r['case_id']:r for r in cases}
        if meta['manifest_sha256']!=sha(data/'manifest.json') or meta['smoke_only'] or meta.get('smoke_checkpoint'):
            raise ValueError('Dataset mismatch or smoke result')
        if len(rows)!=len(truth) or {r['case_id'] for r in rows}!=set(truth):
            raise ValueError('Incomplete or duplicate case IDs')
        for row in rows:
            case=truth[row['case_id']]
            if row['scenario']!=case['scenario'] or row['has_positive']!=case['has_positive']:
                raise ValueError('Case labels changed')
            if row['top_correct']!=bool(row['top_identity'] and row['top_identity']==case['identity']):
                raise ValueError('Top correctness does not match frozen identity')
        result[split]=rows; metadata[split]=meta
    for key in ('adapter_sha256','pixels','training_mode'):
        if metadata['val'][key]!=metadata['test'][key]: raise ValueError('val/test model mismatch')
    result['meta']=metadata['test']
    return result


def analyze(args):
    root=Path(args.root); data=Path(args.data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    verify_manifest(data)
    arms={name:load_arm(root/name,data) for name in ('baseline','sft','dpo')}
    validate_lineage({k:v['meta'] for k,v in arms.items()})
    budget=None
    if args.control:
        control=Path(args.control); run=read_json(control/'control_adapter/run.json')
        dpo=read_json(root/'dpo_adapter/run.json')
        parent=adapter_hash(root/'sft_adapter')
        validate_scored_adapters({k:v['meta'] for k,v in arms.items()},parent,adapter_hash(root/'dpo_adapter'))
        if run['mode']!='continue_sft' or run.get('smoke_only') or run['initial_adapter_sha256']!=parent:
            raise ValueError('Continued SFT has incorrect parent/mode')
        if dpo['reference_adapter_sha256']!=parent: raise ValueError('DPO parent mismatch')
        keys=('steps','accum','seed','lr','pixels','training_pairs','manifest_sha256')
        if any(run[k]!=dpo[k] for k in keys): raise ValueError('Control update/sample settings differ from DPO')
        arms['continue_sft']=load_arm(control/'continue_sft',data)
        if arms['continue_sft']['meta']['adapter_sha256']!=adapter_hash(control/'control_adapter'):
            raise ValueError('Control scores use another checkpoint')
        budget={k:run[k] for k in keys}
        budget.update(control_wall_seconds=run['wall_seconds'],dpo_wall_seconds=dpo['wall_seconds'],
            control_peak_mib=run['peak_allocated_mib'],dpo_peak_mib=dpo['peak_allocated_mib'])
    if len({v['meta']['pixels'] for v in arms.values()})!=1: raise ValueError('Resolution mismatch')
    original=[]; matched=[]
    for arm,value in arms.items():
        t=calibrate(value['val']); original.append(dict(arm=arm,threshold=t,**metrics(value['test'],t)))
        for target in (.05,.10,.20):
            threshold=far_threshold(value['val'],target)
            val=metrics(value['val'],threshold); test=metrics(value['test'],threshold)
            matched.append({'arm':arm,'validation_far_target':target,'threshold':threshold,
                'validation_far_actual':val['false_accept_rate'],'test_far_actual':test['false_accept_rate'],
                'test_correct_accept_rate':test['accepted_correct_rate'],
                'test_correct_accept_count':round(test['accepted_correct_rate']*test['closed_queries']),
                'test_false_accept_count':round(test['false_accept_rate']*test['no_match_queries'])})
    comparisons=[('baseline','sft'),('sft','dpo')]+([('sft','continue_sft'),('continue_sft','dpo')] if args.control else [])
    paired=[]; changes=[]
    truth={r['case_id']:r for r in read_json(data/'test.json')}
    for left,right in comparisons:
        paired.append(dict(before=left,after=right,**paired_counts(arms[left]['test'],arms[right]['test'])))
        a={r['case_id']:r for r in arms[left]['test']}; b={r['case_id']:r for r in arms[right]['test']}
        ta=calibrate(arms[left]['val']); tb=calibrate(arms[right]['val'])
        for case_id in a:
            x,y=a[case_id],b[case_id]; case=truth[case_id]
            accept_a=x['score'] is not None and x['score']>=ta
            accept_b=y['score'] is not None and y['score']>=tb
            kind=('gained' if y['top_correct'] else 'lost') if x['top_correct']!=y['top_correct'] else 'unchanged'
            changes.append({'before':left,'after':right,'case_id':case_id,'category':case['category'],
                'scenario':case['scenario'],'ranking_change':kind,'before_top':x['top_identity'],'after_top':y['top_identity'],
                'true_identity':case['identity'],'before_score':x['score'],'after_score':y['score'],
                'before_accepted':accept_a,'after_accepted':accept_b,'query_image':case['image_path']})
    save_csv(out/'query_changes.csv',changes); save_csv(out/'matched_far.csv',matched)
    save_csv(out/'operating_points.csv',original); save_csv(out/'paired_summary.csv',paired)
    write_json(out/'analysis.json',{'paired':paired,'matched_far':matched,'original_operating_points':original,
                                 'matched_update_budget':budget,'manifest_sha256':sha(data/'manifest.json')})
    text='# 后训练三项补充实验\n\n'
    text+='所有比例为0–1。复用冻结数据和已有三组结果；阈值只使用验证集。真实测试FAR可能不同，不能称为在测试集严格等FAR。配对检验为探索性，单随机种子，不代表重复训练的稳定性。阈值显示四舍五入；并列分数使用略高于边界的阈值，精确值以CSV/JSON为准。\n\n'
    text+='## 逐查询变化（不施加拒绝阈值）\n\n'+table(paired,list(paired[0]))
    text+='\n## 原验证集策略的操作点\n\n'+table(original,['arm','threshold','conditional_top1','retrieval_top1','accepted_correct_rate','false_accept_rate'])
    text+='\n## 相同验证集FAR目标\n\n'+table(matched,list(matched[0]))
    if budget:
        text+='\n## 继续 SFT 对照预算\n\n'
        text+=f"同一个 SFT 检查点出发，双方额外 {budget['steps']} 个优化步、梯度累积 {budget['accum']}，学习率 {budget['lr']}，相同种子/样本顺序/分辨率，重新初始化优化器。\n\n"
        text+=f"继续SFT训练 {budget['control_wall_seconds']:.1f}s；DPO训练（含参考概率计算）{budget['dpo_wall_seconds']:.1f}s。这是等优化步数/样本预算，不是等FLOPs或等时长。\n"
    text+='\n## 排序变化的具体查询\n\n'
    selected=[r for r in changes if r['ranking_change']!='unchanged' and r['scenario']=='closed']
    if selected: text+=table(selected,['before','after','case_id','category','ranking_change','before_top','after_top','true_identity'])
    text+='\n完整逐查询路径、分数及接受状态见 query_changes.csv。\n'
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    print(text,flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--data',required=True)
    p.add_argument('--out',required=True); p.add_argument('--control')
    analyze(p.parse_args())


if __name__=='__main__': main()
