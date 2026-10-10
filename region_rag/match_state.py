"""Evidence-gated inference states; these are not trained three-class probabilities."""
import math
import re

FEATURES={'overall_shape','part_layout','part_count','connection_structure','surface_pattern','object_function'}
LABELS={'possible_match':'可能同款','clearly_different':'明显不同','insufficient_evidence':'证据不足'}


def uncertain(reason):
    return {'status':'insufficient_evidence','reason':reason,'observations':[],
            'identity_confirmed':False}


def validate_visual_comparison(value):
    if not isinstance(value,dict): return uncertain('视觉核验没有返回有效结果。')
    for key in ('target_unambiguous','comparison_possible','same_visible_design'):
        if type(value.get(key)) is not bool: return uncertain('视觉核验字段不完整，无法可靠判断。')
    observations=value.get('observations')
    if not isinstance(observations,list): return uncertain('缺少可核对的可见细节。')
    valid=[]
    for item in observations[:12]:
        if not isinstance(item,dict): continue
        if (item.get('feature') not in FEATURES or item.get('both_visible') is not True
                or item.get('relation') not in ('match','different')): continue
        if any(not isinstance(item.get(k),str) or not item[k].strip() or len(item[k])>250
               for k in ('query','candidate')): continue
        if any(re.fullmatch(r'图[12一二]可见细节|图[12一二]的?描述|query details|candidate details|visible details|\.\.\.|…',
                            item[k].strip(),re.I) for k in ('query','candidate')): continue
        if item['relation']=='different' and item['query'].strip()==item['candidate'].strip(): continue
        valid.append({k:item[k] for k in ('feature','query','candidate','both_visible','relation')})
    if not value['target_unambiguous']:
        return uncertain('对比图片中的目标不够明确，可能混入其他物体。')
    if not value['comparison_possible']:
        return uncertain('关键部位不可见或细节不足，当前图片不足以区分候选款式。')
    differences=[r for r in valid if r['relation']=='different']
    if differences:
        result={'status':'clearly_different','reason':'视觉核验发现两图可见结构或纹样存在差异。'}
    elif value['same_visible_design'] and valid:
        result={'status':'possible_match','reason':'可见设计细节相符，但图片不能确认精确商品身份。'}
    else:
        return uncertain('未获得足以支持同款或明确差异的可见细节。')
    result.update(observations=valid,identity_confirmed=False)
    return result


def ground_comparison(comparison,checks):
    """Only retain claims supported by separate single-image inspections."""
    observations=comparison.get('observations',[])
    if len(checks)!=2: return uncertain('单图证据复核未完成。')
    supported=[]
    for side,check in zip(('目标','候选'),checks):
        if not isinstance(check,dict): return uncertain('单图证据复核输出无效。')
        if check.get('single_target') is not True:
            return uncertain(side+'图中的主要物体不够明确，无法可靠比较款式。')
        if check.get('design_view_sufficient') is not True:
            return uncertain(side+'图缺少足够的整体结构或关键细节，不能只凭局部推断款式。')
        ids=check.get('visible_observation_ids')
        if (not isinstance(ids,list) or any(type(i) is not int or not 0<=i<len(observations) for i in ids)
                or len(set(ids))!=len(ids)):
            return uncertain('单图证据编号无效，不能采用本次观察。')
        supported.append(set(ids))
    shared=supported[0]&supported[1]
    result=validate_visual_comparison({'target_unambiguous':True,'comparison_possible':True,
        'same_visible_design':comparison.get('status')=='possible_match',
        'observations':[o for i,o in enumerate(observations) if i in shared]})
    if not shared: result=uncertain('两张图片分别复核后，没有共同支持的可靠比较依据。')
    result['grounding_method']='separate_single_image_checks'
    return result


def apply_match_state(ranked,hit,visual):
    result=dict(ranked)
    # Only the first ranked candidate is assessed; never generalize its verdict
    # to all recalled products, the entire gallery, or target localization.
    top=next((r for r in ranked.get('audit',[]) if r['item_id']==hit['item_id']),None)
    score=top.get('score') if top else None
    threshold=ranked.get('threshold')
    above=(isinstance(score,(int,float)) and math.isfinite(score) and
           isinstance(threshold,(int,float)) and math.isfinite(threshold) and score>=threshold)
    status=visual.get('status','insufficient_evidence')
    if status not in LABELS: visual=uncertain('核验状态无效。'); status=visual['status']
    if status=='possible_match' and not above:
        visual=dict(visual,status='insufficient_evidence',
                    reason='可见细节有相似之处，但同款重排分数未达到保留条件，两种判断尚不足以支持匹配。')
        status=visual['status']
    messages={
        'possible_match':'可能同款：首位候选通过分数筛选和可见细节核验，仍未确认精确商品身份。',
        'clearly_different':'明显不同（仅当前首位候选）：视觉核验发现可见差异，本次不将该候选作为匹配资料。其他候选尚未逐一核验。',
        'insufficient_evidence':'证据不足：当前无法可靠确认首位候选是否同款。本次不采用其属性回答；这不代表图中没有目标或图库中没有合适商品。'}
    decision=dict(visual,status=status,label=LABELS[status],message=messages[status],
                  scope='top_candidate_only',candidate_item_id=hit['item_id'],
                  candidate_evidence_id=hit.get('evidence_id'),candidate_name=hit.get('name','首位候选'),
                  dpo_score=score,dpo_threshold=threshold,identity_confirmed=False,
                  ranking_method=ranked.get('method','dpo_yes_no'),rank_score=score,rank_threshold=threshold,
                  method='rank_score_plus_visual_evidence_gate',calibrated_three_class=False)
    if ranked.get('calibration_matches_recall_k') is False:
        decision['calibration_note']='当前召回数量与原阈值校准设置不同，错误接受率尚未重新校准。'
    decision['calibration_scope']='当前阈值仅校准该重排模型的 top-1；新增视觉核验后的整体流程尚未校准。'
    hits=[]
    if status=='possible_match':
        # Preserve the DPO metadata and the original top-1-only acceptance policy.
        hits=[dict(h,match=decision) for h in ranked.get('hits',[]) if h['item_id']==hit['item_id']]
        if not hits: raise ValueError('Possible match is not the accepted DPO top candidate')
    result.update(match=decision,hits=hits,identity_confirmed=False,
                  status='ranked' if hits else status)
    return result
