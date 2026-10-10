"""Category routing and validated VLM reranking, separate from CLIP recall."""
from .core import TARGETS

NEIGHBORS = {'chair':{'stool','sofa'},'stool':{'chair'},'sofa':{'chair'},
             'cabinet':{'shelf'},'shelf':{'cabinet'}}
CATALOG_ALIASES = {
    'bookcase':'shelf','bookshelf':'shelf','couch':'sofa','carpet':'rug',
    'coffee table':'table','dining table':'table','side table':'table','dining chair':'chair',
    'office chair':'chair','desk chair':'chair','computer chair':'chair','swivel chair':'chair',
    'executive chair':'chair','arm chair':'chair','bar stool':'stool','footstool':'stool',
    'dresser':'cabinet','chest of drawers':'cabinet','cupboard':'cabinet','nightstand':'cabinet',
    'bed frame':'bed','floor lamp':'lamp','table lamp':'lamp','standing lamp':'lamp',
    'throw pillow':'pillow','cushion':'pillow',
}


def choose_categories(target, scores):
    target = str(target).lower().strip()
    primary = CATALOG_ALIASES.get(target)
    if primary is None:
        primary = next((key for key,(words,aliases) in TARGETS.items()
                        if target == key or target in aliases or target in words),None)
    if primary is None:
        return []  # Unknown categories must not fall back to unrestricted furniture.
    result=[primary]
    ordered=sorted(scores,key=scores.get,reverse=True)
    for category in ordered:
        if category in NEIGHBORS.get(primary,set()) and scores[category]>=scores.get(primary,1)-.02:
            result.append(category)
            break
    return result


def validate_pair(value):
    if not isinstance(value,dict) or type(value.get('same_category')) is not bool:
        raise ValueError('重排缺少可靠的类别判断。')
    for key, upper in [('visual_score',4),('request_score',2)]:
        if type(value.get(key)) is not int or not 0<=value[key]<=upper:
            raise ValueError('重排分数格式无效。')
    return {'accepted':value['same_category'] and value['visual_score']>=2 and value['request_score']>=1,
            'visual_score':value['visual_score'],'request_score':value['request_score'],
            'score':2*value['visual_score']+value['request_score']}


def rank_candidates(hits, assessments, top_k=3):
    if len(hits)!=len(assessments):
        raise ValueError('重排结果与召回候选数不一致。')
    accepted=[dict(hit,rerank=assessment,recall_rank=n+1) for n,(hit,assessment) in enumerate(zip(hits,assessments))
              if assessment.get('accepted') is True]
    return sorted(accepted,key=lambda h:(-h['rerank']['score'],-h['similarity']))[:top_k]
