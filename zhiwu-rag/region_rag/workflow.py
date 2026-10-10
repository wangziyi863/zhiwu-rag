"""Separate target localization, retrieval outcome and sourced attributes."""
import re
from .core import localization_question

ATTRIBUTES={
 'material':('材质',r'材质|面料|material|fabric',{'材质','面料','material','fabric'}),
 'dimensions':('尺寸',r'尺寸|多大|多高|多宽|size|dimension|height|width|length',
               {'高度','宽度','长度','尺寸','height','width','length','depth','size','dimensions'}),
 'color':('颜色',r'颜色|color|colour',{'颜色','color','colour'}),
 'brand':('品牌',r'品牌|brand',{'品牌','brand'}),
 'model':('型号',r'型号|model number',{'型号','model','model_number'}),
 'price':('价格',r'价格|多少钱|price|cost',{'价格','price','cost'}),
}

def split_request(question):
    return {'target_description':localization_question(question),
            'requested_attributes':[k for k,(_,pattern,_) in ATTRIBUTES.items() if re.search(pattern,question,re.I)],
            'original_question':question}

def summarize_stages(result):
    status=result.get('status'); located=bool(result.get('region'))
    stages={'localization':{'status':'located' if located else 'unverified',
        'message':'已定位并通过目标区域检查。' if located else '尚未确认目标区域。'},
        'retrieval':{'status':'not_run','message':'商品检索尚未执行。'},
        'knowledge':{'status':'not_checked','message':'属性资料尚未检查。'},
        'identity':{'status':'unconfirmed','message':'未确认上传物体与候选商品为同款，不能将候选属性认定为原物体属性。'}}
    if not located:
        raw=result.get('disambiguation',{}).get('instance_assessment',{})
        count=raw.get('scene_instance_count')
        if result.get('repair',{}).get('attempts'):
            stages['localization'].update(status='verification_failed',message='已尝试检测/修复，但目标区域仍未通过核验。')
        elif count==0:
            stages['localization'].update(status='not_recognized',message='视觉模型未识别出目标；这不代表图中没有该物体。')
        elif isinstance(count,int) and count>1:
            stages['localization'].update(status='ambiguous',message='模型发现多个候选，尚未确定你指哪一个。')
        return stages
    if status=='knowledge_unavailable':
        stages['retrieval']={'status':'knowledge_unavailable','message':'知识库尚未就绪，本次未执行商品检索。'}
        stages['knowledge']={'status':'unavailable','message':'当前无法从知识库确认属性。'}
        return stages
    if status=='error':
        phase=result.get('failure_stage')
        if phase=='reranking':
            stages['retrieval']={'status':'rerank_failed','message':'已召回候选，但重排执行失败，尚未得到可靠排序。'}
        elif phase=='answer':
            stages['retrieval']={'status':'candidates_found','message':'已获得通过重排的候选资料。'}
            stages['knowledge']={'status':'answer_failed','message':'资料回答执行失败，不表示目标未找到或属性不存在。'}
        else:
            stages['retrieval']={'status':'retrieval_failed','message':'目标已定位，但商品检索执行失败。'}
        return stages
    match=result.get('match')
    if match:
        stages['identity']={'status':match['status'],'message':match['message']}
        if not result.get('hits'):
            stages['retrieval']={'status':match['status'],'message':match['message']}
            stages['knowledge']={'status':'withheld_unverified_match',
                                 'message':'未采用该候选的属性资料，目标材质等信息仍未确认。'}
            return stages
    if result.get('hits'):
        stages['retrieval']={'status':'candidates_found','message':f"已检索到 {len(result['hits'])} 条通过重排的相似商品资料。"}
    elif 'recall' in result:
        rejected=bool(result['recall'].get('hits')) and 'reranking' in result
        if result['recall'].get('hits') and 'reranking' not in result:
            stages['retrieval']={'status':'rerank_pending','message':'已召回候选，尚未完成重排。'}
            return stages
        stages['retrieval']={'status':'rerank_rejected' if rejected else 'no_candidates',
            'message':'本次召回的候选均未通过重排。' if rejected else '本次检索未返回符合条件的候选。'}
        stages['retrieval']['message']+='这不代表图库中一定没有合适商品。'
        return stages
    else:
        return stages
    requested=result.get('intent',{}).get('requested_attributes',[])
    if not requested:
        stages['knowledge']={'status':'reference_only','message':'以下为候选商品的来源资料。'}
        return stages
    coverage={}
    for hit in result['hits']:
        keys={str(k).lower() for k,v in hit.get('attributes',{}).items()
              if v is not None and str(v).strip().lower() not in ('','unknown','n/a','none','未知','不详')}
        coverage[hit['evidence_id']]={a:bool(keys & ATTRIBUTES[a][2]) for a in requested}
    missing=[a for a in requested if not any(c[a] for c in coverage.values())]
    any_present=any(any(c.values()) for c in coverage.values())
    all_present=all(all(c.values()) for c in coverage.values())
    state='candidate_attributes_available' if all_present else 'partial_attributes' if any_present else 'attribute_unverified'
    message=('候选资料含有所问属性字段；只能按来源介绍候选商品。' if all_present else
             '部分候选属性字段缺失；缺失部分保持未知。' if any_present else
             '本次候选资料未找到结构化的'+ '、'.join(ATTRIBUTES[a][0] for a in requested)+'字段；介绍原文仍可供参考，尚不能自动确认所问属性。')
    stages['knowledge']={'status':state,'message':message,'coverage':coverage,'missing_attributes':missing}
    return stages
