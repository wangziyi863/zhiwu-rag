"""Validate model-generated JSON before accepting automatic localization."""
import json
import re

from .core import validate_bbox


def canvas_geometry(size, side=504):
    w,h = size
    scale = min(side/w,side/h)
    rw,rh = round(w*scale),round(h*scale)
    return rw,rh,(side-rw)//2,(side-rh)//2


def canvas_bbox_to_normalized(box, size, side=504):
    x1,y1,x2,y2 = validate_bbox(box,(side,side))
    rw,rh,ox,oy = canvas_geometry(size,side)
    # The model may include a small border, but a box entirely in padding is invalid.
    box=[max(0,x1-ox),max(0,y1-oy),min(rw,x2-ox),min(rh,y2-oy)]
    box=validate_bbox(box,(rw,rh))
    return [round(v*1000/(rw if n%2==0 else rh)) for n,v in enumerate(box)]


def parse_json_output(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    value = json.loads(text)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError('模型输出必须是 JSON 对象。')
    return value


def resolve_selection(value, candidates, size):
    if value.get('status') != 'selected':
        return {'status': 'uncertain', 'reason': str(value.get('reason', '无法可靠定位问题中的目标。'))}
    region_id = value.get('region_id')
    if region_id:
        matches = [r for r in candidates if r['id'] == region_id]
        if len(matches) != 1:
            raise ValueError('定位模型选择了不存在的区域。')
        region = dict(matches[0])
    else:
        box = value.get('bbox_1000')
        normalized = validate_bbox(box, (1000, 1000))
        w, h = size
        pixels = [normalized[0]*w/1000, normalized[1]*h/1000,
                  normalized[2]*w/1000, normalized[3]*h/1000]
        region = {'id': 'V1', 'class_name': str(value.get('target', 'object'))[:80],
                  'bbox': validate_bbox(pixels, size), 'origin': 'vlm_proposed_box'}
        if (region['bbox'][2]-region['bbox'][0]) < 4 or (region['bbox'][3]-region['bbox'][1]) < 4:
            raise ValueError('模型提出的目标框过小。')
    return {'status': 'selected', 'region': region, 'method': 'vlm_disambiguation',
            'reason': str(value.get('reason', '视觉语言模型辅助定位；未经人工确认。'))}


def instance_count(value):
    # Models sometimes serialize an integer as "1". Accept exact ASCII integer
    # strings only, never booleans, fractions, ranges or uncertain prose.
    if isinstance(value, str) and re.fullmatch(r'0|[1-9][0-9]{0,3}', value):
        value = int(value)
    return value if type(value) is int and 0 <= value <= 1000 else None


def resolve_instance_selection(value, candidates, size, intent):
    """A semantic component or a valid box is not proof of a unique instance."""
    count = instance_count(value.get('scene_instance_count'))
    if count == 0:
        return {'status':'uncertain','reason':'视觉模型暂未识别出所指物体，不能据此判断图中没有它。',
                'instance_assessment':value}
    if count is None or count < 1 or value.get('question_resolves_target') is not True:
        return {'status':'uncertain', 'reason':'无法确认问题唯一指向一个物体，请补充位置或外观描述。',
                'instance_assessment':value}
    if count > 1 and intent.get('status') == 'parsed' and not intent.get('position'):
        return {'status':'uncertain', 'reason':f'图中检测到多个同类物体（模型估计 {count} 个），“这个”不足以区分。请说明左边、右边或其他位置。',
                'instance_assessment':value}
    selected = resolve_selection(value, candidates, size)
    selected['instance_assessment'] = value
    if selected['status'] == 'selected':
        selected['requires_instance_verification'] = True
    return selected


def validate_instance_check(value):
    """Fail closed on incomplete, malformed or negative visual verification."""
    valid = (instance_count(value.get('instance_count')) == 1
             and value.get('matches_question') is True
             and value.get('single_complete_target') is True)
    return {'status':'verified' if valid else 'uncertain', 'assessment':value,
            'reason':('视觉模型检查通过：裁剪内为一个完整目标。' if valid else
                      '裁剪区域可能含多个物体、目标不完整或无法确认匹配，本次未继续检索。请补充目标位置。')}


def evidence_fields(hit):
    """Full fields avoid extractive answers silently dropping a negation."""
    fields = {}
    if hit.get('description'):
        fields['description'] = str(hit['description'])
    for key, value in hit.get('attributes', {}).items():
        if value is not None and str(value).strip():
            fields[f'attributes.{key}'] = str(value)
    return fields


def grounded_answer(value, hits):
    """Render only source-verified extracts; no unchecked generated prose."""
    selected = value.get('evidence')
    if not isinstance(selected, list) or len(selected) > 6:
        raise ValueError('回答必须包含最多六条资料字段。')
    sources = {hit['evidence_id']: evidence_fields(hit) for hit in hits}
    lines, citations, seen = [], {'R1'}, set()
    for entry in selected:
        if not isinstance(entry, dict):
            raise ValueError('资料选择格式无效。')
        key, field, quote = entry.get('id'), entry.get('field'), entry.get('quote')
        if not isinstance(key, str) or not isinstance(field, str):
            raise ValueError('资料引用字段必须是文本。')
        if key not in sources or field not in sources[key]:
            raise ValueError('模型选择了不存在的资料字段。')
        if 'quote' in entry and quote != sources[key][field]:
            raise ValueError('生成片段与资料原文不一致。')
        quote = sources[key][field]
        if (key, field) in seen:
            continue
        seen.add((key, field))
        label = '介绍' if field == 'description' else field.split('.', 1)[1]
        lines.append(f'候选 [{key}] 的{label}（资料原文）：{quote}')
        citations.add(key)
    answer = '系统自动定位的目标见 [R1]。检索结果只是相似候选。\n'
    answer += '\n'.join(lines) if lines else '未选出可用于回答的属性资料。'
    answer += '\n这些资料不能确认图中目标的同款身份、真实材质或尺寸；缺少的属性保持未知。'
    return {'status': 'answered', 'answer': answer, 'citations': sorted(citations),
            'answer_mode': 'validated_source_excerpts'}


def answer_choices(hits, question=''):
    choices=[]
    requested=set()
    groups=[]
    for pattern,keys in [
        (r'材质|面料|material|fabric',{'材质','面料','material','fabric'}),
        (r'尺寸|多大|多高|多宽|size|dimension|height|width|length',
         {'高度','宽度','长度','尺寸','height','width','length','depth','size','dimensions'}),
        (r'颜色|color|colour',{'颜色','color','colour'}),
        (r'品牌|brand',{'品牌','brand'}),
        (r'型号|model number',{'型号','model','model_number'})]:
        if re.search(pattern,question,re.I):
            requested.update(keys)
            groups.append(keys)
    for hit in hits:
        available={key.lower() for key in hit.get('attributes',{})}
        need_description=any(not available.intersection(group) for group in groups)
        for field,value in evidence_fields(hit).items():
            relevant=field.startswith('attributes.') and field.split('.',1)[1].lower() in requested
            if requested and not relevant and not (field=='description' and need_description):
                continue
            choices.append({'choice':len(choices)+1,'id':hit['evidence_id'],'field':field,'value':value[:1600]})
    return choices


def resolve_answer_choices(value, choices):
    selected=value.get('choices')
    if not isinstance(selected,list) or len(selected)>6:
        raise ValueError('回答必须选择最多六个资料编号。')
    lookup={entry['choice']:entry for entry in choices}
    result=[]
    for number in selected:
        if type(number) is not int or number not in lookup:
            raise ValueError('回答选择了不存在的资料编号。')
        result.append({key:lookup[number][key] for key in ('id','field')})
    return {'evidence':result}


def source_fallback(hits):
    selected = []
    for hit in hits:
        fields = evidence_fields(hit)
        if fields:
            field, quote = next(iter(fields.items()))
            selected.append({'id': hit['evidence_id'], 'field': field, 'quote': quote})
    result = grounded_answer({'evidence': selected[:6]}, hits)
    result['answer'] = '模型回答未通过来源校验，以下仅展示检索资料原文。\n' + result['answer']
    result.update(status='answered_with_fallback', answer_mode='source_excerpt_fallback')
    return result
