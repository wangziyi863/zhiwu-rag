"""Target parsing, geometric selection and evidence validation; no model loading."""
import math
import re

import numpy as np
from scipy import ndimage


TARGETS = {
    'chair': (('椅子', '椅', 'chair'), ('chair', 'armchair', 'swivelchair', 'seat')),
    'sofa': (('沙发', 'sofa', 'couch'), ('sofa',)),
    'table': (('桌子', '桌', 'table', 'desk'), ('table', 'desk', 'coffeetable')),
    'bed': (('床', 'bed'), ('bed',)),
    'cabinet': (('柜子', '柜', 'cabinet'), ('cabinet', 'wardrobe', 'chestofdrawers', 'bookcase')),
    'lamp': (('台灯', '灯', 'lamp'), ('lamp', 'light', 'chandelier', 'sconce')),
    'shelf': (('书架', '架子', 'shelf'), ('shelf', 'bookcase')),
    'plant': (('盆栽', '植物', 'plant'), ('plant', 'flower', 'palm')),
    'refrigerator': (('冰箱', 'fridge', 'refrigerator'), ('refrigerator',)),
    'television': (('电视', 'television', 'tv'), ('televisionreceiver',)),
    'stool': (('凳子', '凳', 'stool'), ('stool', 'ottoman')),
    'vase': (('花瓶', 'vase'), ('vase',)),
    'mirror': (('镜子', 'mirror'), ('mirror',)),
    'rug': (('地毯', 'rug', 'carpet'), ('rug',)),
    'pillow': (('枕头', '抱枕', 'pillow', 'cushion'), ('pillow', 'cushion')),
}


def localization_question(question):
    """Remove a trailing information request, retaining the actual object description."""
    question=re.sub(r'^(?:请)?(?:告诉我|介绍一下|查询一下)\s*','',question)
    question=re.sub(r'的(?:材质|尺寸|价格|品牌|型号)(?:是什么|是多少)?[？?。]*$','',question)
    brief = re.split(r'[，,；;]\s*(?:并|然后)?(?:比较|对比|介绍|查询|查看|告诉我)'
                     r'(?=(?:一下)?(?:候选|商品|物品|产品|资料|材质|尺寸|价格|型号|品牌|颜色|检索|它们|这些))',
                     question,maxsplit=1)[0]
    brief = re.sub(r'(?:的)?(?:相似|类似)(?:商品|物品|产品)', '', brief)
    brief = re.sub(r'(?:是|的)?(?:什么材质|尺寸是多少|价格是多少)[？?。]*$', '', brief)
    brief = re.split(r'\s+and\s+(?:compare|describe|report|tell me)\s+(?:the\s+)?'
                     r'(?=materials?|fabric|dimensions?|sizes?|prices?|brands?|products?|candidates?)',
                     brief,maxsplit=1,flags=re.I)[0]
    return brief.strip(' ，,；;。') or question


def parse_intent(question):
    question = str(question).strip()
    if not question or len(question) > 2000:
        raise ValueError('问题不能为空，且最多 2000 字符。')
    text = question.lower()
    targets = []
    for target, (aliases, classes) in TARGETS.items():
        if any((re.search(r'\b' + re.escape(alias) + r's?\b', text) if alias.isascii() else alias in text) for alias in aliases):
            targets.append((target, classes))
    directions = {
        'left': ('左', 'left'), 'right': ('右', 'right'),
        'top': ('上方', '上面', 'top', 'upper'), 'bottom': ('下方', '下面', 'bottom', 'lower'),
        'center': ('中间', '中央', 'center', 'middle'),
    }
    pos = [key for key, words in directions.items()
           if any(re.search(r'\b' + re.escape(w) + r'\b', text) if w.isascii() else w in text
                  for w in words)]
    complex_query = re.search(
        r'不要|不是|除了|并非|第[一二三四五六七八九\d]|红色|蓝色|绿色|黑色|白色|黄色|'
        r'棕色|褐色|粉色|紫色|灰色|条纹|格纹|花纹|靠近|旁边|前面|后面|最近|最远|'
        r'\b(not|except|red|blue|green|black|white|brown|pink|purple|gray|grey|striped|'
        r'near|closest|furthest|foreground|background|first|second|third|fourth|fifth|sixth|'
        r'seventh|eighth|ninth|tenth|last|\d+(?:st|nd|rd|th))\b', text)
    if len(targets) != 1 or len(pos) > 1 or complex_query:
        return {'status': 'needs_model', 'question': question, 'classes': [], 'position': None}
    return {'status': 'parsed', 'question': question, 'target': targets[0][0],
            'classes': list(targets[0][1]), 'position': pos[0] if pos else None}


def validate_bbox(box, size):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError('边界框必须包含 x1,y1,x2,y2。')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        raise ValueError('边界框坐标必须是有限数字。')
    x1, y1, x2, y2 = box
    w, h = size
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('边界框坐标越界或为空。')
    out = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
    return out


def extract_regions(labels, names, allowed=None, min_area=100, max_regions=60):
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError('分割类别图必须是二维数组。')
    regions = []
    for class_id in np.unique(labels):
        if not 0 <= class_id < len(names):
            continue
        name = names[int(class_id)].split(';')[0].strip()
        if allowed is not None and name not in allowed:
            continue
        components, _ = ndimage.label(labels == class_id)
        counts = np.bincount(components.ravel())
        for component_id, sl in enumerate(ndimage.find_objects(components), start=1):
            if sl is None or counts[component_id] < min_area:
                continue
            ys, xs = sl
            regions.append({'class_name': name, 'class_id': int(class_id),
                            'bbox': [xs.start, ys.start, xs.stop, ys.stop],
                            'area': int(counts[component_id]), 'origin': 'gla_connected_component'})
    regions.sort(key=lambda r: (-r['area'], r['bbox'][0], r['bbox'][1]))
    return [dict(r, id=f'R{i+1}') for i, r in enumerate(regions[:max_regions])]


def select_region(regions, intent, image_size):
    if intent.get('status') != 'parsed':
        return {'status': 'ambiguous', 'reason': '目标描述需要视觉语言模型解析。'}
    candidates = [r for r in regions if r['class_name'] in intent['classes']]
    if not candidates:
        return {'status': 'not_found', 'reason': '分割结果中没有找到匹配目标。'}
    w, h = image_size
    pos = intent.get('position')
    if not pos:
        if len(candidates) == 1:
            return {'status': 'selected', 'region': candidates[0], 'method': 'unique_category'}
        return {'status': 'ambiguous', 'reason': '存在多个同类候选，问题未明确区分。'}
    def score(r):
        x1, y1, x2, y2 = r['bbox']
        x, y = (x1+x2)/(2*w), (y1+y2)/(2*h)
        return {'left': x, 'right': 1-x, 'top': y, 'bottom': 1-y,
                'center': math.hypot(x-.5, y-.5)}[pos]
    candidates.sort(key=score)
    first = candidates[0]
    x1, y1, x2, y2 = first['bbox']
    broad = ((pos in ('left', 'right') and (x2-x1)/w > .6) or
             (pos in ('top', 'bottom') and (y2-y1)/h > .6))
    tied = len(candidates) > 1 and score(candidates[1]) - score(first) < .08
    if broad or tied or score(first) > .65:
        return {'status': 'ambiguous', 'reason': '候选可能粘连，或空间条件不足以可靠区分。'}
    return {'status': 'selected', 'region': first, 'method': 'category_and_position'}


def validate_answer(result, evidence_ids):
    if not isinstance(result, dict) or not isinstance(result.get('answer'), str) or not result['answer'].strip():
        raise ValueError('问答模型未返回有效正文。')
    citations = result.get('citations', [])
    if not isinstance(citations, list) or any(not isinstance(c, str) for c in citations):
        raise ValueError('引用必须为 ID 数组。')
    allowed = set(evidence_ids) | {'R1'}
    inline = set(re.findall(r'\b[ER]\d+\b', result['answer']))
    if (set(citations) | inline) - allowed:
        raise ValueError('回答引用了本次证据中不存在的 ID。')
    return {'answer': result['answer'].strip(), 'citations': sorted(set(citations) | inline)}
