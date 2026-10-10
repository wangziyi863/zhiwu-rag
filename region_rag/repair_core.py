"""CPU-only repair eligibility, geometry and evidence-image checks."""
import re
import numpy as np
from PIL import Image
from .core import TARGETS, parse_intent, localization_question, validate_bbox
from .ranking import choose_categories


def repair_target(question, selected):
    text=localization_question(question).lower()
    found=[name for name,(aliases,_) in TARGETS.items() if any(
        re.search(r'\b'+re.escape(a)+r's?\b',text) if a.isascii() else a in text for a in aliases)]
    names=choose_categories(selected.get('region',{}).get('class_name',''),{})
    if len(found)>1:
        if re.match(r'\s*(比较|对比|compare)',text): return None
        if names and names[0] in found and re.search(r'旁|边|之间|前|后|靠|\b(near|beside|behind|next to)\b',text):
            return names[0]
        return None
    if found: return found[0]
    return names[0] if names else None


def has_selector(question):
    intent=parse_intent(question)
    return bool(intent.get('position') or re.search(
        r'左|右|前|后|中间|靠|旁|边|第|颜色|红色|蓝色|绿色|黄色|黑色|白色|棕色|褐色|粉色|紫色|灰色|'
        r'条纹|格纹|花纹|扶手|高背|\b(left|right|front|back|near|beside|behind|next to|nearest|closest|furthest|foreground|background|'
        r'black|white|red|blue|green|yellow|brown|pink|purple|gray|grey|striped|first|second|third|fourth|fifth|'
        r'sixth|seventh|eighth|ninth|tenth|last|\d+(?:st|nd|rd|th))\b',localization_question(question),re.I))


def should_repair(question, selected, check):
    if check and check.get('status')=='verified': return False
    if not repair_target(question,selected): return False
    scene=selected.get('instance_assessment',{}).get('scene_instance_count')
    if selected.get('status')=='uncertain' and scene in (0,'0'):
        return True  # Independent detection may recover a missed known category.
    count=(check or {}).get('assessment',{}).get('instance_count')
    if not has_selector(question):
        return selected.get('status')=='selected' and scene in (1,'1') and (
            count in (1,'1') or (check or {}).get('method')=='geometric_merge_suspicion')
    return True


def suspected_merge(region, regions):
    """Conservative suspicion only; a fresh detector and mask check must confirm it."""
    target=choose_categories(region['class_name'],{})
    if not target: return []
    x1,y1,x2,y2=region['bbox']; area=max(1,(x2-x1)*(y2-y1)); parts=[]
    for other in regions:
        if other.get('id')==region.get('id'): continue
        category=choose_categories(other['class_name'],{})
        a,b,c,d=other['bbox']; subarea=max(1,(c-a)*(d-b))
        overlap=max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))
        # Semantic chair/stool confusion can hide separate seats. A substantial,
        # displaced overlapping seat warrants inspection, not an instance claim.
        displacement=max(abs((a+c-x1-x2)/2)/max(1,x2-x1),
                         abs((b+d-y1-y2)/2)/max(1,y2-y1))
        if (target[0] in ('chair','stool') and category and category[0] in ('chair','stool')
                and category!=target and .2<=subarea/area<=1.5
                and overlap/min(area,subarea)>=.3 and displacement>=.3):
            return [region['id'],other['id']]
        if category!=target: continue
        if .10<=subarea/area<=.75 and overlap/subarea>=.5:
            parts.append(other)
    for i,left in enumerate(parts):
        a,b,c,d=left['bbox']
        for right in parts[i+1:]:
            e,f,g,h=right['bbox']
            overlap=max(0,min(c,g)-max(a,e))*max(0,min(d,h)-max(b,f))
            if overlap/max(1,min((c-a)*(d-b),(g-e)*(h-f)))<.10:
                return [left['id'],right['id']]
    return []


def mask_region(mask, detector_box, quality, category, region_id, mask_path):
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2 or mask.sum()<64 or not np.isfinite(quality) or quality<.55:
        raise ValueError('分割掩码为空、过小或质量不足。')
    h,w=mask.shape
    x1,y1,x2,y2=validate_bbox(detector_box,(w,h))
    # Allow small segmentation extensions beyond an imperfect detector box.
    px,py=max(3,round((x2-x1)*.1)),max(3,round((y2-y1)*.1))
    inside=mask[max(0,y1-py):min(h,y2+py),max(0,x1-px):min(w,x2+px)].sum()
    if inside/mask.sum()<.8: raise ValueError('掩码大量超出实例检测框。')
    ys,xs=np.nonzero(mask)
    return {'id':region_id,'class_name':category,'bbox':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
            'area':int(mask.sum()),'mask_path':str(mask_path),'sam_quality':float(quality),
            'detector_bbox':list(detector_box),'origin':'grounding_dino_sam2'}


def masked_crop(image, mask, bbox):
    mask=np.asarray(mask,dtype=bool)
    if mask.shape!=(image.height,image.width) or not mask.any():
        raise ValueError('实例掩码与原图坐标不一致或为空。')
    box=validate_bbox(bbox,image.size)
    x1,y1,x2,y2=box
    if not mask[y1:y2,x1:x2].any(): raise ValueError('目标框没有实例像素。')
    pixels=np.asarray(image.convert('RGB')).copy()
    pixels[~mask]=127
    return Image.fromarray(pixels).crop(box)


def mask_overlay(image, mask):
    pixels=np.asarray(image.convert('RGB')).copy()
    mask=np.asarray(mask,dtype=bool)
    if mask.shape!=pixels.shape[:2]: raise ValueError('掩码坐标不一致。')
    pixels[mask]=(pixels[mask]*.6+np.array([0,220,160])*.4).astype(np.uint8)
    return Image.fromarray(pixels)
