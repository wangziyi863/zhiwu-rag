"""Auditable category prompts and budget-matched recall with global rescue."""
import numpy as np
import re
from .core import TARGETS, localization_question
from .ranking import choose_categories

CATEGORY_DESCRIPTIONS={
 'chair':('chair','dining chair','office chair','armchair'),
 'stool':('stool','bar stool','upholstered footstool','storage ottoman'),
 'sofa':('sofa','couch','loveseat sofa','sectional sofa'),
 'table':('table','dining table','coffee table','side table'),
 'bed':('bed','bed frame','platform bed','bunk bed'),
 'cabinet':('cabinet','storage cupboard','chest of drawers','dresser'),
 'lamp':('lamp','table lamp','floor lamp','pendant light'),
 'shelf':('shelf','bookshelf','bookcase','shelving unit'),
 'plant':('plant','potted plant','houseplant','potted palm'),
 'refrigerator':('refrigerator','fridge','mini fridge','double door refrigerator'),
 'television':('television','flat screen television','television set','wall mounted television'),
 'vase':('vase','flower vase','ceramic vase','glass vase'),
 'mirror':('mirror','wall mirror','standing mirror','vanity mirror'),
 'rug':('rug','carpet','area rug','floor rug'),
 'pillow':('pillow','throw pillow','cushion','decorative pillow'),
}

def resolve_categories(target,scores,question='',detailed=False):
    categories=choose_categories(target,scores)
    if detailed and question:
        text=localization_question(question).lower()
        named=[c for c,(words,_) in TARGETS.items() if any(
            re.search(r'\b'+re.escape(w)+r's?\b',text) if w.isascii() else w in text for w in words)]
        # Appearance/ordinal complexity does not negate a unique category name.
        # Negation and multiple named categories remain ambiguous here.
        if len(named)==1 and not re.search(r'不要|不是|除了|并非|\b(not|except|without)\b',text):
            for c in choose_categories(named[0],{}):
                if c not in categories: categories.append(c)
    return categories

def select_candidates(items,scores,categories,k=12,threshold=.2,dual=False):
    if type(k) is not int or not 1<=k<=100: raise ValueError('召回数量须为 1–100。')
    scores=np.asarray(scores,dtype=float)
    if scores.shape!=(len(items),) or not np.isfinite(scores).all(): raise ValueError('候选分数无效。')
    if not categories: return []
    order=np.argsort(-scores,kind='stable'); selected=[]; seen=set()
    def take(indices,limit,branch):
        for j in indices:
            if len(selected)>=limit: break
            item=items[int(j)]
            identity=(item.get('provenance') or {}).get('item_id') or item['item_id']
            if identity in seen or scores[j]<threshold: continue
            seen.add(identity); selected.append((int(j),branch))
    primary=[j for j in order if items[int(j)].get('category') in categories]
    reserve=max(1,k//3) if dual else 0
    take(primary,k-reserve,'category')
    if dual: take(order,k,'global')
    selected.sort(key=lambda jb:(-scores[jb[0]],jb[0]))
    return [dict(items[j],similarity=float(scores[j]),recall_branch=branch,evidence_id=f'E{n+1}')
            for n,(j,branch) in enumerate(selected)]
