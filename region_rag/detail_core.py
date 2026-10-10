"""Label-free local correspondence and reproducible ranking experiment helpers."""
import hashlib
from pathlib import Path
import numpy as np
from .posttrain_data import read_json,write_json


def pair_key(paths):
    return hashlib.sha256('\0'.join(map(str,paths)).encode()).hexdigest()


def bind_config(folder,config):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    path=folder/'config.json'
    if path.exists() and read_json(path)!=config:
        raise ValueError(f'Configuration changed; choose a new RUN directory: {folder}')
    write_json(path,config)


def mutual_region(q,k,qmask,kmask,grid=28,tau=.65,gap=.03):
    """Mutual best patches + distinctive match + local translation consensus.

    Coordinates refer to letterboxed images. This is visual correspondence,
    not a semantic part detector. All low-confidence cases return None.
    """
    q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8)
    k=k/np.maximum(np.linalg.norm(k,axis=1,keepdims=True),1e-8)
    s=q@k.T; s[~np.asarray(qmask,bool),:]=-2.; s[:,~np.asarray(kmask,bool)]=-2.
    best=s.argmax(1); reverse=s.argmax(0)
    sorted_s=np.sort(s,axis=1)
    valid=[i for i,j in enumerate(best) if reverse[j]==i and s[i,j]>=tau and sorted_s[i,-1]-sorted_s[i,-2]>=gap]
    if len(valid)<3: return None
    best_cluster=None
    for i in valid:
        j=int(best[i]); iy,ix=divmod(i,grid); jy,jx=divmod(j,grid)
        cluster=[]
        for a in valid:
            b=int(best[a]); ay,ax=divmod(a,grid); by,bx=divmod(b,grid)
            if (max(abs(ax-ix),abs(ay-iy))<=3 and max(abs(bx-jx),abs(by-jy))<=3
                    and max(abs((ax-ix)-(bx-jx)),abs((ay-iy)-(by-jy)))<=1):
                cluster.append((a,b))
        if len(cluster)>=3:
            value=float(np.mean([s[a,b] for a,b in cluster]))
            if best_cluster is None or (len(cluster),value)>best_cluster[0]:
                best_cluster=((len(cluster),value),i,j)
    if best_cluster is None: return None
    (_,confidence),i,j=best_cluster
    def box(idx):
        y,x=divmod(idx,grid); half=max(1,grid//5)
        return [max(0,(x-half)/grid),max(0,(y-half)/grid),min(1,(x+half+1)/grid),min(1,(y+half+1)/grid)]
    return {'query_box':box(i),'candidate_box':box(j),'confidence':confidence,
            'support':best_cluster[0][0]}


def rank_objective(pos,neg,margin=.5,ce_weight=.2):
    import torch.nn.functional as F
    ps=pos[0]-pos[1]; ns=neg[0]-neg[1]
    return F.softplus(margin+ns-ps)+ce_weight*(-pos[0]-neg[1])/2


def choose_triplets(pool,scores):
    groups={}
    for p in pool:
        if not p['query_id'].startswith('train_'): raise ValueError('Mining requires training queries only')
        groups.setdefault(p['query_id'],[]).append(p)
    result=[]
    for qid,rows in sorted(groups.items()):
        positives=[p for p in rows if p['label']]
        negatives=[p for p in rows if not p['label'] and not p.get('ambiguous')]
        if len(positives)!=1 or not negatives: continue
        neg=max(negatives,key=lambda p:(scores[qid+'|'+p['candidate_identity']],p['candidate_identity']))
        result.append({'query_id':qid,'positive':positives[0]['images'],'negative':neg['images'],
                       'positive_identity':positives[0]['candidate_identity'],'negative_identity':neg['candidate_identity']})
    if not result: raise ValueError('No usable training triplets')
    return result
