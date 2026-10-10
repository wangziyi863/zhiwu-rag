import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from region_rag.detail_core import mutual_region, rank_objective, choose_triplets, bind_config

class DetailTests(unittest.TestCase):
    def test_uninformative_matches_fall_back(self):
        f=np.ones((16,8),dtype=np.float32)
        self.assertIsNone(mutual_region(f,f,np.ones(16,bool),np.ones(16,bool),4))

    def test_matching_region_maps_different_locations(self):
        f=np.eye(16,dtype=np.float32)
        order=np.roll(np.arange(16).reshape(4,4),1,axis=1).ravel()
        g=f[order]
        m=mutual_region(f,g,np.ones(16,bool),np.ones(16,bool),4)
        self.assertIsNotNone(m)
        self.assertNotEqual(m['query_box'],m['candidate_box'])
        for box in (m['query_box'],m['candidate_box']):
            self.assertTrue(all(0<=x<=1 for x in box))

    def test_rank_gradient_promotes_positive_and_demotes_negative(self):
        pos=torch.tensor([-1.,-2.],requires_grad=True)
        neg=torch.tensor([-.2,-1.],requires_grad=True)
        loss=rank_objective(pos,neg,margin=.5,ce_weight=.2)
        loss.backward()
        self.assertLess(pos.grad[0].item(),0)
        self.assertGreater(neg.grad[0].item(),0)

    def test_mining_skips_ambiguous_negatives_and_picks_highest_remaining(self):
        pool=[{'query_id':'train_0','candidate_identity':'p','label':True,'images':['q','p']},
              {'query_id':'train_0','candidate_identity':'n1','label':False,'images':['q','n1'],'ambiguous':True},
              {'query_id':'train_0','candidate_identity':'n2','label':False,'images':['q','n2']},
              {'query_id':'train_0','candidate_identity':'n3','label':False,'images':['q','n3']}]
        rows=choose_triplets(pool,{'train_0|p':0.,'train_0|n1':10.,'train_0|n2':2.,'train_0|n3':1.})
        self.assertEqual(rows[0]['negative'],['q','n2'])

    def test_config_changes_cannot_reuse_old_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            bind_config(Path(tmp),{'pixels':128})
            with self.assertRaises(ValueError): bind_config(Path(tmp),{'pixels':256})

if __name__=='__main__': unittest.main()
