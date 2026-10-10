import unittest
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch

class PosttrainTests(unittest.TestCase):
    def test_shared_images_join_identities_transitively(self):
        from region_rag.posttrain_data import identity_groups
        rows=[{'identity':'a','tokens':['x']},{'identity':'b','tokens':['x','y']},
              {'identity':'c','tokens':['y']},{'identity':'d','tokens':['z']}]
        g=identity_groups(rows)
        self.assertEqual(g['a'],g['c']); self.assertNotEqual(g['a'],g['d'])

    def test_shared_image_group_is_not_a_same_product_label(self):
        from region_rag.posttrain_data import eligible_gallery
        gallery=[{'identity':'a','group':'family1'}, {'identity':'b','group':'family1'},
                 {'identity':'c','group':'family2'}]
        q={'identity':'a','group':'family1'}
        self.assertEqual(eligible_gallery(gallery,q,'closed'),[0,2])
        self.assertEqual(eligible_gallery(gallery,q,'open'),[2])

    def test_metrics_distinguish_recall_rerank_and_no_match(self):
        from region_rag.posttrain_data import metrics
        rows=[{'scenario':'closed','has_positive':True,'top_correct':True,'score':2},
              {'scenario':'closed','has_positive':False,'top_correct':False,'score':1},
              {'scenario':'open','has_positive':False,'top_correct':False,'score':-2},
              {'scenario':'open','has_positive':False,'top_correct':False,'score':1}]
        m=metrics(rows,0)
        self.assertEqual(m['recall_at_k'],.5)
        self.assertEqual(m['conditional_top1'],1.)
        self.assertEqual(m['retrieval_top1'],.5)
        self.assertEqual(m['false_accept_rate'],.5)
        self.assertIsNone(metrics([],0)['conditional_top1'])

    def test_dpo_increases_preferred_token_probability(self):
        from region_rag.posttrain_model import dpo_loss
        chosen=torch.tensor(-1.,requires_grad=True); rejected=torch.tensor(-2.,requires_grad=True)
        loss=dpo_loss(chosen,rejected,torch.tensor(-1.),torch.tensor(-2.),.1)
        self.assertAlmostEqual(loss.item(),.693147,places=5)
        loss.backward()
        self.assertLess(chosen.grad.item(),0); self.assertGreater(rejected.grad.item(),0)

    def test_fixed_reference_changes_loss(self):
        from region_rag.posttrain_model import dpo_loss
        a=dpo_loss(torch.tensor(-1.),torch.tensor(-3.),torch.tensor(-1.),torch.tensor(-2.),.1)
        b=dpo_loss(torch.tensor(-1.),torch.tensor(-3.),torch.tensor(-1.),torch.tensor(-3.),.1)
        self.assertLess(a.item(),b.item())

    def test_prompt_does_not_expose_identity_or_labels(self):
        from region_rag.posttrain_model import messages
        m=messages()
        self.assertEqual(len([x for x in m[1]['content'] if x['type']=='image']),2)
        self.assertNotIn('item_id',str(m))

    def test_report_rejects_dpo_from_a_different_sft(self):
        from region_rag.posttrain_run import validate_lineage
        with self.assertRaises(ValueError):
            validate_lineage({'baseline':{'adapter_sha256':'base'},
                'sft':{'adapter_sha256':'sft_a','training_mode':'sft','smoke_checkpoint':False},
                'dpo':{'adapter_sha256':'dpo_a','training_mode':'dpo','reference_adapter_sha256':'sft_b','smoke_checkpoint':False}})

    def test_report_rejects_smoke_checkpoint_even_with_full_scores(self):
        from region_rag.posttrain_run import validate_lineage
        with self.assertRaises(ValueError):
            validate_lineage({'baseline':{'adapter_sha256':'base'},
                'sft':{'adapter_sha256':'a','training_mode':'sft','smoke_checkpoint':True},
                'dpo':{'adapter_sha256':'b','training_mode':'dpo','reference_adapter_sha256':'a','smoke_checkpoint':True}})

    def test_calibration_uses_acceptance_and_rejection(self):
        from region_rag.posttrain_data import calibrate,metrics
        rows=[{'scenario':'closed','has_positive':True,'top_correct':True,'score':3},
              {'scenario':'open','has_positive':False,'top_correct':False,'score':1}]
        t=calibrate(rows)
        self.assertGreater(t,1); self.assertLess(t,3)

    def test_interrupted_cache_replace_preserves_previous_json(self):
        from region_rag.posttrain_data import write_json,read_json
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pairs.json'; write_json(path,{'old':1})
            with patch('os.replace',side_effect=OSError('disk failure')):
                with self.assertRaises(OSError): write_json(path,{'new':2})
            self.assertEqual(read_json(path),{'old':1})

    def test_report_runs_on_complete_paired_results(self):
        from argparse import Namespace
        from region_rag.posttrain_data import write_json,sha
        from region_rag.posttrain_run import report
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data'; data.mkdir(); result=root/'result'; result.mkdir()
            expected=[{'case_id':'q_closed'},{'case_id':'q_open'}]
            for split in ('val','test'): write_json(data/f'{split}.json',expected)
            write_json(data/'manifest.json',{'k':12,'files':{n:sha(data/n) for n in ('val.json','test.json')}})
            for arm in ('baseline','sft','dpo'):
                for split in ('val','test'):
                    folder=result/arm/split; folder.mkdir(parents=True)
                    write_json(folder/'complete.json',{'smoke_only':False,'manifest_sha256':sha(data/'manifest.json'),
                        'adapter_sha256':{'baseline':'base','sft':'a','dpo':'b'}[arm],'pixels':128,
                        'training_mode':arm,'smoke_checkpoint':False,'reference_adapter_sha256':'a' if arm=='dpo' else None})
                    write_json(folder/'rows.json',[
                        {'case_id':'q_closed','scenario':'closed','has_positive':True,'top_correct':True,
                         'score':3.,'pair_seconds_sum':1.,'peak_mib':10.},
                        {'case_id':'q_open','scenario':'open','has_positive':False,'top_correct':False,
                         'score':1.,'pair_seconds_sum':1.,'peak_mib':10.}])
            report(Namespace(root=str(result),data=str(data)))
            self.assertTrue((result/'summary.csv').is_file())
            self.assertIn('dpo',(result/'summary.md').read_text())

if __name__=='__main__': unittest.main()
