import math
import tempfile
import unittest
from pathlib import Path
from region_rag.dpo_rerank import rank_pairs, load_policy
from region_rag.pipeline import Pipeline


class DpoRerankTests(unittest.TestCase):
    def test_high_clip_score_cannot_override_dpo_and_only_top_one_is_returned(self):
        hits=[{'item_id':'a','similarity':.9}, {'item_id':'b','similarity':.6}]
        result=rank_pairs(hits,[-.1,.3],-.125)
        self.assertEqual([h['item_id'] for h in result['hits']], ['b'])
        self.assertEqual(result['hits'][0]['recall_rank'],2)
        self.assertEqual(result['hits'][0]['rerank']['method'],'dpo_yes_no')

    def test_precise_validation_threshold_rejects_boundary(self):
        hits=[{'item_id':'a','similarity':.9}]
        result=rank_pairs(hits,[-.125],math.nextafter(-.125,math.inf))
        self.assertEqual(result['status'],'no_match')
        self.assertEqual(result['hits'],[])

    def test_invalid_score_fails_instead_of_becoming_no_match(self):
        with self.assertRaises(ValueError):
            rank_pairs([{'item_id':'a','similarity':.9}],[float('nan')],0.)

    def test_missing_checkpoint_never_silently_uses_base_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((ValueError,FileNotFoundError)):
                load_policy(Path(tmp))

    def test_pipeline_dispatches_to_dpo_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe=Pipeline(tmp); calls=[]
            pipe.rerank_model='dpo'
            pipe.worker=lambda module,request,job,python: calls.append((module,request,python)) or {'hits':[]}
            pipe.rerank_candidates({'question':'床的材质','hits':[{}]},Path(tmp))
            self.assertEqual(calls[0][0],'region_rag.dpo_rerank')
            self.assertTrue(calls[0][2].endswith('.venv-posttrain/bin/python'))
            self.assertEqual(calls[0][1]['task'],'rerank')


if __name__=='__main__': unittest.main()
