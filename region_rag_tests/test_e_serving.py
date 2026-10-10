import tempfile
import unittest
from pathlib import Path
from region_rag.e_rerank import validate_e_metadata, validate_local_inputs
from region_rag.pipeline import Pipeline


class EServingTests(unittest.TestCase):
    def fixture(self):
        config={'limit':0,'quality_review':{'sha256':'review'},'parent':'parent','manifest':'data'}
        run={'arm':'E','mode':'pairwise_rank_plus_ce','smoke':False,'config':config,'parent':'parent'}
        meta={'arm':'E','split':'val','smoke':False,'experiment':config,'adapter':'weights','cases':2}
        rows=[{'case_id':'v_closed','scenario':'closed','score':1.},
              {'case_id':'v_open','scenario':'open','score':0.}]
        return run,meta,rows

    def test_only_verified_clean_e_full_validation_can_be_loaded(self):
        run,meta,rows=self.fixture()
        validate_e_metadata(run,meta,rows,'weights')
        for bad in (dict(run,arm='D'),dict(run,smoke=True),dict(run,config={'limit':0})):
            with self.assertRaises(ValueError): validate_e_metadata(bad,meta,rows,'weights')
        with self.assertRaises(ValueError): validate_e_metadata(run,meta,rows,'other_weights')
        with self.assertRaises(ValueError): validate_e_metadata(run,meta,rows[:1],'weights')

    def test_local_map_requires_correct_paths_and_existing_crops(self):
        from region_rag.detail_core import pair_key
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a'; b=Path(tmp)/'b'; a.write_bytes(b'a'); b.write_bytes(b'b')
            paths=[str(a),str(b)]; key=pair_key(paths)
            validate_local_inputs([paths],{key:{'paths':paths,'crops':None}})
            with self.assertRaises(ValueError): validate_local_inputs([paths],{})
            with self.assertRaises(ValueError): validate_local_inputs([paths],{key:{'paths':paths[::-1],'crops':None}})
            with self.assertRaises(ValueError):
                validate_local_inputs([paths],{key:{'paths':paths,'crops':[str(a),'missing']}})

    def test_e_pipeline_prepares_local_views_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe=Pipeline(tmp); pipe.rerank_model='clean_e'; calls=[]
            def worker(module,request,job,python):
                calls.append((module,request,python))
                if request['task']=='prepare_e': return {'local_map':str(Path(tmp)/'locals.json')}
                return {'hits':[],'audit':[]}
            pipe.worker=worker
            pipe.rerank_candidates({'crop':'q','hits':[]},Path(tmp))
            self.assertEqual([r[1]['task'] for r in calls],['prepare_e','rerank'])
            self.assertEqual(calls[1][0],'region_rag.e_rerank')
            self.assertEqual(calls[1][1]['local_map'],str(Path(tmp)/'locals.json'))
