"""Regression coverage for merged semantic regions bypassing instance checks."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from region_rag.core import parse_intent
from region_rag.catalog import save_index
from region_rag.pipeline import Pipeline
from region_rag import protocol


class InstanceProtocolTests(unittest.TestCase):
    def setUp(self):
        self.regions = [{'id':'R2','class_name':'chair','bbox':[10,10,90,90]}]
        self.value = {'status':'selected','region_id':'R2','scene_instance_count':1,
                      'question_resolves_target':True}

    def test_counted_multiple_chairs_cannot_use_vague_this_chair(self):
        self.value['scene_instance_count'] = 3
        result = protocol.resolve_instance_selection(self.value, self.regions, (100,100), parse_intent('这个椅子'))
        self.assertEqual(result['status'], 'uncertain')

    def test_missing_instance_evidence_never_passes(self):
        result = protocol.resolve_instance_selection({'status':'selected','region_id':'R2'}, self.regions, (100,100), parse_intent('椅子'))
        self.assertEqual(result['status'], 'uncertain')

    def test_single_scene_instance_is_a_proposal_not_final_verification(self):
        result = protocol.resolve_instance_selection(self.value, self.regions, (100,100), parse_intent('椅子'))
        self.assertEqual(result['status'], 'selected')
        self.assertTrue(result['requires_instance_verification'])

    def test_direction_allows_disambiguation_among_multiple(self):
        self.value['scene_instance_count'] = 3
        result = protocol.resolve_instance_selection(self.value, self.regions, (100,100), parse_intent('左边的椅子'))
        self.assertEqual(result['status'], 'selected')

    def test_crop_with_multiple_instances_is_rejected(self):
        result = protocol.validate_instance_check({'instance_count':2,'matches_question':True,'single_complete_target':True})
        self.assertEqual(result['status'], 'uncertain')

    def test_crop_must_be_complete_match_and_single(self):
        base = {'instance_count':1,'matches_question':True,'single_complete_target':True}
        self.assertEqual(protocol.validate_instance_check(base)['status'], 'verified')
        for key, value in [('instance_count',True),('instance_count','1个或2个'),('instance_count',1.5),('matches_question',False),('single_complete_target',False)]:
            self.assertEqual(protocol.validate_instance_check(dict(base, **{key:value}))['status'], 'uncertain')

    def test_unambiguous_numeric_strings_are_compatible(self):
        result = protocol.validate_instance_check({'instance_count':'1','matches_question':True,'single_complete_target':True})
        self.assertEqual(result['status'], 'verified')
        self.value['scene_instance_count'] = '1'
        self.assertEqual(protocol.resolve_instance_selection(self.value, self.regions, (100,100), parse_intent('椅子'))['status'], 'selected')
        result = protocol.validate_instance_check({'instance_count':'3','matches_question':True,'single_complete_target':True})
        self.assertEqual(result['status'], 'uncertain')

    def test_appearance_descriptions_are_not_rejected_as_vague(self):
        self.value['scene_instance_count'] = 2
        for question in ('找棕色的椅子', '找条纹椅子', '找靠近镜头的椅子'):
            intent = parse_intent(question)
            self.assertEqual(intent['status'], 'needs_model')
            self.assertEqual(protocol.resolve_instance_selection(self.value, self.regions, (100,100), intent)['status'], 'selected')


class PipelineInstanceTests(unittest.TestCase):
    def run_case(self, selection_status='uncertain', verification_status='uncertain'):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        image = root/'scene.png'
        Image.new('RGB',(100,100)).save(image)
        save_index(root/'index', [{'item_id':'test'}], np.array([[1.,0.]]))
        region = {'id':'R2','class_name':'chair','bbox':[10,10,90,90]}
        calls = []
        class FakePipeline(Pipeline):
            def worker(self, module, request, job, python):
                task = request['task']
                calls.append(task)
                if task == 'segment':
                    return {'status':'selected','regions':[region], 'selection':{'status':'selected','region':region}}
                if task == 'select':
                    return {'status':selection_status,'region':region,'reason':'目标不明确'}
                if task == 'verify_instance':
                    return {'status':verification_status,'reason':'框中有多把椅子'}
                if task == 'retrieve':
                    Image.new('RGB',(80,80)).save(job/'target.jpg')
                    return {'status':'retrieved','region':request['region'],'hits':[]}
                raise AssertionError('Unexpected worker '+task)
        result = FakePipeline(root).ask(image,'这个椅子',retrieval_only=True)
        return calls, result

    def test_unique_component_still_requires_selection_even_in_retrieval_only(self):
        calls, result = self.run_case()
        self.assertEqual(calls, ['segment','select'])
        self.assertEqual(result['status'], 'target_uncertain')
        self.assertNotIn('hits', result)

    def test_failed_crop_check_never_retrieves(self):
        calls, result = self.run_case('selected')
        self.assertEqual(calls, ['segment','select','verify_instance'])
        self.assertEqual(result['status'], 'target_uncertain')
        self.assertNotIn('hits', result)

    def test_verified_single_target_continues_to_retrieval(self):
        calls, result = self.run_case('selected','verified')
        self.assertEqual(calls, ['segment','select','verify_instance','retrieve'])
        self.assertEqual(result['status'], 'no_match')


if __name__ == '__main__':
    unittest.main()
