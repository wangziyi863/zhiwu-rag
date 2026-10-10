import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path
from PIL import Image
from region_rag.match_state import validate_visual_comparison, apply_match_state, ground_comparison
from region_rag.dpo_rerank import rank_pairs
from region_rag.pipeline import Pipeline
from region_rag.catalog import save_index
from region_rag.match_verify import verify


def observation(relation='match', feature='part_layout'):
    return {'feature':feature,'query':'四个抽屉','candidate':'四个抽屉' if relation=='match' else '六个抽屉',
            'both_visible':True,'relation':relation}


def assessment(relation='match', **changes):
    value={'target_unambiguous':True,'comparison_possible':True,
           'same_visible_design':relation=='match','observations':[observation(relation)],'uncertainties':[]}
    value.update(changes)
    return validate_visual_comparison(value)


class MatchStateTests(unittest.TestCase):
    def decide(self, score, visual):
        hit={'item_id':'a','evidence_id':'E1','similarity':.8}
        return apply_match_state(rank_pairs([hit],[score],0.),hit,visual)

    def test_low_dpo_score_is_insufficient_not_a_proven_difference(self):
        result=self.decide(-1.,assessment())
        self.assertEqual(result['match']['status'],'insufficient_evidence')
        self.assertEqual(result['hits'],[])

    def test_possible_match_requires_both_sources_and_never_confirms_identity(self):
        result=self.decide(1.,assessment())
        self.assertEqual(result['match']['status'],'possible_match')
        self.assertEqual(len(result['hits']),1)
        self.assertFalse(result['identity_confirmed'])

    def test_observed_structure_difference_can_reject_a_high_dpo_score(self):
        result=self.decide(1.,assessment('different'))
        self.assertEqual(result['match']['status'],'clearly_different')
        self.assertEqual(result['match']['scope'],'top_candidate_only')
        self.assertEqual(result['hits'],[])

    def test_ambiguous_target_or_hidden_parts_cannot_prove_difference(self):
        for change in ({'target_unambiguous':False},{'comparison_possible':False}):
            result=self.decide(1.,assessment('different',**change))
            self.assertEqual(result['match']['status'],'insufficient_evidence')

    def test_inferred_material_or_size_is_not_visual_contradiction(self):
        for feature in ('material','dimensions','brand','color'):
            value=assessment('different',observations=[observation('different',feature)])
            self.assertEqual(value['status'],'insufficient_evidence')

    def test_missing_observations_or_string_booleans_fail_closed(self):
        self.assertEqual(assessment(observations=[])['status'],'insufficient_evidence')
        self.assertEqual(assessment(comparison_possible='true')['status'],'insufficient_evidence')
        self.assertEqual(validate_visual_comparison([])['status'],'insufficient_evidence')
        hidden=observation('different'); hidden['both_visible']=False
        self.assertEqual(assessment('different',observations=[hidden])['status'],'insufficient_evidence')

    def test_identical_descriptions_cannot_support_a_contradiction(self):
        row=observation('different'); row['candidate']=row['query']
        self.assertEqual(assessment('different',observations=[row])['status'],'insufficient_evidence')

    def test_copied_prompt_placeholders_are_not_visual_evidence(self):
        row=observation(); row.update(query='图1可见细节',candidate='图2可见细节')
        self.assertEqual(assessment(observations=[row])['status'],'insufficient_evidence')

    def test_partial_view_blocks_even_a_high_scoring_match(self):
        checks=[{'single_target':True,'design_view_sufficient':False,'visible_observation_ids':[0],
                 'reason':'仅桌面局部，看不到整体结构'},
                {'single_target':True,'design_view_sufficient':True,'visible_observation_ids':[0]}]
        visual=ground_comparison(assessment(),checks)
        self.assertEqual(self.decide(5.,visual)['match']['status'],'insufficient_evidence')

    def test_contradiction_requires_support_from_each_image_separately(self):
        checks=[{'single_target':True,'design_view_sufficient':True,'visible_observation_ids':[]},
                {'single_target':True,'design_view_sufficient':True,'visible_observation_ids':[0]}]
        self.assertEqual(ground_comparison(assessment('different'),checks)['status'],'insufficient_evidence')
        checks[0]['visible_observation_ids']=[0]
        self.assertEqual(ground_comparison(assessment('different'),checks)['status'],'clearly_different')

    def test_invalid_grounding_ids_fail_closed(self):
        valid={'single_target':True,'design_view_sufficient':True,'visible_observation_ids':[0]}
        for ids in ([True],[1],[-1],[0,0],'0'):
            invalid=dict(valid,visible_observation_ids=ids)
            self.assertEqual(ground_comparison(assessment(),[valid,invalid])['status'],'insufficient_evidence')

    def test_uncertain_worker_result_is_serializable_without_additional_passes(self):
        raw=json.dumps({'target_unambiguous':True,'comparison_possible':False,
                        'same_visible_design':False,'observations':[]})
        with patch('region_rag.vlm.generate',return_value=(raw,{'generation_seconds':1.,'peak_allocated_mib':2.})):
            result=verify({'crop':'query','candidate_image':'candidate'})
        self.assertEqual(json.loads(json.dumps(result))['status'],'insufficient_evidence')
        self.assertEqual(result['stats']['passes'],1)

    def test_worker_uses_separate_images_and_removes_unsupported_claims(self):
        responses=[{'target_unambiguous':True,'comparison_possible':True,'same_visible_design':True,
                    'observations':[observation()]},
                   {'single_target':True,'design_view_sufficient':False,'visible_observation_ids':[]},
                   {'single_target':True,'design_view_sufficient':True,'visible_observation_ids':[0]}]
        def generate(request,prompt,paths):
            expected=[['query','candidate'],['query'],['candidate']][3-len(responses)]
            self.assertEqual(paths,expected)
            return json.dumps(responses.pop(0)),{'generation_seconds':1.,'peak_allocated_mib':2.}
        with patch('region_rag.vlm.generate',side_effect=generate):
            result=verify({'crop':'query','candidate_image':'candidate'})
        self.assertEqual(result['status'],'insufficient_evidence')
        self.assertEqual(result['observations'],[])
        self.assertEqual(result['stats']['passes'],3)

    def test_accepted_candidate_without_score_audit_cannot_skip_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            pipe=Pipeline(temp)
            pipe.rerank_model='dpo'
            pipe.worker=lambda *args: {'hits':[{'item_id':'a'}]}
            with self.assertRaises(RuntimeError):
                pipe.rerank_candidates({'hits':[{'item_id':'a'}]},Path(temp))

    def test_verifier_failure_returns_uncertainty_and_preserves_localization(self):
        self.pipeline_case(fail=True)

    def test_rejected_candidate_never_enters_attribute_answering(self):
        self.pipeline_case(fail=False)

    def pipeline_case(self,fail):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); image=root/'input.jpg'; Image.new('RGB',(100,100)).save(image)
            hit={'item_id':'a','evidence_id':'E1','name':'候选柜子','image_path':str(image),'similarity':.9}
            save_index(root/'index',[hit],[[1,0]])
            region={'id':'R1','class_name':'bed','bbox':[5,5,95,95]}
            calls=[]
            class Fake(Pipeline):
                def worker(self,module,request,job,python):
                    calls.append(request['task'])
                    if request['task']=='segment': return {'regions':[region]}
                    if request['task']=='select': return {'status':'selected','region':region}
                    if request['task']=='verify_instance': return {'status':'verified'}
                    if request['task']=='retrieve':
                        Image.new('RGB',(90,90)).save(job/'target.jpg')
                        return {'region':region,'hits':[hit]}
                    if request['task']=='rerank': return rank_pairs([hit],[1.],0.)
                    if request['task']=='prepare_e': return {'local_map':'unused-in-worker-fixture'}
                    if request['task']=='verify_match':
                        self_test.assertNotIn('question',request)
                        self_test.assertNotIn('attributes',request)
                        if fail: raise RuntimeError('verification worker failed')
                        return assessment('different')
                    raise AssertionError('Rejected candidates must not be sent to answer generation')
            self_test=self
            result=Fake(root).ask(image,'这个床是什么材质')
            expected='insufficient_evidence' if fail else 'clearly_different'
            self.assertEqual(result['match']['status'],expected)
            self.assertEqual(result['stages']['identity']['status'],expected)
            self.assertEqual(result['stages']['localization']['status'],'located')
            self.assertEqual(result['hits'],[])
            self.assertNotIn('answer',calls)
            self.assertIn('match_candidate.jpg',result['files'])


if __name__=='__main__': unittest.main()
