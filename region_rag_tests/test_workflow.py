import unittest,tempfile,time
from pathlib import Path
from PIL import Image
from region_rag.workflow import split_request, summarize_stages
from region_rag.pipeline import Pipeline
from region_rag.catalog import save_index

class WorkflowTests(unittest.TestCase):
    def test_direct_question_does_not_present_candidate_material_as_target_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); pipe=Pipeline(root)
            r={'status':'answered','question':'告诉我这个床的材质','answer':'候选 E1 材质：Steel',
               'region':{'class_name':'bed'},'recall':{'hits':[{}]},
               'hits':[{'evidence_id':'E1','attributes':{'材质':'Steel'}}],
               'intent':split_request('告诉我这个床的材质')}
            r=pipe.finish(r,root,time.perf_counter())
            self.assertIn('不能确定这件物体的材质',r['answer'])
            self.assertEqual(r['stages']['knowledge']['status'],'candidate_attributes_available')
    def test_worker_failure_is_not_rerank_rejection(self):
        r={'status':'error','failure_stage':'reranking','region':{'class_name':'bed'},'recall':{'hits':[{}]},'hits':[]}
        self.assertEqual(summarize_stages(r)['retrieval']['status'],'rerank_failed')

    def test_description_only_material_does_not_claim_no_material_evidence(self):
        r={'status':'answered','region':{'class_name':'bed'},'recall':{'hits':[{}]},
           'hits':[{'evidence_id':'E1','attributes':{},'description':'Material: solid oak wood.'}],
           'intent':split_request('告诉我这个床的材质')}
        self.assertIn('结构化',summarize_stages(r)['knowledge']['message'])
    def test_separate_target_from_requested_attribute(self):
        intent=split_request('告诉我这个床的材质')
        self.assertEqual(intent['target_description'],'这个床')
        self.assertEqual(intent['requested_attributes'],['material'])

    def test_no_retrieval_does_not_mean_no_target_or_no_database_item(self):
        s=summarize_stages({'status':'no_match','region':{'class_name':'bed'},'recall':{'hits':[]},'hits':[]})
        self.assertEqual(s['localization']['status'],'located')
        self.assertEqual(s['retrieval']['status'],'no_candidates')
        self.assertIn('不代表',s['retrieval']['message'])

    def test_attribute_missing_is_separate_from_target_found(self):
        r={'status':'answered','region':{'class_name':'bed'},'recall':{'hits':[{}]},
           'hits':[{'evidence_id':'E1','attributes':{},'description':'a nice bed'}],
           'intent':split_request('告诉我这个床的材质')}
        s=summarize_stages(r)
        self.assertEqual(s['localization']['status'],'located')
        self.assertEqual(s['retrieval']['status'],'candidates_found')
        self.assertEqual(s['knowledge']['status'],'attribute_unverified')

    def test_candidate_material_does_not_confirm_query_identity(self):
        r={'status':'answered','region':{'class_name':'bed'},'recall':{'hits':[{}]},
           'hits':[{'evidence_id':'E1','attributes':{'材质':'Steel'}}],
           'intent':split_request('告诉我这个床的材质')}
        s=summarize_stages(r)
        self.assertEqual(s['knowledge']['status'],'candidate_attributes_available')
        self.assertEqual(s['identity']['status'],'unconfirmed')

    def test_perception_receives_only_target_and_no_match_preserves_localization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=root/'scene.png'; Image.new('RGB',(100,100),'white').save(image)
            save_index(root/'index',[{'item_id':'a'}],[[1,0]])
            region={'id':'R1','class_name':'bed','bbox':[10,10,90,90]}
            calls=[]
            class Fake(Pipeline):
                def worker(self,module,request,job,python):
                    task=request['task']; calls.append((task,request['question']))
                    if task=='segment': return {'regions':[region]}
                    if task=='select': return {'status':'selected','region':region}
                    if task=='verify_instance': return {'status':'verified'}
                    if task=='retrieve': return {'region':region,'hits':[]}
                    raise AssertionError(task)
            r=Fake(root).ask(image,'告诉我这个床的材质')
            for task,q in calls:
                if task in ('segment','select','verify_instance'): self.assertEqual(q,'这个床')
            self.assertEqual(r['stages']['localization']['status'],'located')
            self.assertEqual(r['status'],'no_match')

    def test_empty_knowledge_base_does_not_prevent_visual_localization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=root/'scene.png'; Image.new('RGB',(100,100),'white').save(image)
            region={'id':'R1','class_name':'bed','bbox':[10,10,90,90]}
            class Fake(Pipeline):
                def worker(self,module,request,job,python):
                    if request['task']=='segment': return {'regions':[region]}
                    if request['task']=='select': return {'status':'selected','region':region}
                    if request['task']=='verify_instance': return {'status':'verified'}
                    raise AssertionError('Unavailable index must not be searched')
            r=Fake(root).ask(image,'告诉我这个床的材质')
            self.assertEqual(r['status'],'knowledge_unavailable')
            self.assertEqual(r['stages']['localization']['status'],'located')

    def test_retrieval_error_returns_localized_partial_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=root/'scene.png'; Image.new('RGB',(100,100),'white').save(image)
            save_index(root/'index',[{'item_id':'a'}],[[1,0]])
            region={'id':'R1','class_name':'bed','bbox':[10,10,90,90]}
            class Fake(Pipeline):
                def worker(self,module,request,job,python):
                    if request['task']=='segment': return {'regions':[region]}
                    if request['task']=='select': return {'status':'selected','region':region}
                    if request['task']=='verify_instance': return {'status':'verified'}
                    raise RuntimeError('retrieval worker failed')
            r=Fake(root).ask(image,'告诉我这个床的材质')
            self.assertEqual(r['status'],'error')
            self.assertEqual(r['stages']['localization']['status'],'located')
            self.assertEqual(r['stages']['retrieval']['status'],'retrieval_failed')
            self.assertIn('selected.jpg',r['files'])

if __name__=='__main__': unittest.main()
