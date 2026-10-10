import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from region_rag.abo import convert_listing
from region_rag.catalog import save_index, search_index
from region_rag.ranking import validate_pair, rank_candidates, choose_categories
from region_rag.pipeline import Pipeline
from region_rag.core import localization_question
from region_rag.protocol import canvas_bbox_to_normalized
from region_rag.protocol import answer_choices, resolve_answer_choices
from PIL import Image


class ABOTests(unittest.TestCase):
    def test_answer_can_only_select_known_numeric_field_choices(self):
        hits=[{'evidence_id':'E2','attributes':{'材质':'Steel'},'description':'Not wood.'}]
        choices=answer_choices(hits)
        material=next(c for c in choices if c['field']=='attributes.材质')
        selection=resolve_answer_choices({'choices':[material['choice']]},choices)
        self.assertEqual(selection,{'evidence':[{'id':'E2','field':'attributes.材质'}]})
        for bad in (True,999,'1'):
            with self.assertRaises(ValueError): resolve_answer_choices({'choices':[bad]},choices)

    def test_material_comparison_offers_each_candidates_material(self):
        hits=[{'evidence_id':f'E{i}','attributes':{'材质':v,'品牌':'Brand'},'description':'A chair'}
              for i,v in enumerate(('Steel','Wood','Plastic'),1)]
        choices=answer_choices(hits,'比较这些候选商品的材质')
        self.assertEqual([c['id'] for c in choices],['E1','E2','E3'])
        self.assertTrue(all(c['field']=='attributes.材质' for c in choices))

    def test_material_in_description_remains_available_per_candidate(self):
        hits=[{'evidence_id':'E1','attributes':{'材质':'Steel'}},
              {'evidence_id':'E2','description':'Solid wood frame','attributes':{'品牌':'A'}}]
        choices=answer_choices(hits,'比较材质')
        self.assertEqual([(c['id'],c['field']) for c in choices],
                         [('E1','attributes.材质'),('E2','description')])

    def test_relative_height_is_not_a_product_information_request(self):
        q='找左边的椅子，比较高的那把'
        self.assertEqual(localization_question(q),q)

    def test_canvas_coordinates_map_to_original_not_resized_pixels(self):
        self.assertEqual(canvas_bbox_to_normalized([126,0,378,504],(100,200)),[0,0,1000,1000])
        self.assertEqual(canvas_bbox_to_normalized([189,126,315,378],(100,200)),[250,250,750,750])
        with self.assertRaises(ValueError):
            canvas_bbox_to_normalized([0,0,100,100],(100,200))
        with self.assertRaises(ValueError):
            canvas_bbox_to_normalized([float('nan'),0,300,400],(100,200))

    def test_localization_uses_target_not_downstream_material_question(self):
        question=localization_question('找左边椅子的相似商品，并比较材质')
        self.assertIn('椅子',question)
        self.assertIn('左',question)
        self.assertNotIn('材质',question)
        complex_query='找黑色椅子，不要右边那个'
        self.assertEqual(localization_question(complex_query),complex_query)
        self.assertIn('带扶手',localization_question('找带扶手椅子的相似商品，并比较材质'))

    def test_source_identity_units_missing_values(self):
        row = {'item_id':'B123','domain_name':'amazon.com','product_type':[{'value':'CHAIR'}],
               'main_image_id':'img','item_name':[{'language_tag':'en_US','value':'Chair'}],
               'item_dimensions':{'height':{'value':30,'unit':'inches'}}}
        item = convert_listing(row, {'img':'/tmp/chair.jpg'}, 'listings_0.json.gz', 12)
        self.assertEqual(item['item_id'],'abo:amazon.com:B123')
        self.assertEqual(item['category'],'chair')
        self.assertNotIn('材质',item['attributes'])
        self.assertEqual(item['attributes']['高度'],'30 inches')
        self.assertEqual(item['provenance']['line'],12)
        self.assertEqual(item['provenance']['item_id'],'B123')
        row['domain_name']='amazon.co.uk'
        self.assertNotEqual(convert_listing(row, {'img':'/tmp/chair.jpg'},'x',1)['item_id'],item['item_id'])

    def test_accessories_and_unmapped_images_excluded(self):
        row={'item_id':'a','domain_name':'amazon.com','product_type':[{'value':'FURNITURE_COVER'}],
             'main_image_id':'img','item_name':[{'value':'Chair cover'}]}
        self.assertIsNone(convert_listing(row, {'img':'/tmp/x'},'x',1))
        row['product_type']=[{'value':'CHAIR'}]
        self.assertIsNone(convert_listing(row, {},'x',1))

    def test_category_filter_and_low_similarity_reject(self):
        with tempfile.TemporaryDirectory() as root:
            items=[{'item_id':'bath','category':'bathroom'}, {'item_id':'chair','category':'chair'}]
            save_index(root, items, [[1,0],[.6,.8]])
            hits=search_index(root,[1,0],12,categories=['chair'],min_similarity=.2)
            self.assertEqual([h['item_id'] for h in hits],['chair'])
            self.assertEqual(search_index(root,[1,0],12,categories=[],min_similarity=.2),[])
            self.assertEqual(search_index(root,[1,0],12,categories=['chair'],min_similarity=.7),[])

    def test_independent_rank_changes_order_and_rejects_invalid(self):
        hits=[{'evidence_id':'E1','similarity':.9}, {'evidence_id':'E2','similarity':.7}]
        checks=[validate_pair({'same_category':False,'visual_score':4,'request_score':2}),
                validate_pair({'same_category':True,'visual_score':3,'request_score':2})]
        self.assertEqual([h['evidence_id'] for h in rank_candidates(hits,checks)],['E2'])
        for bad in (True, float('nan'), '3', 8):
            with self.assertRaises(ValueError):
                validate_pair({'same_category':True,'visual_score':bad,'request_score':2})
        self.assertEqual(rank_candidates(hits,[dict(accepted=False),dict(accepted=False)]),[])

    def test_categories_unknown_does_not_search_all_furniture(self):
        self.assertEqual(choose_categories('bottle', {'chair':.3}),[])
        self.assertEqual(choose_categories('swivelchair', {'chair':.3,'stool':.29,'sofa':.1}),['chair','stool'])
        self.assertEqual(choose_categories('chair', {'chair':.3,'stool':.1}),['chair'])

    def test_vlm_category_aliases_and_bookcase(self):
        for target,category in [('bookcase','shelf'),('couch','sofa'),('carpet','rug'),
                                ('coffee table','table'),('dining chair','chair'),('dresser','cabinet')]:
            self.assertEqual(choose_categories(target, {category:.3})[0],category)
        self.assertEqual(choose_categories('chair cover', {'chair':.3}),[])

    def pipeline_case(self, accepted, retrieval_only):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); im=root/'input.jpg'; Image.new('RGB',(100,100)).save(im)
            hit={'item_id':'a','evidence_id':'E1','image_path':str(im),'similarity':.8}
            save_index(root/'index',[hit],[[1,0]])
            calls=[]
            region={'class_name':'chair','bbox':[1,1,99,99]}
            class Stub(Pipeline):
                def worker(self,module,request,job,python):
                    task=request['task']; calls.append(task)
                    if task=='segment': return {'regions':[region]}
                    if task=='select': return {'status':'selected','region':region}
                    if task=='verify_instance': return {'status':'verified'}
                    if task=='retrieve':
                        Image.new('RGB',(98,98)).save(job/'target.jpg')
                        return {'region':region,'hits':[hit]}
                    if task=='rerank':
                        from region_rag.dpo_rerank import rank_pairs
                        return rank_pairs([hit],[1. if accepted else -1.],0.)
                    if task=='prepare_e': return {'local_map':'unused-in-worker-fixture'}
                    if task=='verify_match':
                        from region_rag.match_state import validate_visual_comparison
                        return validate_visual_comparison({'target_unambiguous':True,'comparison_possible':True,
                            'same_visible_design':True,'observations':[{'feature':'part_layout',
                            'query':'椅背与扶手布局','candidate':'椅背与扶手布局','both_visible':True,'relation':'match'}]})
                    raise AssertionError('Should not generate an answer here')
            result=Stub(root).ask(im,'椅子',retrieval_only=retrieval_only)
            return calls,result

    def test_retrieval_only_still_reranks(self):
        calls,result=self.pipeline_case(True,True)
        self.assertEqual(calls[-1],'verify_match')
        self.assertEqual(result['status'],'retrieval_only')

    def test_rejected_recall_cannot_generate_answer(self):
        calls,result=self.pipeline_case(False,False)
        self.assertEqual(calls[-1],'verify_match')
        self.assertEqual(result['status'],'insufficient_evidence')
        self.assertEqual(result['hits'],[])


if __name__=='__main__':
    unittest.main()
