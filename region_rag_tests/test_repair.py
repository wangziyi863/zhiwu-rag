import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from region_rag.repair_core import repair_target, should_repair, mask_region, masked_crop, suspected_merge
from region_rag.pipeline import Pipeline
from region_rag.catalog import save_index


class RepairCoreTests(unittest.TestCase):
    def test_inline_attribute_question_does_not_enter_localization(self):
        from region_rag.core import localization_question
        self.assertEqual(localization_question('告诉我这个床的材质'),'这个床')
        self.assertEqual(localization_question('请告诉我前面折叠床的尺寸是多少？'),'前面折叠床')
        self.assertEqual(localization_question('找材质为木头的椅子'),'找材质为木头的椅子')
    def test_zero_detected_known_target_can_retry_without_position(self):
        selected={'status':'uncertain','instance_assessment':{'scene_instance_count':0}}
        self.assertTrue(should_repair('告诉我这个床的材质',selected,None))
        self.assertFalse(should_repair('告诉我这个东西的材质',selected,None))
    def test_vague_multiple_target_cannot_be_auto_selected(self):
        selected={'status':'uncertain','instance_assessment':{'scene_instance_count':2,'question_resolves_target':False}}
        self.assertFalse(should_repair('找这个椅子',selected,None))
        self.assertTrue(should_repair('找左边的椅子',selected,None))

    def test_normal_verified_target_never_runs_repair(self):
        self.assertFalse(should_repair('左边的椅子',{'status':'selected'},{'status':'verified'}))

    def test_all_supported_appearance_and_ordinals_can_repair(self):
        failed={'status':'uncertain','assessment':{'instance_count':2}}
        for q in ('绿色的椅子','黄色的椅子','the third chair','the striped chair'):
            self.assertTrue(should_repair(q,{'status':'selected','region':{'class_name':'chair'}},failed),q)

    def test_landmark_is_not_a_second_requested_object(self):
        self.assertEqual(repair_target('桌子旁边的椅子',{'region':{'class_name':'chair'}}),'chair')
        for relation in ('beside','behind','next to'):
            self.assertTrue(should_repair(f'the chair {relation} the table',
                {'region':{'class_name':'chair'}},{'status':'uncertain','assessment':{'instance_count':2}}))

    def test_spatially_separated_subcandidates_trigger_merge_suspicion(self):
        selected={'id':'R1','class_name':'chair','bbox':[0,0,100,100]}
        regions=[selected,{'id':'R2','class_name':'armchair','bbox':[0,0,40,45]},
                 {'id':'R3','class_name':'chair','bbox':[60,55,100,100]}]
        self.assertTrue(suspected_merge(selected,regions))
        self.assertFalse(suspected_merge(selected,regions[:2]))

    def test_target_mapping_and_unsupported_multi_object_request(self):
        self.assertEqual(repair_target('左边的椅子',{}),'chair')
        self.assertIsNone(repair_target('比较椅子和沙发',{}))

    def test_displaced_seating_component_requires_instance_recheck(self):
        chair={'id':'R1','class_name':'chair','bbox':[0,0,100,100]}
        seat={'id':'R2','class_name':'stool','bbox':[0,70,90,130]}
        self.assertTrue(suspected_merge(chair,[chair,seat]))
        self.assertFalse(suspected_merge(chair,[chair,dict(seat,bbox=[0,90,10,100])]))
        self.assertFalse(suspected_merge(chair,[chair,dict(seat,class_name='desk')]))

    def test_disconnected_visible_parts_are_allowed_but_empty_mask_is_not(self):
        mask=np.zeros((100,100),bool); mask[10:30,10:30]=True; mask[60:85,20:30]=True
        r=mask_region(mask,[5,5,35,90],.9,'chair','I1','mask.png')
        self.assertEqual(r['bbox'],[10,10,30,85])
        with self.assertRaises(ValueError): mask_region(np.zeros_like(mask),[0,0,100,100],.9,'chair','I1','x')

    def test_masked_crop_excludes_neighbor_and_rejects_wrong_coordinate_size(self):
        image=Image.new('RGB',(100,100),'red'); mask=np.zeros((100,100),bool); mask[10:30,10:30]=True
        result=masked_crop(image,mask,[5,5,40,40])
        self.assertEqual(result.getpixel((10,10)),(255,0,0))
        self.assertEqual(result.getpixel((0,0)),(127,127,127))
        with self.assertRaises(ValueError): masked_crop(image,mask[:50],[5,5,40,40])


class RepairPipelineTests(unittest.TestCase):
    def case(self,success,enabled=True):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root=Path(temp.name); image=root/'scene.png'; Image.new('RGB',(100,100),'white').save(image)
        save_index(root/'index',[{'item_id':'a'}],[[1,0]])
        calls=[]; region={'id':'R1','class_name':'chair','bbox':[10,10,90,90]}
        class Fake(Pipeline):
            def worker(self,module,request,job,python):
                task=request['task']; calls.append(task)
                if task=='segment': return {'regions':[region]}
                if task=='select':
                    return {'status':'selected','region':request['regions'][0]}
                if task=='verify_instance':
                    repaired=bool(request.get('mask'))
                    return {'status':'verified' if repaired and success else 'uncertain',
                            'reason':'two chairs','assessment':{'instance_count':2}}
                if task=='repair':
                    mask=job/'I1.png'; arr=np.zeros((100,100),np.uint8); arr[10:90,10:40]=255
                    Image.fromarray(arr).save(mask)
                    return {'status':'candidates','regions':[dict(region,id='I1',bbox=[10,10,40,90],mask_path=str(mask))]}
                if task=='retrieve':
                    assert request['region'].get('mask_path')
                    return {'region':request['region'],'hits':[]}
                raise AssertionError(task)
        result=Fake(root).ask(image,'找左边椅子的相似商品',repair_enabled=enabled)
        return calls,result

    def test_successful_repair_is_verified_before_retrieval(self):
        calls,result=self.case(True)
        self.assertEqual(calls,['segment','select','verify_instance','repair','select','verify_instance','retrieve'])
        self.assertEqual(result['repair']['status'],'repaired')

    def test_two_failed_repairs_never_retrieve(self):
        calls,result=self.case(False)
        self.assertEqual(calls.count('repair'),2)
        self.assertNotIn('retrieve',calls)
        self.assertEqual(result['status'],'target_uncertain')

    def test_disabled_repair_preserves_baseline_failure(self):
        calls,result=self.case(False,False)
        self.assertNotIn('repair',calls)
        self.assertEqual(result['status'],'target_uncertain')


if __name__=='__main__': unittest.main()
