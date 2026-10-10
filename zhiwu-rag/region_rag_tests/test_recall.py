import unittest
from region_rag.recall import select_candidates, resolve_categories, CATEGORY_DESCRIPTIONS
from region_rag.core import TARGETS

class RecallTests(unittest.TestCase):
    def test_zero_instance_reason_is_not_ambiguity(self):
        from region_rag.protocol import resolve_instance_selection
        r=resolve_instance_selection({'status':'uncertain','scene_instance_count':0,'question_resolves_target':False},[],(100,100),{})
        self.assertIn('未识别',r['reason'])
    def test_global_rescue_keeps_wrong_category_high_similarity(self):
        items=[{'item_id':str(i),'category':'stool' if i==0 else 'sofa'} for i in range(8)]
        hits=select_candidates(items,[.99,.9,.8,.7,.6,.5,.4,.3],['sofa'],3,.2,True)
        self.assertEqual(len(hits),3)
        self.assertIn('0',[h['item_id'] for h in hits])
        self.assertEqual(next(h for h in hits if h['item_id']=='0')['recall_branch'],'global')

    def test_unknown_target_does_not_open_unrestricted_search(self):
        self.assertEqual(select_candidates([{'item_id':'a','category':'chair'}],[.9],[],12,.2,True),[])

    def test_threshold_and_cross_market_dedup(self):
        items=[{'item_id':x,'category':'chair','provenance':{'item_id':p}}
               for x,p in [('uk:a','a'),('us:a','a'),('us:b','b'),('us:c','c')]]
        hits=select_candidates(items,[.9,.89,.8,.1],['chair'],3,.2,True)
        self.assertEqual([h['item_id'] for h in hits],['uk:a','us:b'])

    def test_fill_when_primary_pool_empty(self):
        items=[{'item_id':str(i),'category':'stool'} for i in range(5)]
        self.assertEqual(len(select_candidates(items,[.9]*5,['sofa'],3,.2,True)),3)

    def test_explicit_question_conflict_retains_both_supported_categories(self):
        self.assertEqual(set(resolve_categories('sofa',{'sofa':.4,'stool':.3},'找这个收纳凳的相似商品',True)),{'sofa','stool'})

    def test_equal_prompt_count_and_coverage(self):
        self.assertEqual(set(CATEGORY_DESCRIPTIONS),set(TARGETS))
        self.assertEqual({len(v) for v in CATEGORY_DESCRIPTIONS.values()},{4})

    def test_unique_named_category_survives_color_and_ordinal(self):
        for q in ('找红色收纳凳的相似商品','the third stool','the green stool'):
            self.assertIn('stool',resolve_categories('sofa',{'sofa':.4,'stool':.3},q,True))
        self.assertEqual(resolve_categories('sofa',{'sofa':.4},'比较椅子和凳子',True),['sofa'])
        self.assertEqual(resolve_categories('sofa',{'sofa':.4},'不要凳子',True),['sofa'])

if __name__=='__main__': unittest.main()
