import unittest


class FollowupTests(unittest.TestCase):
    def test_far_threshold_controls_validation_ties(self):
        from region_rag.posttrain_followup import far_threshold
        from region_rag.posttrain_data import metrics
        rows=[{'scenario':'open','score':s,'top_correct':False,'has_positive':False} for s in (1.,1.,2.,3.)]
        rows.append({'scenario':'closed','score':4.,'top_correct':True,'has_positive':True})
        t=far_threshold(rows,.25)
        self.assertGreater(t,2.); self.assertLessEqual(t,3.)
        self.assertEqual(metrics(rows,t)['false_accept_rate'],.25)

    def test_threshold_has_no_dependency_on_test_scores(self):
        from region_rag.posttrain_followup import far_threshold
        val=[{'scenario':'open','score':1.},{'scenario':'open','score':3.}]
        self.assertGreater(far_threshold(val,0),3.)

    def test_paired_changes_count_gains_and_losses_separately(self):
        from region_rag.posttrain_followup import paired_counts
        a=[{'case_id':str(i),'scenario':'closed','top_correct':x} for i,x in enumerate([True,False,True,False])]
        b=[dict(r,top_correct=x) for r,x in zip(a,[False,True,True,False])]
        result=paired_counts(a,b)
        self.assertEqual((result['gained'],result['lost'],result['both_correct'],result['both_wrong']),(1,1,1,1))
        self.assertEqual(result['mcnemar_exact_p'],1.)

    def test_continue_sft_requires_and_loads_parent(self):
        from region_rag.posttrain_run import training_parent
        self.assertEqual(training_parent('continue_sft','parent'),'parent')
        with self.assertRaises(ValueError): training_parent('continue_sft',None)
        with self.assertRaises(ValueError): training_parent('sft','ignored-parent')

    def test_budget_binds_to_models_that_were_actually_scored(self):
        from region_rag.posttrain_followup import validate_scored_adapters
        metadata={'sft':{'adapter_sha256':'old_sft'},'dpo':{'adapter_sha256':'old_dpo'}}
        with self.assertRaises(ValueError): validate_scored_adapters(metadata,'new_sft','new_dpo')
        validate_scored_adapters(metadata,'old_sft','old_dpo')

if __name__=='__main__': unittest.main()
