import unittest

from region_rag.posttrain_direct_compare import (
    build_blind_review,
    rank_web_case,
    summarize_rows,
    validate_shared_cases,
)


def _case(case_id="q1_closed", scenario="closed", candidates=None):
    return {
        "case_id": case_id,
        "query_id": "q1",
        "scenario": scenario,
        "identity": "true-product",
        "has_positive": scenario == "closed",
        "image_path": "/query.jpg",
        "category": "chair",
        "candidates": candidates or [
            {"identity": "wrong-product", "image_path": "/wrong.jpg", "similarity": 0.91},
            {"identity": "true-product", "image_path": "/true.jpg", "similarity": 0.72},
        ],
    }


class DirectComparisonTests(unittest.TestCase):

 def test_web_ranking_uses_production_acceptance_before_score(self):
    """A visually high-scoring candidate rejected by the web rule must never win."""
    case = _case()
    assessments = {
        "wrong-product": {"accepted": False, "visual_score": 4, "request_score": 2, "score": 10},
        "true-product": {"accepted": True, "visual_score": 3, "request_score": 1, "score": 7},
    }

    row = rank_web_case(case, assessments)

    self.assertEqual(row["top_identity"], "true-product")
    self.assertIs(row["top_correct"], True)
    self.assertEqual(row["score"], 7)
    self.assertIs(row["accepted"], True)


 def test_shared_case_validation_rejects_different_candidate_order(self):
    """The direct comparison must fail if an arm was scored on reordered candidates."""
    case = _case()
    row = {
        "case_id": case["case_id"],
        "scenario": "closed",
        "candidate_identities": ["true-product", "wrong-product"],
    }

    with self.assertRaisesRegex(ValueError, "candidate"):
        validate_shared_cases([case], [row])


 def test_summary_uses_product_identity_and_keeps_no_match_separate(self):
    """Closed-product accuracy and open-set false accepts need distinct denominators."""
    rows = [
        {"scenario": "closed", "has_positive": True, "top_correct": True, "accepted": True},
        {"scenario": "closed", "has_positive": False, "top_correct": False, "accepted": False},
        {"scenario": "open", "has_positive": False, "top_correct": False, "accepted": True},
        {"scenario": "open", "has_positive": False, "top_correct": False, "accepted": False},
    ]

    result = summarize_rows(rows)

    self.assertEqual(result, {
        "closed_queries": 2,
        "no_match_queries": 2,
        "recall_at_k": 0.5,
        "conditional_top1": 1.0,
        "retrieval_top1": 0.5,
        "accepted_correct_rate": 0.5,
        "false_accept_rate": 0.5,
    })


 def test_blind_review_deduplicates_candidates_and_hides_arm_names(self):
    """Human similarity review must not leak which system proposed a candidate."""
    case = _case()
    arms = {
        "web_original": [{"case_id": case["case_id"], "top_identity": "wrong-product"}],
        "dpo": [{"case_id": case["case_id"], "top_identity": "wrong-product"}],
        "continue_sft": [{"case_id": case["case_id"], "top_identity": "true-product"}],
    }

    review, key = build_blind_review([case], arms, seed=7, limit=1)

    self.assertEqual(len(review), 2)
    self.assertEqual({r["candidate_code"] for r in review}, {"C1", "C2"})
    self.assertTrue(all("arm" not in r and "model" not in r for r in review))
    self.assertTrue(all("candidate_identity" not in r and "same_product" not in r for r in review))
    self.assertEqual({r["candidate_identity"] for r in key}, {"wrong-product", "true-product"})
    self.assertEqual(sorted(key[0]["proposed_by"]), ["dpo", "web_original"])


if __name__ == "__main__":
    unittest.main()
