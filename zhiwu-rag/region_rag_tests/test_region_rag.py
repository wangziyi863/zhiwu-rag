import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class ModuleAvailable(unittest.TestCase):
    def test_automatic_region_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('region_rag'), 'Automatic region RAG module has not been implemented')


from region_rag.core import parse_intent, select_region, extract_regions, validate_bbox, validate_answer
from region_rag.catalog import load_catalog, save_index, search_index


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.regions = [
            {'id': 'R1', 'class_name': 'chair', 'bbox': [70, 20, 95, 80], 'area': 1000},
            {'id': 'R2', 'class_name': 'chair', 'bbox': [5, 20, 30, 80], 'area': 900},
            {'id': 'R3', 'class_name': 'sofa', 'bbox': [35, 30, 60, 90], 'area': 1100},
        ]

    def test_chinese_left_chair_selects_left_not_input_order(self):
        intent = parse_intent('帮我找左边这把椅子的类似产品')
        result = select_region(self.regions, intent, (100, 100))
        self.assertEqual(result['status'], 'selected')
        self.assertEqual(result['region']['id'], 'R2')

    def test_two_chairs_without_direction_are_ambiguous(self):
        self.assertEqual(select_region(self.regions, parse_intent('这把椅子是什么材质'), (100, 100))['status'], 'ambiguous')

    def test_missing_table_never_falls_back_to_chair(self):
        self.assertEqual(select_region(self.regions, parse_intent('右边的桌子'), (100, 100))['status'], 'not_found')

    def test_two_target_categories_require_model_disambiguation(self):
        self.assertEqual(parse_intent('椅子和沙发有什么区别')['status'], 'needs_model')

    def test_negation_is_not_treated_as_target(self):
        self.assertEqual(parse_intent('不要椅子，找沙发')['status'], 'needs_model')

    def test_unknown_target_is_not_made_up(self):
        self.assertEqual(parse_intent('这个东西是什么')['status'], 'needs_model')

    def test_bright_is_not_a_right_direction(self):
        intent = parse_intent('Find the bright chair')
        self.assertIsNone(intent['position'])
        self.assertEqual(select_region(self.regions, intent, (100,100))['status'], 'ambiguous')

    def test_english_ordinals_require_model(self):
        for word in ('second', '2nd', 'third', 'last'):
            self.assertEqual(parse_intent(f'Find the {word} chair from the left')['status'], 'needs_model')

    def test_broad_connected_region_requires_disambiguation(self):
        merged = [{'id': 'R1', 'class_name': 'chair', 'bbox': [0, 10, 99, 90], 'area': 6000}]
        self.assertEqual(select_region(merged, parse_intent('左边的椅子'), (100, 100))['status'], 'ambiguous')

    def test_connected_components_keep_original_coordinates(self):
        labels = np.zeros((20, 30), dtype=np.uint8)
        labels[2:8, 3:9] = 1
        labels[10:18, 21:28] = 1
        regions = extract_regions(labels, ['wall', 'chair'], ['chair'], min_area=4)
        self.assertEqual(sorted(r['bbox'] for r in regions), [[3, 2, 9, 8], [21, 10, 28, 18]])

    def test_bbox_rejects_reversed_and_outside(self):
        for box in ([20, 10, 5, 30], [-1, 0, 20, 30], [0, 0, 101, 90]):
            with self.assertRaises(ValueError):
                validate_bbox(box, (100, 100))

    def test_answer_rejects_nonexistent_evidence(self):
        with self.assertRaises(ValueError):
            validate_answer({'answer': '是实木椅子 [E99]', 'citations': ['E99']}, ['E1'])

    def test_answer_checks_inline_references_not_just_metadata(self):
        with self.assertRaises(ValueError):
            validate_answer({'answer': '资料见 [E3]', 'citations': ['E1']}, ['E1'])

    def test_vlm_selection_rejects_unknown_region(self):
        from region_rag.protocol import resolve_selection
        with self.assertRaises(ValueError):
            resolve_selection({'status': 'selected', 'region_id': 'R999'}, self.regions, (100,100))

    def test_vlm_box_uses_width_and_height_in_correct_order(self):
        from region_rag.protocol import resolve_selection
        result = resolve_selection({'status': 'selected', 'bbox_1000': [100,200,500,600]}, [], (200,100))
        self.assertEqual(result['region']['bbox'], [20,20,100,60])

    def test_vlm_uncertainty_is_not_a_selection(self):
        from region_rag.protocol import resolve_selection
        self.assertEqual(resolve_selection({'status': 'uncertain'}, self.regions, (100,100))['status'], 'uncertain')

    def test_singleton_model_json_list_is_unwrapped(self):
        from region_rag.protocol import parse_json_output
        result = parse_json_output('```json\n[{"answer":"资料不足","citations":[]}]\n```')
        self.assertEqual(result['answer'], '资料不足')

    def test_multiple_model_answers_are_not_silently_chosen(self):
        from region_rag.protocol import parse_json_output
        with self.assertRaises(ValueError):
            parse_json_output('[{"answer":"甲"},{"answer":"乙"}]')

    def test_grounded_answer_rejects_invented_material(self):
        from region_rag.protocol import grounded_answer
        hits = [{'evidence_id':'E1', 'description':'外观为浅棕色。', 'attributes':{}}]
        with self.assertRaises(ValueError):
            grounded_answer({'evidence':[{'id':'E1','field':'description','quote':'实木材质'}]}, hits)

    def test_grounded_answer_keeps_candidate_attribute_separate(self):
        from region_rag.protocol import grounded_answer
        hits = [{'evidence_id':'E1', 'description':'', 'attributes':{'材质':'钢'}}]
        result = grounded_answer({'evidence':[{'id':'E1','field':'attributes.材质','quote':'钢'}]}, hits)
        self.assertIn('[E1]', result['answer'])
        self.assertIn('材质', result['answer'])
        self.assertIn('不能确认', result['answer'])
        self.assertEqual(result['answer_mode'], 'validated_source_excerpts')

    def test_grounded_answer_requires_whole_field_to_preserve_negation(self):
        from region_rag.protocol import grounded_answer
        hits = [{'evidence_id':'E1', 'description':'不是实木，材质未提供。', 'attributes':{}}]
        with self.assertRaises(ValueError):
            grounded_answer({'evidence':[{'id':'E1','field':'description','quote':'实木'}]}, hits)

    def test_source_fallback_is_explicit(self):
        from region_rag.protocol import source_fallback
        result = source_fallback([{'evidence_id':'E1','description':'未提供商品属性。','attributes':{}}])
        self.assertEqual(result['answer_mode'], 'source_excerpt_fallback')
        self.assertIn('校验', result['answer'])

    def test_model_can_select_field_without_rewriting_source(self):
        from region_rag.protocol import grounded_answer
        hits = [{'evidence_id':'E1','description':'不是实木，材质未提供。','attributes':{}}]
        result = grounded_answer({'evidence':[{'id':'E1','field':'description'}]}, hits)
        self.assertIn('不是实木，材质未提供。', result['answer'])


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        Image.new('RGB', (16, 16), 'red').save(self.root / 'a.png')

    def tearDown(self):
        self.tmp.cleanup()

    def write_catalog(self, lines):
        p = self.root / 'items.jsonl'
        p.write_text('\n'.join(lines), encoding='utf-8')
        return p

    def test_relative_image_path_and_missing_description(self):
        items = load_catalog(self.write_catalog(['{"item_id":"a","image_path":"a.png","name":"椅子","source":"本地演示"}']))
        self.assertEqual(items[0]['image_path'], str(self.root / 'a.png'))
        self.assertEqual(items[0]['description'], '')

    def test_duplicate_id_is_rejected(self):
        line = '{"item_id":"a","image_path":"a.png","name":"椅子","source":"本地演示"}'
        with self.assertRaises(ValueError):
            load_catalog(self.write_catalog([line, line]))

    def test_empty_catalog_is_rejected(self):
        with self.assertRaises(ValueError):
            load_catalog(self.write_catalog([]))

    def test_ranking_preserves_item_vector_pairing_after_reload(self):
        items = [{'item_id': 'first'}, {'item_id': 'second'}]
        save_index(self.root / 'idx', items, np.array([[1, 0], [0, 2]], dtype=np.float32))
        hits = search_index(self.root / 'idx', np.array([0, 1], dtype=np.float32), top_k=1)
        self.assertEqual(hits[0]['item_id'], 'second')
        self.assertEqual(hits[0]['evidence_id'], 'E1')
        self.assertAlmostEqual(hits[0]['similarity'], 1.0)

    def test_wrong_vector_dimension_is_rejected(self):
        save_index(self.root / 'idx', [{'item_id': 'a'}], np.array([[1, 0]], dtype=np.float32))
        with self.assertRaises(ValueError):
            search_index(self.root / 'idx', np.array([1, 0, 0], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
