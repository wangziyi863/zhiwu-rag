import tempfile
import unittest
from pathlib import Path
from region_rag.posttrain_data import read_json, write_json, sha
from region_rag import training_quality as quality


class TrainingQualityTests(unittest.TestCase):
    def fixture(self, root):
        data = root / 'data'; data.mkdir()
        image = root / 'query.jpg'; image.write_bytes(b'fixed query contents')
        cases = [{'query_id': q, 'image_path': str(image), 'scenario': s}
                 for q in ('train_1', 'train_2') for s in ('closed', 'open')]
        write_json(data / 'train.json', cases)
        write_json(data / 'train_pairs.json', [dict(query_id=q, images=[str(image), 'product'], label=True)
                                             for q in ('train_1', 'train_2')])
        for name in ('val.json', 'test.json'):
            (data / name).write_bytes(b'[  {"preserve":"exact bytes"}  ]\n')
        write_json(data / 'manifest.json', {'files': {n: sha(data/n) for n in
                   ('train.json', 'train_pairs.json', 'val.json', 'test.json')}})
        review = root / 'review.json'
        write_json(review, {'source_manifest_sha256': sha(data/'manifest.json'), 'exclusions': [
            {'query_id': 'train_1', 'image_sha256': sha(image), 'reason': 'insufficient_visible_shape',
             'evidence': 'Reviewed query does not reveal the discriminating shape.'}]})
        return data, review, image

    def test_filter_removes_all_pairs_and_triplets_for_query_but_not_other_queries(self):
        rows = [{'query_id':'train_1', 'label':True}, {'query_id':'train_1', 'label':False},
                {'query_id':'train_2', 'negative_identity':'same_model_but_visible_difference'}]
        kept, removed = quality.filter_training(rows, {'train_1'})
        self.assertEqual(kept, [rows[2]])
        self.assertEqual(removed, rows[:2])
        with self.assertRaises(ValueError):
            quality.filter_training([{'query_id':'test_1'}], {'train_1'})

    def test_export_is_reproducible_preserves_source_and_records_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data,review,_=self.fixture(root)
            before={p.name:p.read_bytes() for p in data.iterdir()}
            report=quality.export_clean_training(data, review, root/'clean')
            self.assertEqual(report['queries_after'], 1)
            self.assertEqual(len(read_json(root/'clean/train_pairs.json')), 1)
            self.assertEqual(len(read_json(root/'clean/quarantine.json')['train_cases']), 2)
            self.assertEqual(before, {p.name:p.read_bytes() for p in data.iterdir()})
            quality.export_clean_training(data, review, root/'clean')
            self.assertEqual(report, read_json(root/'clean/report.json'))

    def test_review_cannot_be_applied_to_changed_images_or_another_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data,review,image=self.fixture(root)
            self.assertEqual(quality.load_exclusions(data, review), {'train_1'})
            image.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'image'):
                quality.load_exclusions(data, review)
            r=read_json(review); r['source_manifest_sha256']='wrong'; write_json(review,r)
            with self.assertRaisesRegex(ValueError, 'manifest'):
                quality.load_exclusions(data, review)

    def test_review_cannot_silently_ignore_unknown_or_evaluation_queries(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data,review,_=self.fixture(root)
            for qid in ('train_missing', 'val_1'):
                r=read_json(review); r['exclusions'][0]['query_id']=qid; write_json(review,r)
                with self.assertRaises(ValueError): quality.load_exclusions(data, review)


if __name__ == '__main__': unittest.main()
