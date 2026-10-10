"""Conservative, evidence-bound quarantine of ambiguous training queries.

This applies reviewed exclusions; it does not automatically certify all remaining
images as identifiable, and never edits evaluation cases or the original data.
"""
import argparse
from pathlib import Path
from .posttrain_data import read_json, write_json, sha, verify_manifest
from .detail_core import bind_config


def load_exclusions(data, review):
    data = Path(data)
    verify_manifest(data)
    policy = read_json(review)
    if policy['source_manifest_sha256'] != sha(data/'manifest.json'):
        raise ValueError('Quality review source manifest mismatch')
    queries = {}
    for case in read_json(data/'train.json'):
        queries.setdefault(case['query_id'], set()).add(case['image_path'])
    excluded = set()
    for item in policy['exclusions']:
        qid = item['query_id']
        if not qid.startswith('train_') or qid not in queries:
            raise ValueError(f'Review must refer to an existing training query: {qid}')
        if qid in excluded or not item.get('reason') or not item.get('evidence'):
            raise ValueError(f'Duplicate or undocumented review: {qid}')
        if len(queries[qid]) != 1 or sha(next(iter(queries[qid]))) != item['image_sha256']:
            raise ValueError(f'Reviewed query image changed: {qid}')
        excluded.add(qid)
    return excluded


def filter_training(rows, excluded):
    kept, removed = [], []
    for row in rows:
        if not row['query_id'].startswith('train_'):
            raise ValueError('Quality filter only accepts training queries')
        (removed if row['query_id'] in excluded else kept).append(row)
    return kept, removed


def export_clean_training(data, review, out, experiment=None):
    data, review, out = Path(data), Path(review), Path(out)
    excluded = load_exclusions(data, review)
    config = {'source_manifest_sha256': sha(data/'manifest.json'), 'review_sha256': sha(review),
              'code_sha256': sha(__file__)}
    sources = {'train_cases': data/'train.json', 'train_pairs': data/'train_pairs.json'}
    if experiment:
        experiment = Path(experiment)
        prep = read_json(experiment/'prepare/config.json')
        mining = read_json(experiment/'mining/config.json')
        if (prep['manifest'] != config['source_manifest_sha256'] or
                mining['experiment']['manifest'] != config['source_manifest_sha256'] or
                mining['pool'] != sha(experiment/'prepare/pool.json')):
            raise ValueError('Mining data provenance mismatch')
        sources.update(pool=experiment/'prepare/pool.json', triplets=experiment/'mining/triplets.json')
    config['inputs'] = {name:sha(path) for name,path in sources.items()}
    bind_config(out, config)
    quarantine, cleaned, counts = {}, {}, {}
    names = {'train_cases':'train.json', 'train_pairs':'train_pairs.json',
             'pool':'pool.json', 'triplets':'triplets.json'}
    for name, path in sources.items():
        rows = read_json(path)
        kept, removed = filter_training(rows, excluded)
        if not kept: raise ValueError(f'No usable {name} after quarantine')
        cleaned[name], quarantine[name] = kept, removed
        counts[name] = {'before': len(rows), 'after': len(kept), 'excluded': len(removed)}
        write_json(out/names[name], kept)
    write_json(out/'review.json', read_json(review))
    write_json(out/'quarantine.json', quarantine)
    evaluation = {name: sha(data/name) for name in ('val.json', 'test.json')}
    report = {'queries_before': len({r['query_id'] for r in read_json(data/'train.json')}),
              'queries_after': len({r['query_id'] for r in cleaned['train_cases']}),
              'excluded_queries': sorted(excluded), 'counts': counts,
              'unchanged_evaluation_sha256': evaluation, 'source_manifest_sha256':config['source_manifest_sha256'],
              'review_sha256':config['review_sha256'],
              'scope':'Known reviewed ambiguities only; remaining queries are not certified clean. '
                      'Original image files, gallery, validation and test are unchanged. '
                      'Filtered triplets retain the original mining choices; no model was retrained.'}
    write_json(out/'report.json', report)
    lines = ['# 训练样本清理结果', '',
             f"训练查询：{report['queries_before']} → {report['queries_after']}。", '',
             '| 内容 | 清理前 | 清理后 | 隔离 |', '|---|---:|---:|---:|']
    lines += [f"| {name} | {v['before']} | {v['after']} | {v['excluded']} |" for name,v in counts.items()]
    lines += ['', '## 隔离依据', '']
    lines += [f"- {r['query_id']}：{r['evidence']}" for r in read_json(review)['exclusions']]
    lines += ['', '仅应用已复查的歧义清单，不代表剩余样本全部无歧义。原图、原数据、商品库和评测集保留。',
              '这是训练数据导出目录，不是完整 DATA 数据集。继续使用原 DATA，并通过 QUALITY_REVIEW 指定清单。',
              '已有 DPO 父模型曾使用原训练集；清理只影响下一轮训练，不能消除父模型已经学到的影响。',
              '尚未重新训练，不能据此宣称准确率提高。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--experiment')
    args = parser.parse_args()
    report = export_clean_training(args.data, args.review, args.out, args.experiment)
    print(f"Training queries: {report['queries_before']} -> {report['queries_after']}")
    print(report['counts'])


if __name__ == '__main__': main()
