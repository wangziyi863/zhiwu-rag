"""Validated catalog and atomic, pickle-free cosine index."""
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

ENCODER_ID = 'openai-ViT-B-16-standard-cls-224-rgb-v1'


def load_catalog(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as f:
        if path.suffix.lower() == '.csv':
            rows = list(csv.DictReader(f))
        else:
            rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError('知识库为空，请先导入带图片和来源的资料。')
    seen, items = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('每个条目必须是 JSON 对象。')
        for key in ('item_id', 'name', 'image_path', 'source'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'条目缺少必填字段 {key}。')
        item_id = row['item_id'].strip()
        if item_id in seen:
            raise ValueError(f'重复 item_id: {item_id}')
        seen.add(item_id)
        image_path = Path(row['image_path']).expanduser()
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        image_path = image_path.resolve()
        with Image.open(image_path) as image:
            image.verify()
        attrs = row.get('attributes') or {}
        if isinstance(attrs, str):
            attrs = json.loads(attrs)
        if not isinstance(attrs, dict):
            raise ValueError(f'{item_id}: attributes 必须为对象。')
        item = {key: str(row.get(key) or '') for key in ('name', 'description', 'source', 'category')}
        item.update(item_id=item_id, image_path=str(image_path), attributes=attrs,
                    image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest())
        if isinstance(row.get('provenance'), dict):
            item['provenance'] = row['provenance']
        items.append(item)
    return items


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError('向量包含非有限数值。')
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('向量不能为空或全零。')
    return vectors / norm


def save_index(directory, items, vectors):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or not items or len(items) != len(vectors):
        raise ValueError('索引条目与向量数量不一致或为空。')
    vectors = normalize(vectors)
    metadata = json.dumps({'encoder': ENCODER_ID, 'items': items}, ensure_ascii=False)
    fd, temp_path = tempfile.mkstemp(prefix='index-', suffix='.npz', dir=directory)
    os.close(fd)
    try:
        np.savez_compressed(temp_path, vectors=vectors, metadata=np.asarray(metadata))
        os.replace(temp_path, directory / 'index.npz')
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def read_index(directory):
    path = Path(directory) / 'index.npz'
    if not path.is_file():
        raise ValueError('尚未建立知识库索引。请先运行 index 命令。')
    with np.load(path, allow_pickle=False) as data:
        vectors = data['vectors'].copy()
        metadata = json.loads(str(data['metadata']))
    if metadata.get('encoder') != ENCODER_ID or len(metadata['items']) != len(vectors):
        raise ValueError('索引版本或条目不匹配，请重新建立索引。')
    return metadata['items'], normalize(vectors)


def search_index(directory, query, top_k=3, categories=None, min_similarity=-1):
    items, vectors = read_index(directory)
    query = normalize(np.asarray(query).reshape(-1))
    if vectors.shape[1] != query.shape[0]:
        raise ValueError('查询向量与索引维度不匹配。')
    if not 1 <= top_k <= 20:
        raise ValueError('top_k 必须在 1 到 20 之间。')
    similarities = vectors @ query
    order = [idx for idx in np.argsort(-similarities, kind='stable')
             if similarities[idx] >= min_similarity and
             (categories is None or items[int(idx)].get('category') in categories)][:top_k]
    return [dict(items[int(idx)], evidence_id=f'E{rank+1}', similarity=float(similarities[idx]))
            for rank, idx in enumerate(order)]
