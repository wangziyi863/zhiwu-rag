"""Prepare a traceable subset of ABO using explicit product types, without labels."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import re

SOURCE = 'https://amazon-berkeley-objects.s3.amazonaws.com/index.html'
TYPE_MAP = {
    'CHAIR':'chair', 'BEAN_BAG_CHAIR':'chair', 'SOFA':'sofa',
    'TABLE':'table', 'DESK':'table', 'CABINET':'cabinet', 'DRESSER':'cabinet',
    'BED':'bed', 'BED_FRAME':'bed', 'STOOL_SEATING':'stool', 'OTTOMAN':'stool',
    'SHELF':'shelf', 'BOOKCASE':'shelf', 'LAMP':'lamp', 'LIGHT_FIXTURE':'lamp',
    'HOME_MIRROR':'mirror', 'RUG':'rug', 'PILLOW':'pillow', 'VASE':'vase',
}


def values(field):
    if isinstance(field, str):
        return [field]
    entries = [x for x in (field or []) if isinstance(x, dict) and x.get('value') is not None]
    for language in ('zh_CN', 'en_US', 'en_GB'):
        chosen = [str(x['value']) for x in entries if x.get('language_tag') == language]
        if chosen:
            return list(dict.fromkeys(chosen))
    return list(dict.fromkeys(str(x['value']) for x in entries))


def convert_listing(row, images, shard, line):
    types = values(row.get('product_type'))
    category = next((TYPE_MAP[t] for t in types if t in TYPE_MAP), None)
    image_id = row.get('main_image_id')
    names = values(row.get('item_name'))
    if not category or image_id not in images or not names or not row.get('item_id') or not row.get('domain_name'):
        return None
    attrs = {}
    for key, label in [('material','材质'),('fabric_type','面料'),('color','颜色'),('brand','品牌'),
                       ('model_number','型号'),('finish_type','表面处理'),('style','风格')]:
        vals = values(row.get(key))
        if vals:
            attrs[label] = '；'.join(vals)
    # Keep the original axes and units; never reinterpret length as depth.
    for axis, label in [('height','高度'),('width','宽度'),('length','长度')]:
        dim = (row.get('item_dimensions') or {}).get(axis, {})
        if isinstance(dim,dict) and dim.get('value') is not None and dim.get('unit'):
            attrs[label] = f"{dim['value']} {dim['unit']}"
    description = '\n'.join(values(row.get('bullet_point')) or values(row.get('product_description')))
    description = html.unescape(re.sub(r'<[^>]*>', ' ', description)).strip()
    domain, product_id = row['domain_name'], row['item_id']
    return {'item_id':f'abo:{domain}:{product_id}', 'name':names[0], 'category':category,
            'image_path':images[image_id], 'description':description, 'attributes':attrs,
            'source':SOURCE, 'provenance':{'dataset':'Amazon Berkeley Objects','item_id':product_id,
                'domain_name':domain,'image_id':image_id,'shard':shard,'line':line,
                'product_type':types,'license':'CC BY 4.0','credit':'Amazon.com',
                'record_sha256':hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}}


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    images = {}
    with gzip.open(root/'images/metadata/images.csv.gz','rt',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            path = (root/'images/small'/row['path']).resolve()
            if path.is_relative_to(root/'images/small') and path.is_file():
                images[row['image_id']] = str(path)
    counts, coverage, seen = Counter(), Counter(), set()
    records, scanned, excluded = [], 0, 0
    for shard in sorted((root/'listings/metadata').glob('listings_*.json.gz')):
        with gzip.open(shard,'rt',encoding='utf-8') as f:
            for line, text in enumerate(f,1):
                scanned += 1
                item = convert_listing(json.loads(text),images,shard.name,line)
                if not item:
                    excluded += 1
                    continue
                if item['item_id'] in seen:
                    continue
                seen.add(item['item_id']); records.append(item); counts[item['category']] += 1
                coverage.update(item['attributes'].keys())
    if not records:
        raise ValueError('ABO 中未找到可用的家具主图和元数据，请检查解压目录。')
    output.parent.mkdir(parents=True,exist_ok=True)
    temp = output.with_suffix('.jsonl.tmp')
    with temp.open('w',encoding='utf-8') as f:
        for item in records:
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
    temp.replace(output)
    report={'scanned':scanned,'items':len(records),'categories':dict(counts),'attribute_coverage':dict(coverage),
            'excluded':excluded,'catalog':str(output),'source':SOURCE,'license':'CC BY 4.0',
            'selection':'Exact product_type whitelist; one main image per (domain_name,item_id). Generic types and accessories excluded.'}
    output.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    output.with_suffix('.ATTRIBUTION.txt').write_text(
        'Amazon Berkeley Objects © Amazon.com\nCC BY 4.0: https://creativecommons.org/licenses/by/4.0/\n'
        'Dataset creators: Matthieu Guillaumin, Thomas Dideriksen, Kenan Deng, Himanshu Arora, Jasmine Collins, Jitendra Malik.\n'
        f'Source: {SOURCE}\nChanges: selected product types, selected language fields, removed HTML, joined image metadata; images unchanged.\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description='Prepare ABO furniture/decor catalog')
    p.add_argument('--root',required=True); p.add_argument('--output',required=True)
    args=p.parse_args()
    print(json.dumps(prepare(args.root,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
