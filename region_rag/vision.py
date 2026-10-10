"""GPU worker running exclusively in the existing GLA-CLIP environment."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms

from .catalog import load_catalog, save_index, search_index, ENCODER_ID
from .core import TARGETS, extract_regions, parse_intent, select_region, validate_bbox
from .ranking import choose_categories
from .repair_core import masked_crop
from .recall import CATEGORY_DESCRIPTIONS, resolve_categories, select_candidates

REPO = Path(__file__).resolve().parents[1]
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def rgb_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def load_clip():
    from open_clip import create_model
    return create_model('ViT-B/16', pretrained='openai', precision='fp16').eval().cuda()


@torch.inference_mode()
def standard_clip_batch(clip, images):
    """Use all original residual blocks and CLS pooling, bypassing dense forward.

    This repository's encode_image replaces the last block for segmentation;
    calling it would not yield a standard CLIP image embedding.
    """
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    x = torch.stack([preprocess(image) for image in images]).to(device='cuda', dtype=clip.visual.conv1.weight.dtype)
    visual = clip.visual
    x = visual.conv1(x).flatten(2).transpose(1, 2)
    cls = visual.class_embedding.to(x.dtype).reshape(1, 1, -1).expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1) + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x).transpose(0, 1)
    for block in visual.transformer.resblocks:
        x = block(x)
    result = visual.ln_post(x[0]) @ visual.proj
    return F.normalize(result.float(), dim=-1).cpu().numpy()


def standard_clip_embedding(clip, image):
    return standard_clip_batch(clip, [image])[0]


@torch.inference_mode()
def category_scores(clip, vector, detailed=False):
    from open_clip import tokenize
    categories = list(TARGETS)
    phrases=[p for c in categories for p in (CATEGORY_DESCRIPTIONS[c] if detailed else (c,))]
    tokens = tokenize([f'a photo of a {c}.' for c in phrases]).cuda()
    embeddings = F.normalize(clip.encode_text(tokens).float(),dim=-1).cpu().numpy()
    values=(embeddings @ vector).reshape(len(categories),-1).max(axis=1)
    return dict(zip(categories,values.astype(float).tolist()))


def draw_regions(image, regions, selected=None):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for r in regions:
        color = '#14b8a6' if selected == r['id'] else '#ffb020'
        draw.rectangle(r['bbox'], outline=color, width=max(2, image.width//300))
        label = f"{r['id']} {r['class_name']}"
        x, y = r['bbox'][:2]
        draw.rectangle((x, y, min(image.width, x+len(label)*7+4), y+16), fill=color)
        draw.text((x+2, y+2), label, fill='black')
    return image


@torch.inference_mode()
def segment(request, out):
    from cfg import ModelConfig
    from gla_clip_segmentor import GLA_CLIPSegmentation
    from mmseg.structures import SegDataSample
    mode = request.get('compression', 'location')
    if mode not in ('none', 'location', 'sim08'):
        raise ValueError('compression must be none, location or sim08')
    name_path = REPO / 'configs/cls_ade20k.txt'
    mc = ModelConfig(name_path=str(name_path), work_dir=str(out), token_norm=True,
                     KV_token_extension=True, proxy_sim=True, mini_iters=2,
                     initial_crit_pos=.6, dynamic_beta=True, dynamic_gamma=True,
                     beta_alpha=.3, gamma_alpha=30, overlap_compression=mode != 'none',
                     compression_mode='sim' if mode == 'sim08' else mode,
                     merge_tau=.8, count_adapt=True)
    model = GLA_CLIPSegmentation(clip_type='openai', model_type='ViT-B/16', vfm_model='dino',
                                 name_path=str(name_path), model_cfg=mc).eval()
    image = rgb_image(request['image'])
    w, h = image.size
    ratio = min(336/min(w, h), 1344/max(w, h))
    rw, rh = int(w*ratio+.5), int(h*ratio+.5)
    if min(rw, rh) < 224:
        raise ValueError('图片长宽比过大，当前滑窗适配器不支持该全景图。')
    resized = image.resize((rw, rh), Image.Resampling.BILINEAR)
    x = transforms.ToTensor()(resized)
    x = transforms.Normalize(MEAN, STD)(x).unsqueeze(0).cuda()
    sample = SegDataSample(metainfo=dict(ori_shape=(h,w), img_shape=(rh,rw),
                                       img_path=str(request['image']), padding_size=[0,0,0,0]))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    result = model.predict(x, [sample])[0]
    torch.cuda.synchronize()
    segment_seconds = time.perf_counter() - begin
    mask = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
    Image.fromarray(mask).save(out / 'semantic_mask.png')
    names = name_path.read_text().splitlines()
    allowed = {c for _, classes in TARGETS.values() for c in classes}
    allowed.update(('computer', 'monitor', 'bottle', 'book', 'clock', 'fan', 'microwave', 'oven', 'person', 'animal'))
    regions = extract_regions(mask, names, allowed, min_area=max(50, int(w*h*.001)))
    intent = parse_intent(request['question'])
    selection = select_region(regions, intent, (w,h))
    stats = {'predict_seconds': segment_seconds,
             'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
             'compression': mode, 'kv': getattr(mc, '_kv_stats', None),
             'original_size': [w,h], 'inference_size': [rw,rh]}
    selected = selection.get('region')
    draw_regions(image, regions, selected['id'] if selected else None).save(out / 'regions.jpg')
    response = {'status': selection['status'], 'intent': intent, 'selection': selection,
                'regions': regions, 'stats': stats}
    # Geometry/category selection is only a proposal. Always release this worker
    # and verify instance uniqueness before any crop is embedded for retrieval.
    response['requires_instance_verification'] = True
    return response


def retrieve(request, out, clip, region, image=None):
    image = image or rgb_image(request['image'])
    box = validate_bbox(region['bbox'], image.size)
    crop = image.crop(box)
    if region.get('mask_path'):
        with Image.open(region['mask_path']) as im: mask=np.asarray(im)>0
        crop=masked_crop(image,mask,box)
    crop.save(out / 'target.jpg')
    query_image = image if request.get('retrieval_mode') == 'whole' else crop
    vector = standard_clip_embedding(clip, query_image)
    detailed=bool(request.get('detailed_categories',False)); dual=bool(request.get('dual_recall',False))
    scores = category_scores(clip, standard_clip_embedding(clip,crop) if request.get('retrieval_mode')=='whole' else vector,detailed)
    categories = resolve_categories(region['class_name'],scores,request.get('question',''),detailed)
    threshold = float(request.get('min_similarity',.20))
    from .catalog import read_index
    items,vectors=read_index(request['index'])
    hits=select_candidates(items,vectors@vector,categories,int(request.get('recall_k',12)),threshold,dual)
    return {'status': 'retrieved', 'region': region, 'hits': hits,
            'categories':categories,'category_scores':scores,'min_similarity':threshold,
            'encoder': ENCODER_ID, 'retrieval_mode': request.get('retrieval_mode', 'region'),
            'dual_recall':dual,'detailed_categories':detailed,'recall_k':int(request.get('recall_k',12)),
            'prompt_aggregation':'max over 4 equally weighted descriptions' if detailed else 'single description'}


def build_index(request):
    items = load_catalog(request['catalog'])
    directory = Path(request['index']).resolve()
    assets = directory / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    clip = load_clip()
    vectors = []
    for start in range(0,len(items),32):
        batch=items[start:start+32]
        images=[rgb_image(item['image_path']) for item in batch]
        vectors.append(standard_clip_batch(clip,images))
        for item,image in zip(batch,images):
            # Freeze evidence content so later source-file changes cannot alter an index.
            image.thumbnail((1600,1600))
            frozen = assets / (item['image_sha256'] + '.png')
            if not frozen.exists():
                image.save(frozen)
            item['original_image_path'] = item['image_path']
            item['image_path'] = str(frozen)
        print(f'Indexed {min(start+32,len(items))}/{len(items)}',flush=True)
    save_index(directory, items, np.concatenate(vectors))
    return {'status': 'indexed', 'items': len(items), 'index': str(directory), 'encoder': ENCODER_ID}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    out = Path(request['output']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    task = request['task']
    if not torch.cuda.is_available():
        raise RuntimeError('GLA-CLIP 环境中 CUDA 不可用。')
    if task == 'index':
        result = build_index(request)
    elif task == 'segment':
        result = segment(request, out)
    elif task == 'retrieve':
        result = retrieve(request, out, load_clip(), request['region'])
    else:
        raise ValueError(f'Unknown vision task: {task}')
    (out / f'{task}_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'task': task, 'status': result['status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
