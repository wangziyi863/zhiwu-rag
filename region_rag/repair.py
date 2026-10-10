"""Local, sequential Grounding DINO -> SAM 2 repair worker; no runtime downloads."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from .core import validate_bbox
from .repair_core import mask_region

DETECTOR='IDEA-Research/grounding-dino-tiny'
SEGMENTER='facebook/sam2.1-hiera-tiny'


def repair(request,out):
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection,Sam2Processor,Sam2Model
    image=Image.open(request['image']).convert('RGB')
    w,h=image.size; attempt=int(request.get('attempt',1)); roi=[0,0,w,h]
    proposal=request.get('region',{}).get('bbox')
    if attempt==1 and proposal:
        x1,y1,x2,y2=validate_bbox(proposal,image.size)
        dx,dy=max(24,(x2-x1)*.25),max(24,(y2-y1)*.25)
        roi=[max(0,int(x1-dx)),max(0,int(y1-dy)),min(w,int(x2+dx)),min(h,int(y2+dy))]
    view=image.crop(roi); view.save(out/'detection_input.jpg')
    start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    processor=AutoProcessor.from_pretrained(DETECTOR,local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR,local_files_only=True).eval().cuda()
    size={'shortest_edge':1000,'longest_edge':1600} if attempt==2 else {'shortest_edge':800,'longest_edge':1333}
    detection_text={'bed':'bed. folding bed. camping cot.'}.get(request['target'],request['target']+'.')
    inputs=processor(images=view,text=detection_text,size=size,return_tensors='pt').to('cuda')
    with torch.inference_mode(): outputs=model(**inputs)
    result=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=.25,
                text_threshold=.20,target_sizes=[(view.height,view.width)])[0]
    boxes=[]
    # High-IoU duplicate suppression only: genuinely overlapping objects can remain.
    from torchvision.ops import nms
    keep=nms(result['boxes'],result['scores'],.85).cpu().tolist()[:8]
    for idx in keep:
        b=result['boxes'][idx].cpu().tolist()
        b=[min(w,max(0,b[0]+roi[0])),min(h,max(0,b[1]+roi[1])),
           min(w,max(0,b[2]+roi[0])),min(h,max(0,b[3]+roi[1]))]
        try: b=validate_bbox(b,image.size)
        except ValueError: continue
        boxes.append((b,float(result['scores'][idx])))
    del model,processor,inputs,outputs,result
    gc.collect(); torch.cuda.empty_cache()
    if not boxes:
        return {'status':'not_found','regions':[],'roi':roi,'attempt':attempt,'reason':'未找到可用的实例检测框。'}
    processor=Sam2Processor.from_pretrained(SEGMENTER,local_files_only=True)
    model=Sam2Model.from_pretrained(SEGMENTER,local_files_only=True).eval().cuda()
    inputs=processor(images=image,input_boxes=[[b for b,_ in boxes]],return_tensors='pt').to('cuda')
    with torch.inference_mode(): outputs=model(**inputs,multimask_output=True)
    masks=processor.post_process_masks(outputs.pred_masks.cpu(),inputs['original_sizes'].cpu())[0].numpy()
    qualities=outputs.iou_scores.detach().float().cpu().numpy()[0]
    regions=[]; accepted_masks=[]; rejected=[]
    for n,(box,score) in enumerate(boxes):
        best=int(np.argmax(qualities[n])); mask=masks[n,best].astype(bool)
        try:
            if any(np.logical_and(mask,m).sum()/max(1,np.logical_or(mask,m).sum())>.95 for m in accepted_masks):
                raise ValueError('重复实例掩码。')
            path=out/f'instance_{n+1}.png'
            region=mask_region(mask,box,float(qualities[n,best]),request['target'],f'I{n+1}',path)
            Image.fromarray(mask.astype(np.uint8)*255).save(path)
            region['detection_score']=score
            regions.append(region); accepted_masks.append(mask)
        except ValueError as error:
            rejected.append({'box':box,'reason':str(error)})
    torch.cuda.synchronize()
    return {'status':'candidates' if regions else 'not_found','regions':regions,'rejected':rejected,
            'roi':roi,'attempt':attempt,'detector_resize':size,'detection_text':detection_text,'detector':DETECTOR,'segmenter':SEGMENTER,
            'stats':{'seconds':time.perf_counter()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--request',required=True); args=p.parse_args()
    request=json.loads(Path(args.request).read_text()); out=Path(request['output']); out.mkdir(exist_ok=True,parents=True)
    try:
        result=repair(request,out)
    except OSError as error:
        result={'status':'unavailable','regions':[],'reason':'修复模型未就绪，请运行 tools/setup_instance_repair.sh。',
                'error':str(error)}
    (out/'repair_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'regions':len(result['regions'])}))


if __name__=='__main__': main()
