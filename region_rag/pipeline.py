"""Sequential workers free GPU memory at process boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import re
from collections import Counter

from PIL import Image, ImageDraw, ImageOps

from .catalog import read_index
from .core import validate_bbox
from .workflow import split_request, summarize_stages, ATTRIBUTES
from .match_state import apply_match_state,uncertain
from .repair_core import should_repair, repair_target, masked_crop, mask_overlay, suspected_merge, has_selector
import numpy as np

REPO = Path(__file__).resolve().parents[1]


class Pipeline:
    def __init__(self, home=None):
        self.home = Path(home or os.environ.get('GLA_RAG_HOME', REPO/'rag_data')).resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.index = self.home/'index'
        self.jobs = self.home/'jobs'
        self.jobs.mkdir(exist_ok=True)
        self.gla_python = os.environ.get('GLA_PYTHON', sys.executable)
        self.vlm_python = os.environ.get('RAG_PYTHON', str(REPO/'.venv-rag/bin/python'))
        self.dpo_python = os.environ.get('DPO_PYTHON',str(REPO/'.venv-posttrain/bin/python'))
        self.dpo_root = os.environ.get('RAG_DPO_ROOT',str(REPO/'models/dpo'))
        self.e_root = os.environ.get('RAG_E_ROOT',str(REPO/'models/clean_e'))
        self.rerank_model = os.environ.get('RAG_RERANK_MODEL','clean_e')
        if self.rerank_model not in ('clean_e','dpo'): raise ValueError('Unknown reranking model')

    def new_job(self):
        job = self.jobs/(time.strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
        job.mkdir()
        return job

    def worker(self, module, request, job, python):
        if not Path(python).is_file():
            raise RuntimeError(f'模型运行环境未就绪：{python}。请运行 tools/setup_region_rag.sh。')
        task = request['task']
        request = dict(request, output=str(job))
        path = job/f'{task}_request.json'
        path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['CUDA_VISIBLE_DEVICES'] = env.get('GPU', env.get('CUDA_VISIBLE_DEVICES', '0'))
        # Never download large model files implicitly during a browser request.
        if module.endswith(('.vlm','.repair','.dpo_rerank','.match_verify','.e_prepare','.e_rerank')):
            env['HF_HUB_OFFLINE'] = '1'
        log_path = job/f'{task}.log'
        begin = time.perf_counter()
        with log_path.open('w') as log:
            proc = subprocess.run([python, '-m', module, '--request', str(path)],
                                  cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f'{task} 阶段未完成，详情已保存到：{log_path}')
        value = json.loads((job/f'{task}_result.json').read_text())
        value['worker_wall_seconds'] = time.perf_counter()-begin
        return value

    def build_index(self, catalog):
        job = self.new_job()
        return self.worker('region_rag.vision', {'task':'index', 'catalog':str(Path(catalog).expanduser().resolve()),
                            'index':str(self.index)}, job, self.gla_python)

    def status(self):
        try:
            items, _ = read_index(self.index)
            count = len(items)
        except (ValueError, OSError):
            count = 0
            items = []
        return {'items': count, 'home': str(self.home),
                'categories':dict(Counter(i.get('category','unknown') for i in items)),
                'vlm_environment': Path(self.vlm_python).is_file(),
                'reranker':'e_pairwise_rank' if self.rerank_model=='clean_e' else 'dpo_yes_no',
                'reranker_label':'E 组（清理后）＋三态核验' if self.rerank_model=='clean_e' else 'DPO 重排＋三态核验',
                'match_states':['possible_match','clearly_different','insufficient_evidence'],
                'note': '问答模型权重须通过 setup 脚本提前下载；环境存在不代表权重已就绪。'}

    def rerank_candidates(self,request,job):
        if self.rerank_model=='clean_e':
            preparation=self.worker('region_rag.e_prepare',dict(request,task='prepare_e',policy_root=self.e_root),job,self.gla_python)
            ranked=self.worker('region_rag.e_rerank',dict(request,task='rerank',policy_root=self.e_root,
                               local_map=preparation['local_map']),job,self.dpo_python)
            ranked['local_preparation']=preparation
        else:
            ranked=self.worker('region_rag.dpo_rerank',dict(request,task='rerank',policy_root=self.dpo_root),
                               job,self.dpo_python)
        audit=ranked.get('audit',[])
        if not audit:
            if ranked.get('hits'): raise RuntimeError('重排缺少评分记录，无法进入同款核验。')
            return ranked  # No ranked candidate to inspect.
        hit=next(h for h in request['hits'] if h['item_id']==audit[0]['item_id'])
        try:
            visual=self.worker('region_rag.match_verify',{'task':'verify_match','crop':request['crop'],
                               'candidate_image':hit['image_path']},job,self.vlm_python)
        except (RuntimeError,OSError,ValueError) as error:
            visual=uncertain('视觉核验未完成，暂不能可靠确认同款。')
            visual['verification_error']=str(error)
        result=apply_match_state(ranked,hit,visual)
        with Image.open(hit['image_path']) as im:
            im=im.convert('RGB'); im.thumbnail((640,640)); im.save(job/'match_candidate.jpg')
        (job/'match_result.json').write_text(json.dumps(result['match'],ensure_ascii=False,indent=2),encoding='utf-8')
        return result

    def check_region(self, request, region, job, image):
        box=validate_bbox(region['bbox'],image.size)
        crop=job/'proposed_target.jpg'
        extra={}
        if region.get('mask_path'):
            with Image.open(region['mask_path']) as im: mask=np.asarray(im)>0
            masked_crop(image,mask,box).save(crop)
            overlay=job/'repair_overlay.jpg'; mask_overlay(image,mask).save(overlay)
            extra={'mask':region['mask_path'],'overlay':str(overlay)}
        else:
            image.crop(box).save(crop)
        return self.worker('region_rag.vlm',dict(request,task='verify_instance',crop=str(crop),region=region,**extra),
                           job,self.vlm_python)

    def repair_selection(self, request, selected, job, image):
        audit={'status':'failed','attempts':[]}; start=time.perf_counter()
        for attempt in (1,2):
            sub=job/f'repair_{attempt}'; sub.mkdir()
            entry={'attempt':attempt}; audit['attempts'].append(entry)
            try:
                repaired=self.worker('region_rag.repair',dict(request,task='repair',attempt=attempt,
                    target=repair_target(request['question'],selected),region=selected.get('region',{})),sub,self.vlm_python)
                entry['generation']=repaired
                if repaired['status']=='unavailable':
                    audit.update(status='unavailable',reason=repaired['reason']); break
                if not repaired.get('regions'): continue
                if len(repaired['regions'])>1 and not has_selector(request['question']):
                    audit.update(status='needs_clarification',reason='自动拆分得到多个候选，请说明目标的位置或外观。'); break
                choice=self.worker('region_rag.vlm',dict(request,task='select',regions=repaired['regions'],
                                    instances_only=True),sub,self.vlm_python)
                entry['selection']=choice
                if choice['status']!='selected': continue
                # Only an actual generated mask candidate can enter the repaired path.
                candidates={r['id']:r for r in repaired['regions']}
                if choice.get('region',{}).get('id') not in candidates: continue
                choice['region']=candidates[choice['region']['id']]
                check=self.check_region(request,choice['region'],sub,image)
                entry['verification']=check
                if check['status']=='verified':
                    audit.update(status='repaired',seconds=time.perf_counter()-start)
                    return choice,check,audit
            except (RuntimeError,ValueError,OSError) as error:
                entry['error']=str(error)
                # Missing environments or execution errors cannot be corrected by a larger ROI.
                audit.update(status='error',reason=str(error)); break
        audit['seconds']=time.perf_counter()-start
        return None,None,audit

    def ask(self, image_path, question, compression='location', retrieval_mode='region', retrieval_only=False,repair_enabled=True,
            dual_recall=True,detailed_categories=False,recall_k=12):
        index_error=None
        try: read_index(self.index)
        except (ValueError,OSError) as error: index_error=str(error)
        if compression not in ('none','location','sim08') or retrieval_mode not in ('region','whole'):
            raise ValueError('不支持的压缩或检索配置。')
        if type(recall_k) is not int or not 1<=recall_k<=100: raise ValueError('召回数量须为 1–100。')
        if not isinstance(question, str) or not question.strip() or len(question)>2000:
            raise ValueError('请输入不超过 2000 字符的问题。')
        job = self.new_job()
        begin = time.perf_counter()
        with Image.open(image_path) as im:
            if im.width*im.height > 24_000_000:
                raise ValueError('图片超过 2400 万像素，请先缩小后上传。')
            image = ImageOps.exif_transpose(im).convert('RGB')
        image.thumbnail((1600,1600))
        canonical = job/'input.png'
        image.save(canonical)
        request = {'image':str(canonical), 'question':question.strip(), 'index':str(self.index),
                   'compression':compression, 'retrieval_mode':retrieval_mode, 'top_k':3,'recall_k':recall_k,
                   'dual_recall':bool(dual_recall),'detailed_categories':bool(detailed_categories)}
        intent=split_request(question.strip())
        perception_request=dict(request,question=intent['target_description'])
        result = {'job_id':job.name, 'question':question.strip(), 'compression':compression,
                  'retrieval_mode':retrieval_mode, 'files':['input.png'],
                  'intent':intent,'repair':{'status':'not_needed' if repair_enabled else 'disabled','attempts':[]}}
        stage='localization'
        try:
            vision = self.worker('region_rag.vision', dict(perception_request,task='segment'), job, self.gla_python)
            result['perception'] = vision
            result['files'].append('regions.jpg')
            selected = self.worker('region_rag.vlm', dict(perception_request,task='select',regions=vision['regions']), job, self.vlm_python)
            result['disambiguation'] = selected
            check=self.check_region(perception_request,selected['region'],job,image) if selected['status']=='selected' else None
            if check: result['instance_verification']=check
            if repair_enabled and selected['status']=='selected':
                parts=suspected_merge(selected['region'],vision['regions'])
                if parts:
                    result['merge_suspicion']={'method':'overlapping_semantic_candidates','candidate_ids':parts}
                    check={'status':'uncertain','method':'geometric_merge_suspicion',
                           'reason':'当前区域与多个物体或错位座椅候选重叠，需要实例拆分复查。'}
            if not check or check['status']!='verified':
                if repair_enabled and should_repair(perception_request['question'],selected,check):
                    fixed,fixed_check,audit=self.repair_selection(perception_request,selected,job,image)
                    result['repair']=audit
                    if fixed:
                        selected,check=fixed,fixed_check
                        result['instance_verification']=check
                elif repair_enabled:
                    result['repair']['status']='needs_clarification'
            if not check or check['status']!='verified':
                reason=(check or selected).get('reason','无法确认唯一目标。')
                if result['repair']['attempts']:
                    reason='自动目标检测与实例复查后仍无法确认目标，本次未继续检索。'+result['repair'].get('reason','可补充更明确的位置或换一个拍摄角度。')
                result.update(status='target_uncertain',answer=reason)
                return self.finish(result, job, begin)
            result['region']=selected['region']  # Preserve successful localization even if retrieval fails.
            annotated=image.copy(); draw=ImageDraw.Draw(annotated)
            draw.rectangle(result['region']['bbox'],outline='#14b8a6',width=4)
            annotated.save(job/'selected.jpg'); result['files'].append('selected.jpg')
            if index_error:
                result.update(status='knowledge_unavailable',knowledge_error=index_error,
                    answer='已定位图中目标，但知识库尚未就绪，当前无法检索商品资料或确认材质等属性。')
                return self.finish(result,job,begin)
            if selected['region'].get('mask_path'):
                with Image.open(selected['region']['mask_path']) as im: mask=np.asarray(im)>0
                Image.fromarray(mask.astype(np.uint8)*255).save(job/'repair_mask.png')
                mask_overlay(image,mask).save(job/'repair_overlay.jpg')
                image.crop(selected['region']['bbox']).save(job/'target_unmasked.jpg')
                result['files'].extend(['repair_mask.png','repair_overlay.jpg','target_unmasked.jpg'])
            stage='retrieval'
            vision = self.worker('region_rag.vision', dict(request,task='retrieve',region=selected['region']), job, self.gla_python)
            result['region'] = vision['region']
            result['recall'] = vision
            result['hits'] = []
            result['files'].append('target.jpg')
            if vision['hits']:
                stage='reranking'
                ranked = self.rerank_candidates(dict(request,crop=str(job/'target.jpg'),hits=vision['hits']),job)
                result['reranking'] = ranked
                result['hits'] = ranked['hits']
                if ranked.get('match'):
                    result['match']=ranked['match']
                    result['files'].append('match_candidate.jpg')
            if not result['hits']:
                if result.get('match'):
                    result.update(status=result['match']['status'],answer='已定位图中目标。'+
                                  result['match']['message']+'\n'+result['match']['reason'])
                else:
                    result.update(status='no_match',answer='已定位图中目标，但本次检索或重排未找到可靠的商品资料，因此无法据此回答属性。检索未命中不代表图库中一定没有合适商品。')
                return self.finish(result,job,begin)
            for hit in result['hits']:
                filename = f'evidence_{hit["evidence_id"]}.jpg'
                with Image.open(hit['image_path']) as evidence:
                    evidence = evidence.convert('RGB')
                    evidence.thumbnail((640,640))
                    evidence.save(job/filename)
                result['files'].append(filename)
            if retrieval_only:
                result.update(status='retrieval_only',answer='已完成单实例检查与检索。当前仅检索模式，跳过资料回答；以下是候选资料，不代表同款确认。')
            else:
                stage='answer'
                answer = self.worker('region_rag.vlm', dict(request,task='answer',hits=result['hits']), job, self.vlm_python)
                result.update(status=answer['status'],answer=answer['answer'],citations=answer['citations'],
                              answer_mode=answer['answer_mode'],generation_stats=answer['stats'])
            return self.finish(result, job, begin)
        except Exception as error:
            labels={'localization':'目标定位','retrieval':'商品检索','reranking':'候选重排','answer':'资料回答'}
            result.update(status='error',error=str(error),failure_stage=stage,
                answer=('已定位图中目标。' if result.get('region') else '')+labels[stage]+'执行失败，详情已记录。'+str(error))
            return self.finish(result, job, begin)

    def finish(self, result, job, begin):
        result['stages']=summarize_stages(result)
        knowledge=result['stages']['knowledge']
        if result.get('status') in ('answered','answered_with_fallback') and knowledge['status']=='attribute_unverified':
            result['generation_status']=result['status']
            result['status']='attribute_unverified'
            result['answer']='已定位图中目标，并检索到了相似商品。'+knowledge['message']+'\n'+result['answer']
        elif result.get('status') in ('answered','answered_with_fallback') and result.get('intent',{}).get('requested_attributes'):
            if not re.search(r'候选|相似|类似|比较|对比|similar|candidate|compare',result['question'],re.I):
                attributes='、'.join(ATTRIBUTES[k][0] for k in result['intent']['requested_attributes'])
                result['answer']='已定位图中目标，但尚未确认同款，因此不能确定这件物体的'+attributes+'。以下仅列出相似候选的来源资料。\n'+result['answer']
        result['total_wall_seconds'] = time.perf_counter()-begin
        result['result_path'] = str(job/'result.json')
        (job/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result
