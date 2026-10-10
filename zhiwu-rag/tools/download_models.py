"""Download explicitly; web requests never initiate Hugging Face downloads."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/'models/base_manifest.json').read_text())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def download(info, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file() and digest(target)==info['sha256']:
        print('Already verified:',target); return
    if target.exists(): raise ValueError(f'Existing file has the wrong hash; keep it aside before retrying: {target}')
    part=target.with_suffix(target.suffix+'.part')
    subprocess.run(['curl','--fail','--location','--retry','3','--continue-at','-',
                    '--output',str(part),info['url']],check=True)
    if digest(part)!=info['sha256']: raise ValueError(f'Download hash mismatch: {part}')
    part.replace(target)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('part',choices=['vision','vlm']); args=parser.parse_args()
    if args.part=='vision':
        download(MANIFEST['dino'],Path(os.environ.get('DINO_WEIGHTS',ROOT/'models/base/dino_vitbase8_pretrain.pth')).expanduser())
        download(MANIFEST['clip'],Path.home()/'.cache/clip/ViT-B-16.pt')
    else:
        from huggingface_hub import snapshot_download
        cache=Path(os.environ.get('HF_HUB_CACHE',Path.home()/'.cache/huggingface/hub'))
        for model in MANIFEST['huggingface']:
            ref=cache/('models--'+model['repo_id'].replace('/','--'))/'refs/main'
            if ref.exists() and ref.read_text().strip()!=model['revision']:
                raise ValueError(f'Existing main cache has another revision: {ref}. Set HF_HUB_CACHE to a separate empty directory in .env.')
            path=snapshot_download(model['repo_id'],revision=model['revision'],cache_dir=str(cache),
                 allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE*','NOTICE*'],max_workers=2)
            # Evaluated model code loads the named model offline at main; bind it to this pinned snapshot.
            ref.parent.mkdir(parents=True,exist_ok=True); ref.write_text(model['revision'])
            print('Prepared:',path)

if __name__=='__main__': main()
