"""Image key -> discrete text search -> three matched text-key training experiments."""
import argparse,fcntl,json,os,random,re,string,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from common import read,save,sha
from data import Spec,build_keys
from engine import Engine,load_rgb
from report import report
import torch

HERE=Path(__file__).resolve().parent;REPO=HERE.parent

def arguments():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--image',help='One redacted image; omit with --all-new')
 p.add_argument('--all-new',action='store_true');p.add_argument('--target');p.add_argument('--questions');p.add_argument('--id')
 p.add_argument('--registry',default=str(HERE/'registry.json'))
 p.add_argument('--model',required=True);p.add_argument('--basis',required=True);p.add_argument('--normal-dir',required=True)
 p.add_argument('--normal-pools',default=str(REPO/'data/image/negative_pools.json'))
 p.add_argument('--model-identity',default=str(REPO/'data/model_identity.json'))
 p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=2000);p.add_argument('--search-steps',type=int,default=200)
 p.add_argument('--seed',type=int,default=42);p.add_argument('--eval-every',type=int,default=100)
 p.add_argument('--check-only',action='store_true');p.add_argument('--smoke-only',action='store_true')
 return p.parse_args()

def stage(card,name,cmd,root):
 save(root/'pipeline_status.json',dict(stage='running',image=card.name,operation=name,pid=os.getpid(),time=datetime.now().astimezone().isoformat()))
 with (card/f'{name}.log').open('a',encoding='utf-8') as log:
  worker=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  last=None
  while True:
   # Refresh each completed evaluation group, including the original training worker.
   signature=[(str(p),p.stat().st_mtime_ns) for p in root.glob('*/*/*_summary.json')]
   if signature!=last:report(root);last=signature
   try:rc=worker.wait(timeout=10);break
   except subprocess.TimeoutExpired:pass
 if rc:raise RuntimeError(f'{card.name}/{name} failed ({rc}); see {card/name} and {card/(name+".log")}')
 report(root)

def prepare(a,record,root):
 image=Path(record['image']).resolve();questions=Path(record['questions']).resolve();card=root/record['id'];card.mkdir(parents=True,exist_ok=True)
 with Image.open(image) as im:size=im.size
 cfg=dict(id=record['id'],image=str(image),input_sha256=sha(image),target=record['target'],field=record.get('field','identifier'),questions=str(questions),questions_sha256=sha(questions),
          model=str(Path(a.model).resolve()),model_identity_sha256=sha(a.model_identity),model_identity=str(Path(a.model_identity).resolve()),basis=str(Path(a.basis).resolve()),basis_sha256=sha(a.basis),normal_dir=str(Path(a.normal_dir).resolve()),normal_pools=str(Path(a.normal_pools).resolve()),normal_pools_sha256=sha(a.normal_pools),
          key_init=str(card/'key_initial.png'),canvas_wh=list(size),encoder_wh=[896,672],resize='Original resolution RGB optimization; direct bicubic resize to 896x672, including aspect change; identical for all methods',steps=a.steps,search_steps=a.search_steps,seed=a.seed,search_seed=20261028,key_schedule_seed=20261029,subset_seed=20261030,random_key_seed=20261031,lr=1/255,epsilon=32/255,eval_every=a.eval_every,text_initialization='zero delta from original redacted input',image_initialization='0.2 redacted input + 0.8 resized shared basis',prompt='natural question; optional literal key line; no permission or format instructions',selection='development balanced exact-match, then lower CE; no test selection',evaluation='train60 known and correct; fixed train10 heldout; test10 and test_extra10 all conditions; best and final',target_verification=record.get('verification','User-supplied target'))
 if (card/'config.json').exists():assert read(card/'config.json')==cfg,'Input/config changed: select a new output directory'
 else:save(card/'config.json',cfg)
 if not (card/'key_initial.png').exists():
  im=load_rgb(image);basis=load_rgb(a.basis).resize(im.size,Image.Resampling.BICUBIC)
  # Integer arithmetic reproduces the earlier 20/80 mixture exactly.
  arr=(np.asarray(im,dtype=np.uint16)+4*np.asarray(basis,dtype=np.uint16)+2)//5
  Image.fromarray(arr.astype(np.uint8)).save(card/'key_initial.png')
 provenance=dict(input_sha256=cfg['input_sha256'],basis_sha256=cfg['basis_sha256'],key_sha256=sha(card/'key_initial.png'))
 if (card/'fusion.json').exists():assert read(card/'fusion.json')==provenance
 save(card/'fusion.json',provenance)
 return card,cfg

def random_keys(tok,found,cfg):
 rng=random.Random(cfg['random_key_seed']);char_key=''.join(rng.choice(string.ascii_letters+string.digits) for _ in range(len(found)))
 count=len(tok.encode(found,add_special_tokens=False))
 pool=[i for i in range(len(tok)) if i not in tok.all_special_ids and re.fullmatch(r' [A-Za-z]{2,20}',tok.decode([i]))]
 for _ in range(10000):
  ids=[rng.choice(pool) for _ in range(count)];token_key=tok.decode(ids)
  if tok.encode(token_key,add_special_tokens=False)==ids and token_key!=found:break
 else:raise RuntimeError('Could not generate token-count-matched roundtrip text')
 assert len(char_key)==len(found) and char_key!=found
 assert cfg['target'] not in char_key and cfg['target'] not in token_key, 'Regenerate random seed if answer is accidentally embedded'
 return dict(searched_text=found,random_chars=char_key,random_tokens=token_key)

def main():
 a=arguments();assert a.steps>=4 and a.search_steps>=1
 torch.set_num_threads(4)
 root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
 lock=(root/'.pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 registry=read(a.registry)
 if a.all_new:
  assert not a.image
  records=[dict(r,image=str(REPO/r['image']),questions=str(REPO/r['questions'])) for r in registry.values()]
 else:
  if not a.image:raise ValueError('Supply --image or --all-new')
  image=Path(a.image).resolve();record=dict(registry.get(image.name,{}))
  if a.target is not None:record['target']=a.target
  if a.questions:record['questions']=str(Path(a.questions).resolve())
  if not record.get('target') or not record.get('questions'):raise ValueError('Unregistered redacted image requires --target and --questions; target cannot be inferred from mosaic pixels')
  record.update(id=a.id or record.get('id') or image.stem,image=str(image));record['questions']=str(REPO/record['questions']);records=[record]
 for record in records:assert re.fullmatch(r'[A-Za-z0-9_-]+',record['id'])
 identity=read(a.model_identity)
 index=read(Path(a.model)/'model.safetensors.index.json')
 assert all((Path(a.model)/n).is_file() for n in set(index['weight_map'].values())),'Model shard missing'
 for n,digest in identity['files'].items():assert sha(Path(a.model)/n)==digest,f'Model metadata mismatch: {n}'
 for group in read(a.normal_pools).values():
  for r in group:
   assert Path(r['filename']).stem not in {v['id'] for v in records},'A target card occurs in normal-negative pool'
   assert sha(Path(a.normal_dir)/r['filename'])==r['file_sha256']
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True)
 prepared=[]
 for r in records:
  card,cfg=prepare(a,r,root);spec=Spec(cfg,'image')
  # Check exact native processor compatibility at each input resolution, without weights.
  e=Engine(spec,a.model,torch.device('cpu'),load_weights=False);del e
  save(card/'preflight.json',dict(image_sha256=cfg['input_sha256'],size=cfg['canvas_wh'],target=cfg['target'],question_counts={k:len(v) for k,v in spec.questions.items()},processor_equivalence=True,known_image_cases={k:len(spec.evaluation_tasks(k)) for k in ['train','test','test_extra']},heldout_text_expected=dict(train=360,test=360,test_extra=360)))
  prepared.append((card,cfg))
 report(root)
 if a.check_only:save(root/'check_completed.json',dict(images=len(prepared)));return
 if a.smoke_only:
  card,cfg=prepared[0];key=' smoke privacy student number test access validation token example'
  save(card/'smoke_keys.json',build_keys(key,cfg['steps'],cfg['key_schedule_seed'],tok))
  for kind in ['image','text']:
   cmd=[sys.executable,str(HERE/'train.py'),'--config',str(card/'config.json'),'--kind',kind,'--out',str(card/f'smoke_{kind}'),'--smoke']
   if kind=='text':cmd+=['--keys',str(card/'smoke_keys.json')]
   stage(card,f'smoke_{kind}',cmd,root)
  save(root/'smoke_completed.json',dict(real_model_image_and_text=True));return
 failures=[]
 for card,cfg in prepared:
  try:
   def train(method,kind,keypath=None):
    cmd=[sys.executable,str(HERE/'train_extended.py'),'--config',str(card/'config.json'),'--kind',kind,'--out',str(card/method)]
    if keypath:cmd+=['--keys',str(keypath)]
    stage(card,method,cmd,root)
   train('image_key','image')
   stage(card,'text_search',[sys.executable,str(HERE/'search.py'),'--config',str(card/'config.json'),'--source',str(card/'image_key'),'--out',str(card/'text_search')],root)
   found=(card/'text_search/best_key.txt').read_text(encoding='utf-8');keys=random_keys(tok,found,cfg)
   keyinfo=[]
   for method,key in keys.items():
    f=card/f'{method}_keys.json'
    if f.exists():assert read(f)['key']==key
    else:save(f,build_keys(key,cfg['steps'],cfg['key_schedule_seed'],tok))
    (card/f'{method}_key.txt').write_text(key,encoding='utf-8')
    keyinfo.append(dict(method=method,key=key,characters=len(key),tokens=len(tok.encode(key,add_special_tokens=False)),random_key_seed=cfg['random_key_seed']))
   save(card/'key_comparison.json',keyinfo)
   for method in keys:train(method,'text',card/f'{method}_keys.json')
   save(card/'completed.json',dict(stages=['image_key','text_search',*keys],time=datetime.now().astimezone().isoformat()))
  except Exception:
   failure=dict(image=card.name,traceback=traceback.format_exc());failures.append(failure);save(card/'pipeline_failure.json',failure)
  report(root)
 if failures:
  save(root/'pipeline_status.json',dict(stage='failed',failures=failures));raise RuntimeError('Some image pipelines failed; inspect per-image pipeline_failure.json; successful stages are resumable')
 save(root/'completed.json',dict(images=[c.name for c,_ in prepared],time=datetime.now().astimezone().isoformat()))
 save(root/'pipeline_status.json',dict(stage='completed',images=[c.name for c,_ in prepared]))
if __name__=='__main__':main()
