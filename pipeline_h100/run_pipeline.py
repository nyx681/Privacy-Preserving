"""Single-image pipeline, with per-image state and shared incremental result tables."""
import argparse,fcntl,json,os,random,re,signal,string,subprocess,sys,time,traceback
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer
from common import read,save,sha,field_contains,key_split_compatible
from data import Spec,build_keys
from engine import Engine,load_rgb
from report import report
from runtime import inherited_fds
HERE=Path(__file__).resolve().parent
active=None;stopping=False

def stop(signum,frame):
 global stopping
 stopping=True
 if active is not None and active.poll() is None:active.send_signal(signum)

def stage(card,name,cmd,root):
 global active
 if stopping:raise SystemExit(130)
 save(card/'pipeline_status.json',dict(stage='running',operation=name,pid=os.getpid(),time=datetime.now().astimezone().isoformat()))
 with (card/f'{name}.log').open('a',encoding='utf-8') as log:
  active=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=inherited_fds());signature=None
  if stopping:active.terminate()
  while True:
   now=sorted((str(p),p.stat().st_mtime_ns) for p in card.glob('*/*_summary.json'))
   if now!=signature:report(root);signature=now
   try:rc=active.wait(timeout=5);break
   except subprocess.TimeoutExpired:pass
 active=None;report(root)
 if stopping or rc in [130,-signal.SIGINT,-signal.SIGTERM]:raise SystemExit(130)
 if rc:raise RuntimeError(f'{name} failed ({rc}); see {card/(name+".log")}')

def random_keys(tok,key,cfg):
 rng=random.Random(cfg['random_key_seed']);count=len(tok.encode(key,add_special_tokens=False))
 def safe(s):return s!=key and key_split_compatible(s,cfg['key_schedule_seed']) and not any(field_contains(s,c,cfg['field']) for c in cfg['answer_components'])
 for _ in range(10000):
  chars=''.join(rng.choice(string.ascii_letters+string.digits) for _ in range(len(key)))
  if safe(chars):break
 else:raise RuntimeError('Cannot construct random character key')
 pool=[i for i in range(len(tok)) if i not in tok.all_special_ids and re.fullmatch(r' [A-Za-z]{2,20}',tok.decode([i]))]
 for _ in range(10000):
  ids=[rng.choice(pool) for _ in range(count)];text=tok.decode(ids)
  if tok.encode(text,add_special_tokens=False)==ids and safe(text):break
 else:raise RuntimeError('Cannot construct token-count-matched roundtrip key')
 return dict(searched_text=key,random_chars=chars,random_tokens=text)

def arguments():
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--target',required=True);p.add_argument('--questions',required=True);p.add_argument('--id',required=True);p.add_argument('--field',default='identifier');p.add_argument('--components-json',default='[]')
 for name in ['model','basis','normal-dir','normal-pools','model-identity','output']:p.add_argument('--'+name,required=True)
 p.add_argument('--steps',type=int,default=2000);p.add_argument('--search-steps',type=int,default=200);p.add_argument('--eval-every',type=int,default=100);p.add_argument('--checkpoint-every',type=int,default=50);p.add_argument('--seed',type=int,default=42);p.add_argument('--max-new-tokens',type=int,default=64);p.add_argument('--check-only',action='store_true');p.add_argument('--smoke-only',action='store_true')
 return p.parse_args()

def main():
 a=arguments();assert a.steps>=4 and min(a.search_steps,a.eval_every,a.checkpoint_every,a.max_new_tokens)>0;assert re.fullmatch(r'[A-Za-z0-9_-]+',a.id)
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
 torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
 root=Path(a.output).resolve();card=root/a.id;card.mkdir(parents=True,exist_ok=True)
 lock=(card/'.pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 im=load_rgb(a.image);basis=load_rgb(a.basis).resize(im.size,Image.Resampling.BICUBIC)
 fusion=Image.fromarray(((np.asarray(im,dtype=np.uint16)+4*np.asarray(basis,dtype=np.uint16)+2)//5).astype(np.uint8))
 if (card/'key_initial.png').exists():assert np.array_equal(np.asarray(load_rgb(card/'key_initial.png')),np.asarray(fusion)),'Initial key image changed'
 else:fusion.save(card/'key_initial.png')
 cfg=dict(version='h100-single-image-v1',id=a.id,image=str(Path(a.image).resolve()),input_sha256=sha(a.image),target=a.target,answer_components=json.loads(a.components_json) or [a.target],field=a.field,
  questions=str(Path(a.questions).resolve()),questions_sha256=sha(a.questions),model=str(Path(a.model).resolve()),model_identity=str(Path(a.model_identity).resolve()),model_identity_sha256=sha(a.model_identity),basis=str(Path(a.basis).resolve()),basis_sha256=sha(a.basis),normal_dir=str(Path(a.normal_dir).resolve()),normal_pools=str(Path(a.normal_pools).resolve()),normal_pools_sha256=sha(a.normal_pools),key_init=str(card/'key_initial.png'),key_initial_sha256=sha(card/'key_initial.png'),canvas_wh=list(Image.open(a.image).size),encoder_wh=[896,672],steps=a.steps,search_steps=a.search_steps,eval_every=a.eval_every,checkpoint_every=a.checkpoint_every,seed=a.seed,search_seed=20261028,key_schedule_seed=20261029,subset_seed=20261030,random_key_seed=20261031,max_new_tokens=a.max_new_tokens,lr=1/255,epsilon=32/255,text_initialization='zero delta from original redacted input',image_initialization='0.2 redacted input + 0.8 resized shared basis',prompt='natural question with optional literal key line; no permission/format instructions',resize='full-size optimization; bicubic forward resize to 896x672 with matching differentiable proxy',selection='balanced development exact accuracy, CE tie-break; tests excluded from selection',evaluation='train60 correct/known; fixed train10 heldout; test10 + extra10; text easy36 + hard36 per question; best and final')
 if (card/'config.json').exists():assert read(card/'config.json')==cfg,'Configuration/data changed: preserve original settings or use a new OUTPUT_ROOT'
 else:save(card/'config.json',cfg)
 identity=read(a.model_identity)
 for n,digest in identity['files'].items():assert sha(Path(a.model)/n)==digest,f'Model identity mismatch: {n}'
 index=read(Path(a.model)/'model.safetensors.index.json');assert all((Path(a.model)/n).exists() for n in set(index['weight_map'].values()))
 pools=read(a.normal_pools);assert [len(pools[k]) for k in ['train','validation','test']]==[16,4,4]
 for rs in pools.values():
  for r in rs:
   assert Path(r['filename']).stem!=a.id,'Target image appears in negative pool'
   assert sha(Path(a.normal_dir)/r['filename'])==r['file_sha256']
 spec=Spec(cfg,'image');engine=Engine(spec,a.model,torch.device('cpu'),load_weights=False);del engine
 save(card/'preflight.json',dict(input_sha256=cfg['input_sha256'],questions_sha256=cfg['questions_sha256'],processor_equivalence=True,counts={k:len(v) for k,v in spec.questions.items()},canvas_wh=cfg['canvas_wh']))
 report(root)
 if stopping:raise SystemExit(130)
 if a.check_only:save(card/'check_completed.json',dict(input_checks=True));return
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True)
 if a.smoke_only:
  key=' smoke privacy access identity verification example token'
  save(card/'smoke_keys.json',build_keys(key,cfg['steps'],cfg['key_schedule_seed'],tok))
  for kind in ['image','text']:
   cmd=[sys.executable,str(HERE/'train.py'),'--config',str(card/'config.json'),'--kind',kind,'--out',str(card/f'smoke_{kind}'),'--smoke']
   if kind=='text':cmd+=['--keys',str(card/'smoke_keys.json')]
   stage(card,f'smoke_{kind}',cmd,root)
  return
 def train(method,kind,keypath=None):
  cmd=[sys.executable,str(HERE/'train_extended.py'),'--config',str(card/'config.json'),'--kind',kind,'--out',str(card/method)]
  if keypath:cmd+=['--keys',str(keypath)]
  stage(card,method,cmd,root)
 try:
  train('image_key','image')
  stage(card,'text_search',[sys.executable,str(HERE/'search.py'),'--config',str(card/'config.json'),'--source',str(card/'image_key'),'--out',str(card/'text_search')],root)
  found=(card/'text_search/best_key.txt').read_text(encoding='utf-8');keys=random_keys(tok,found,cfg);info=[]
  for method,key in keys.items():
   kp=card/f'{method}_keys.json'
   if kp.exists():assert read(kp)['key']==key
   else:save(kp,build_keys(key,cfg['steps'],cfg['key_schedule_seed'],tok))
   (card/f'{method}_key.txt').write_text(key,encoding='utf-8');info.append(dict(method=method,key=key,characters=len(key),tokens=len(tok.encode(key,add_special_tokens=False))))
  save(card/'key_comparison.json',info)
  for method in keys:train(method,'text',card/f'{method}_keys.json')
  save(card/'completed.json',dict(stages=['image_key','text_search',*keys],time=datetime.now().astimezone().isoformat()))
  save(card/'pipeline_status.json',dict(stage='completed',pid=os.getpid()))
 except SystemExit:
  save(card/'pipeline_status.json',dict(stage='interrupted',pid=os.getpid(),resume='rerun the identical command'));raise
 except Exception:
  save(card/'pipeline_failure.json',dict(traceback=traceback.format_exc()));save(card/'pipeline_status.json',dict(stage='failed',pid=os.getpid()));raise
 finally:report(root)
if __name__=='__main__':main()
