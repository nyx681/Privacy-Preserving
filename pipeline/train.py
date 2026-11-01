"""One frozen-model stage, resumable optimization and exhaustive scheduled evaluation."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse,fcntl,json,time,traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from common import read,save,sha
from data import Spec
from engine import Engine

def evaluate(engine,spec,out,label,split,image_label,heldout=False):
 tasks=spec.evaluation_tasks(split,heldout);path=out/f'{label}.json';summary_path=out/f'{label}_summary.json'
 if path.exists() and summary_path.exists():
  rows=read(path);assert len(rows)==len(tasks)
  assert all(all(row[k]==v for k,v in task.items()) for row,task in zip(rows,tasks))
  return read(summary_path)
 partial=out/f'{label}.jsonl';rows=[]
 if partial.exists():
  lines=partial.read_text(encoding='utf-8').splitlines()
  for i,line in enumerate(lines):
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:
    if i!=len(lines)-1:raise
  with partial.open('w',encoding='utf-8') as f:
   for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 assert all(all(row[k]==v for k,v in task.items()) for row,task in zip(rows,tasks))
 saved=engine.load_images(out,image_label)
 with partial.open('a',encoding='utf-8') as f,torch.no_grad():
  for task in tasks[len(rows):]:
   row=engine.response(task,saved);rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
   save(out/'status.json',dict(stage='evaluating',label=label,completed=len(rows),total=len(tasks),pid=os.getpid(),time=datetime.now().astimezone().isoformat()))
 summary=spec.summarize(rows);save(path,rows);save(summary_path,summary);partial.unlink(missing_ok=True)
 engine.prefix_cache.clear()
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--kind',choices=['image','text'],required=True);p.add_argument('--keys');p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 cfg=read(a.config);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);started=time.time()
 lock=(out/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
 spec=Spec(cfg,a.kind,a.keys)
 finger=dict(config_sha=sha(a.config),keys_sha=sha(a.keys) if a.keys else None,code={n:sha(Path(__file__).parent/n) for n in ['train.py','engine.py','data.py','common.py']},torch=torch.__version__)
 if (out/'fingerprint.json').exists():assert read(out/'fingerprint.json')==finger,'Configuration/source changed: use a new output directory'
 save(out/'fingerprint.json',finger)
 if (out/'completed.json').exists():return
 def status(stage,**extra):
  record=dict(stage=stage,pid=os.getpid(),seconds=time.time()-started,time=datetime.now().astimezone().isoformat(),**extra);save(out/'status.json',record);print(json.dumps(record,ensure_ascii=False),flush=True)
 try:
  status('loading_model');e=Engine(spec,cfg['model'],torch.device('cuda:0'))
  opt=torch.optim.Adam(e.delta,lr=cfg['lr']);step=0;best=None;best_step=None;ck=out/'checkpoint_latest.pt'
  if ck.exists():
   state=torch.load(ck,map_location='cpu');assert state['fingerprint']==finger
   with torch.no_grad():
    for delta,value in zip(e.delta,state['delta']):delta.copy_(value.to(e.device))
   opt.load_state_dict(state['optimizer']);step=state['step'];best=state['best'];best_step=state['best_step']
   for name,content in state['best_images'].items():(out/name).write_bytes(content)
   save(out/'best.json',state['best_record'])
   # Development rows after this checkpoint are stale if a crash occurred before saving it.
   for f in out.glob('development_*'):
    try:n=int(f.stem.split('_')[1])
    except ValueError:continue
    if n>step:f.unlink()
  def select(n):
   e.save_images(out,'latest');summary=evaluate(e,spec,out,f'development_{n:04d}','validation','latest')
   nonlocal best,best_step
   score=[summary['balanced_accuracy'],-summary['weighted_ce']]
   if best is None or tuple(score)>tuple(best):
    best=score;best_step=n;e.save_images(out,'best');save(out/'best.json',dict(step=n,score=score,summary=summary))
  def checkpoint():
   names=['protected_best.png']+(['key_best.png'] if a.kind=='image' else [])
   state=dict(fingerprint=finger,step=step,delta=[x.detach().cpu() for x in e.delta],optimizer=opt.state_dict(),best=best,best_step=best_step,best_record=read(out/'best.json'),best_images={n:(out/n).read_bytes() for n in names})
   torch.save(state,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(ck)
  if not ck.exists():
   e.save_images(out,'initial')
   if not a.smoke:select(0);checkpoint()
  total=1 if a.smoke else cfg['steps']
  for idx in range(step,total):
   t0=time.time();tasks=spec.training_tasks(idx);grad=[torch.zeros_like(x) for x in e.delta];details=[]
   assert abs(sum(t['weight'] for t in tasks)-1)<1e-8
   for task in tasks:
    gg,detail=e.gradient(task);details.append(detail)
    for acc,g in zip(grad,gg):
     if g is not None:acc.add_(g,alpha=task['weight'])
   opt.zero_grad(set_to_none=True)
   for delta,g in zip(e.delta,grad):delta.grad=g
   opt.step()
   with torch.no_grad():
    for x,delta in zip(e.x,e.delta):delta.clamp_(-cfg['epsilon'],cfg['epsilon']);delta.copy_((x+delta).clamp(0,1)-x)
   step=idx+1;e.frozen();e.prefix_cache.clear()
   row=dict(step=step,seconds=time.time()-started,update_seconds=time.time()-t0,conditions=details)
   with (out/'training_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
   status('training',step=step,total=total,update_seconds=row['update_seconds'])
   if a.smoke:
    e.save_images(out,'smoke');saved=e.load_images(out,'smoke');rs=[e.response(t,saved) for t in tasks];e.frozen()
    save(out/'completed.json',dict(smoke=True,steps=1,gradient_and_png_and_frozen_checks=True,rows=rs));return
   if step==1 or step%cfg['eval_every']==0 or step==total:select(step)
   if step==1 or step%50==0 or step==total:checkpoint()
  e.save_images(out,'final')
  for image_label in ['best','final']:
   for split in ['train','test','test_extra']:
    evaluate(e,spec,out,f'{image_label}_{split}_known',split,image_label)
    heldsplit='train_subset' if split=='train' else split
    evaluate(e,spec,out,f'{image_label}_{split}_heldout',heldsplit,image_label,True)
  e.frozen();save(out/'completed.json',dict(steps=step,best_step=best_step,seconds=time.time()-started,frozen_model=True))
  status('completed',step=step,best_step=best_step)
 except Exception:
  save(out/'failure.json',dict(traceback=traceback.format_exc()));status('failed');raise
if __name__=='__main__':main()
