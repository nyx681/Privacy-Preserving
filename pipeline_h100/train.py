"""One image/text optimization stage. Atomic checkpoints and resumable evaluation."""
import argparse,fcntl,json,os,random,time,traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from common import read,save,sha,repair_jsonl,rewrite_jsonl
from data import Spec
from engine import Engine
import runtime

HERE=Path(__file__).resolve().parent

def fingerprint(config,keys):
 return dict(config_sha=sha(config),keys_sha=sha(keys) if keys else None,code={n:sha(HERE/n) for n in ['train.py','engine.py','data.py','common.py','runtime.py']},torch=torch.__version__)

def evaluate(engine,spec,out,label,split,image_label,heldout=False):
 tasks=spec.evaluation_tasks(split,heldout);path=out/f'{label}.json';sp=out/f'{label}_summary.json'
 if path.exists() and sp.exists():
  rows=read(path);assert len(rows)==len(tasks) and all(all(r[k]==v for k,v in t.items()) for r,t in zip(rows,tasks))
  return read(sp)
 partial=out/f'{label}.jsonl';rows=repair_jsonl(partial)
 assert len(rows)<=len(tasks) and all(all(r[k]==v for k,v in t.items()) for r,t in zip(rows,tasks))
 saved=engine.load_images(out,image_label)
 with partial.open('a',encoding='utf-8') as f,torch.no_grad():
  for task in tasks[len(rows):]:
   if runtime.stopped():raise SystemExit(130)
   row=engine.response(task,saved);rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
   save(out/'status.json',dict(stage='evaluating',label=label,completed=len(rows),total=len(tasks),pid=os.getpid()))
 summary=spec.summarize(rows);save(path,rows);save(sp,summary);partial.unlink(missing_ok=True);engine.prefix_cache.clear()
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--kind',choices=['image','text'],required=True);p.add_argument('--keys');p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 runtime.install_handlers();cfg=read(a.config);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);started=time.time()
 lock=(out/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')));torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
 spec=Spec(cfg,a.kind,a.keys);finger=fingerprint(a.config,a.keys)
 if (out/'fingerprint.json').exists():assert read(out/'fingerprint.json')==finger,'Configuration/source differs; use the original files or a new output directory'
 save(out/'fingerprint.json',finger)
 if (out/'completed.json').exists():lock.close();return
 def status(stage,**kw):
  r=dict(stage=stage,pid=os.getpid(),seconds=time.time()-started,time=datetime.now().astimezone().isoformat(),**kw);save(out/'status.json',r);print(json.dumps(r,ensure_ascii=False),flush=True)
 step=0;best=None;best_step=None;last_selected=-1
 try:
  if runtime.stopped():raise SystemExit(130)
  status('loading_model');e=Engine(spec,cfg['model'],torch.device('cuda:0'));opt=torch.optim.Adam(e.delta,lr=cfg['lr']);checkpoint=out/'checkpoint_latest.pt'
  if checkpoint.exists():
   state=runtime.load_checkpoint(checkpoint);assert state['fingerprint']==finger
   with torch.no_grad():
    for d,v in zip(e.delta,state['delta']):d.copy_(v.to(e.device))
   opt.load_state_dict(state['optimizer']);step=state['step'];best=state['best'];best_step=state['best_step'];last_selected=state['last_selected']
   for name,content in state['best_images'].items():(out/name).write_bytes(content)
   if state['best_record'] is not None:save(out/'best.json',state['best_record'])
   runtime.restore_rng(state['rng']);status('resumed',step=step,best_step=best_step)
   log=out/'training_log.jsonl';rewrite_jsonl(log,[r for r in repair_jsonl(log) if r['step']<=step])
   for f in out.glob('development_*'):
    try:n=int(f.name.split('_')[1].split('.')[0])
    except ValueError:continue
    if n>step:f.unlink()
  def select(n):
   nonlocal best,best_step,last_selected
   e.save_images(out,'latest');summary=evaluate(e,spec,out,f'development_{n:04d}','validation','latest');score=[summary['balanced_accuracy'],-summary['weighted_ce']]
   if best is None or tuple(score)>tuple(best):
    best=score;best_step=n;e.save_images(out,'best');save(out/'best.json',dict(step=n,score=score,summary=summary))
   last_selected=n
  def save_checkpoint():
   names=['protected_best.png']+(['key_best.png'] if a.kind=='image' else [])
   state=dict(fingerprint=finger,step=step,delta=[d.detach().cpu() for d in e.delta],optimizer=opt.state_dict(),best=best,best_step=best_step,last_selected=last_selected,best_record=read(out/'best.json') if best is not None else None,best_images={n:(out/n).read_bytes() for n in names if (out/n).exists()},rng=runtime.rng_state())
   runtime.atomic_torch_save(checkpoint,state)
  if runtime.stopped():raise SystemExit(130)
  if not checkpoint.exists():e.save_images(out,'initial');save_checkpoint()
  total=1 if a.smoke else cfg['steps']
  due=lambda n:n==0 or n==1 or n%cfg['eval_every']==0 or n==total
  if not a.smoke and due(step) and last_selected!=step:select(step);save_checkpoint()
  for idx in range(step,total):
   if runtime.stopped():save_checkpoint();raise SystemExit(130)
   t0=time.time();tasks=spec.training_tasks(idx);grads=[torch.zeros_like(x) for x in e.delta];details=[]
   assert abs(sum(t['weight'] for t in tasks)-1)<1e-8
   for task in tasks:
    gg,detail=e.gradient(task);details.append(detail)
    for acc,g in zip(grads,gg):
     if g is not None:acc.add_(g,alpha=task['weight'])
   opt.zero_grad(set_to_none=True)
   for d,g in zip(e.delta,grads):d.grad=g
   opt.step()
   with torch.no_grad():
    for x,d in zip(e.x,e.delta):d.clamp_(-cfg['epsilon'],cfg['epsilon']);d.copy_((x+d).clamp(0,1)-x)
   step=idx+1;e.frozen();e.prefix_cache.clear()
   row=dict(step=step,seconds=time.time()-started,update_seconds=time.time()-t0,conditions=details)
   with (out/'training_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
   status('training',step=step,total=total)
   # Persist before validation, so an interruption inside validation can resume it.
   if step%cfg['checkpoint_every']==0 or due(step) or runtime.stopped():save_checkpoint()
   if runtime.stopped():raise SystemExit(130)
   if a.smoke:
    e.save_images(out,'smoke');saved=e.load_images(out,'smoke');rows=[e.response(t,saved) for t in tasks];e.frozen();save(out/'completed.json',dict(smoke=True,steps=1,rows=rows));return
   if due(step):select(step);save_checkpoint()
  e.save_images(out,'final')
  for ck in ['best','final']:
   for split in ['train','test','test_extra']:
    evaluate(e,spec,out,f'{ck}_{split}_known',split,ck)
    evaluate(e,spec,out,f'{ck}_{split}_heldout','train_subset' if split=='train' else split,ck,True)
  e.frozen();save(out/'completed.json',dict(steps=step,best_step=best_step,seconds=time.time()-started,frozen_model=True));status('completed',step=step,best_step=best_step)
 except SystemExit:
  status('interrupted',step=step,checkpoint_step=runtime.load_checkpoint(out/'checkpoint_latest.pt')['step'] if (out/'checkpoint_latest.pt').exists() else None);raise
 except Exception:
  save(out/'failure.json',dict(traceback=traceback.format_exc()));status('failed',step=step);raise
 finally:lock.close()
if __name__=='__main__':main()
