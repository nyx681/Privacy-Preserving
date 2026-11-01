"""Resumable evaluation of unseen easy keys on existing protected best/final PNGs."""
import argparse,json,os,time
from pathlib import Path
import torch
from common import read,save,sha
from data import Spec
from easy_keys import build_easy,VERSION,SEED
from engine import Engine
from report import report
import runtime

def run(config,keys,out):
 runtime.install_handlers()
 out=Path(out);root=out.parent.parent;cfg=read(config);spec=Spec(cfg,'text',keys)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True,use_fast=True)
 protocol=build_easy(spec.keys,tok,cfg['target'],cfg.get('field','identifier'),cfg.get('answer_components'))
 protocol.update(config_sha256=sha(config),keys_sha256=sha(keys))
 pp=out/'easy_protocol.json'
 if pp.exists():assert read(pp)==protocol,'Easy-key protocol changed; preserve original evaluation'
 else:save(pp,protocol)
 tasks_by_split={}
 for split in ['train','test','test_extra']:
  qs=spec.questions[split]
  if split=='train':qs=[qs[i] for i in spec.subset]
  tasks_by_split[split]=[dict(**spec.task(q,c),case_index=i*36+j) for i,q in enumerate(qs) for j,c in enumerate(protocol['cases'])]
 pending=[]
 for ck in ['best','final']:
  for split,tasks in tasks_by_split.items():
   label=f'{ck}_{split}_easy';path=out/f'{label}.json';sp=out/f'{label}_summary.json'
   if path.exists() and sp.exists():
    rows=read(path);assert len(rows)==len(tasks) and all(all(r[k]==v for k,v in t.items()) for r,t in zip(rows,tasks))
   else:pending.append((ck,split,tasks))
 finger=dict(torch_version=torch.__version__,protocol_sha256=sha(pp),protected_best_sha256=sha(out/'protected_best.png'),protected_final_sha256=sha(out/'protected_final.png'),code={n:sha(Path(__file__).parent/n) for n in ['evaluate_easy.py','easy_keys.py','engine.py','data.py','common.py','runtime.py']})
 fp=out/'easy_fingerprint.json'
 if fp.exists():assert read(fp)==finger,'Easy evaluation images/protocol/source changed'
 else:save(fp,finger)
 report(root)
 if not pending:return
 torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')));engine=Engine(spec,cfg['model'],torch.device('cuda:0'));started=time.time()
 for ck,split,tasks in pending:
  label=f'{ck}_{split}_easy';path=out/f'{label}.json';sp=out/f'{label}_summary.json';partial=out/f'{label}.jsonl';rows=[]
  # Reuse only byte-identical protected PNGs and identical task sets.
  other='final' if ck=='best' else 'best';op=out/f'{other}_{split}_easy.json'
  if op.exists() and sha(out/f'protected_{other}.png')==sha(out/f'protected_{ck}.png'):
   rows=read(op)
  elif partial.exists():
   for i,line in enumerate(partial.read_text(encoding='utf-8').splitlines()):
    try:rows.append(json.loads(line))
    except json.JSONDecodeError:
     if i!=len(partial.read_text(encoding='utf-8').splitlines())-1:raise
  assert len(rows)<=len(tasks) and all(all(r[k]==v for k,v in t.items()) for r,t in zip(rows,tasks))
  with partial.open('w',encoding='utf-8') as f:
   for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  saved=engine.load_images(out,ck)
  with partial.open('a',encoding='utf-8') as f,torch.no_grad():
   for task in tasks[len(rows):]:
    if runtime.stopped():
     save(out/'easy_status.json',dict(stage='interrupted',label=label,completed=len(rows),total=len(tasks)))
     raise SystemExit(130)
    rows.append(engine.response(task,saved));f.write(json.dumps(rows[-1],ensure_ascii=False)+'\n');f.flush()
    save(out/'easy_status.json',dict(stage='evaluating',label=label,completed=len(rows),total=len(tasks),pid=os.getpid(),seconds=time.time()-started))
  save(path,rows);save(sp,spec.summarize(rows));partial.unlink(missing_ok=True);engine.prefix_cache.clear();report(root)
 save(out/'easy_completed.json',dict(version=VERSION,seed=SEED,groups=6,queries=2160))
 save(out/'easy_status.json',dict(stage='completed',pid=os.getpid(),seconds=time.time()-started))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--keys',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.config,a.keys,a.out)
