"""Engineering-only tests; tiny fake Engine results never enter experiment outputs."""
import contextlib,io,json,os,random,signal,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
import common,runtime,train
from data import build_keys
from easy_keys import build_easy
from launch_one import acquire
from report import report

class Tokenizer:
 def encode(self,s,add_special_tokens=False):return list(map(ord,s))
 def __call__(self,ss,add_special_tokens=False):
  from types import SimpleNamespace
  return SimpleNamespace(input_ids=[self.encode(s) for s in ss])

class FakeEngine:
 interrupt_after=None
 def __init__(self,spec,model,device):
  self.spec=spec;self.device=torch.device('cpu');self.x=[torch.ones(1)*.5 for _ in range(2 if spec.kind=='image' else 1)];self.delta=[torch.zeros(1,requires_grad=True) for x in self.x];self.prefix_cache={};self.calls=0
  self.weight=torch.nn.Parameter(torch.rand(1),requires_grad=False);self.version=self.weight._version
 def gradient(self,t):
  self.calls+=1
  if self.interrupt_after and self.calls==self.interrupt_after:runtime.STOP_REQUESTED=True
  z=sum(self.delta)+self.weight
  # Deliberately consume all RNG types to verify complete restoration.
  goal=.1*torch.rand(1)+.1*np.random.rand()+.1*random.random()
  loss=((z-goal)*(1 if t['authorized'] else 2)).square().sum()
  return torch.autograd.grad(loss,self.delta),dict(condition=t['condition'],ce=float(loss.detach()))
 def frozen(self):assert self.weight._version==self.version and not self.weight.requires_grad
 def save_images(self,out,label):
  for x,d,name in zip(self.x,self.delta,['protected','key']):
   v=int(round(float((x+d).detach())*255));Image.new('RGB',(2,2),(v,v,v)).save(out/f'{name}_{label}.png')
 def load_images(self,out,label):return [Image.open(out/f'protected_{label}.png').copy()]
 def response(self,t,saved):
  answer=self.spec.target if t['authorized'] else '无法恢复'
  return dict(**t,response=answer,ce=float(torch.rand(1)),**{**common.metrics(answer,self.spec.target),'exact_match':answer==t['target']})

class Tests(unittest.TestCase):
 def test_metrics_keep_email_domain_and_components(self):
  self.assertFalse(common.metrics('1370077873@gmail.com','1370077873@qq.com','email')['leaks_any_component'])
  self.assertFalse(common.metrics('2763 97330','2763 NW DAYLILY AVE CORVALLIS OR 97330','address')['leaks_any_component'])
  self.assertTrue(common.metrics('编号 A0000001','A0000001','student_card_number')['leaks_any_component'])
  self.assertFalse(common.metrics('73330','073330','student_card_number')['leaks_any_component'])
  m=common.metrics('ROBERT DAVID HARPER','ROBERT DAVID HARPER\nALAN JAMES WALLER','name',['ROBERT DAVID HARPER','ALAN JAMES WALLER']);self.assertTrue(m['leaks_any_component']);self.assertFalse(m['contains_id_normalized'])
 def test_non_degenerate_search_keys(self):
  self.assertFalse(common.key_split_compatible('a'*200))
  self.assertTrue(common.key_split_compatible(' '.join(['!']*32)))
  tok=Tokenizer();key=' access verification ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  self.assertTrue(common.key_split_compatible(key));d=build_keys(key,2000,20261029,tok);self.assertEqual(len(d['known']),16)
 def test_easy_disjoint_and_far(self):
  tok=Tokenizer();d=build_keys(' privacy access ABCDEFGHIJKLMNOPQRSTUVWXYZ 12345',2000,42,tok);e=build_easy(d,tok,'073330','student_card_number')
  banned={r['key'] for row in d['schedule'] for r in row}|{r['key'] for k in ['development','heldout'] for r in d[k]}
  self.assertEqual(len(e['cases']),36);self.assertTrue(all(r['key'] not in banned and r['normalized_edit_distance']>=.5 for r in e['cases']))
 def test_checkpoint_resume_matches_uninterrupted_optimizer(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);image=root/'image.png';Image.new('RGB',(2,2),'white').save(image);q=root/'questions.json';groups={}
   for split,n in [('train',60),('validation',2),('test',10),('test_extra',10)]:groups[split]=[dict(id=f'{split}{i}',split=split,question=f'{split} question {i}') for i in range(n)]
   common.save(q,groups);pool=root/'pools.json';common.save(pool,dict(train=[],validation=[],test=[]));cfg=root/'config.json'
   common.save(cfg,dict(image=str(image),input_sha256=common.sha(image),key_init=str(image),questions=str(q),questions_sha256=common.sha(q),target='073330',steps=8,seed=42,subset_seed=8,normal_dir=str(root),normal_pools=str(pool),model='fake-engineering-only',lr=1/255,epsilon=32/255,eval_every=3,checkpoint_every=3))
   def run(out,interrupt=None):
    FakeEngine.interrupt_after=interrupt;runtime.STOP_REQUESTED=False
    with patch.object(train,'Engine',FakeEngine),patch.object(runtime,'install_handlers'),patch.object(sys,'argv',['train.py','--config',str(cfg),'--kind','image','--out',str(out)]),contextlib.redirect_stdout(io.StringIO()):train.main()
   run(root/'full')
   with self.assertRaises(SystemExit):run(root/'resume',19) # interrupt during fourth six-condition update
   state=runtime.load_checkpoint(root/'resume/checkpoint_latest.pt');self.assertEqual(state['step'],4)
   run(root/'resume');a=runtime.load_checkpoint(root/'full/checkpoint_latest.pt');b=runtime.load_checkpoint(root/'resume/checkpoint_latest.pt')
   self.assertEqual((a['step'],a['last_selected'],a['best_step']),(b['step'],b['last_selected'],b['best_step']))
   for x,y in zip(a['delta'],b['delta']):self.assertTrue(torch.equal(x,y))
   for k in a['optimizer']['state']:
    for n,x in a['optimizer']['state'][k].items():self.assertTrue(torch.equal(x,b['optimizer']['state'][k][n]))
   self.assertTrue(torch.equal(a['rng']['torch'],b['rng']['torch']));self.assertEqual(a['rng']['python'],b['rng']['python']);self.assertTrue(np.array_equal(a['rng']['numpy'][1],b['rng']['numpy'][1]));self.assertEqual(a['best_images'],b['best_images'])
   self.assertEqual([r['step'] for r in common.repair_jsonl(root/'resume/training_log.jsonl')],list(range(1,9)))
 def test_jsonl_truncated_tail(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'eval.jsonl';p.write_text('{"case_index":0}\n{"case_index":');rows=common.repair_jsonl(p);self.assertEqual(rows,[dict(case_index=0)]);self.assertEqual(len(p.read_text().splitlines()),1)
 def test_gpu_lock_survives_supervisor_exit(self):
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'gpu.lock';f=acquire(path,'GPU')
   child=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.read()'],stdin=subprocess.PIPE,pass_fds=(f.fileno(),));f.close()
   try:
    with self.assertRaises(RuntimeError):acquire(path,'GPU')
    other=acquire(Path(td)/'other.lock','Other GPU');other.close()
   finally:child.communicate(timeout=5)
   f=acquire(path,'GPU');f.close()
 def test_incremental_per_image_reports_and_refusal(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   for cid in ['card_a','card_b']:common.save(root/cid/'config.json',dict(id=cid,steps=2000))
   out=root/'card_a/searched_text';out.mkdir();common.save(out/'best.json',dict(step=100))
   common.save(out/'best_test_easy_summary.json',dict(positive_total=dict(n=0,exact_recovery=0,normalized_recovery=0),negative_total=dict(n=360,leaks_id=3,exact_refusal=350,other=7),easy_random=dict(n=120,leaks_id=3)))
   report(root);r=next(r for r in common.read(root/'card_a/summary.json')['results'] if r['method']=='searched_text' and r['checkpoint']=='best' and r['split']=='test')
   self.assertEqual(r['easy_leak'],'3/360 (0.8%)');self.assertEqual(r['hard_leak'],'未完成');self.assertTrue((root/'card_a/response_breakdown.csv').exists());self.assertIn('未完成',(root/'card_b/summary.md').read_text())
if __name__=='__main__':unittest.main()
