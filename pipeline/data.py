"""Common question schedule and disjoint key conditions for all image/text stages."""
import random,string
from pathlib import Path
from collections import Counter
from common import read,save,sha,summarize

def literal_prompt(q,k):return q if k is None else '密钥：'+k+'\n'+q

def question_schedule(questions,steps,seed):
 rng=random.Random(seed);order=[]
 while len(order)<steps:
  block=list(range(len(questions)));rng.shuffle(block);order.extend(block)
 return order[:steps]

def build_keys(key,steps,seed,tokenizer):
 rng=random.Random(seed);N=len(key)
 if N<20:raise ValueError('Key too short for 18 disjoint deletion variants')
 def edit(kind,p,c='~'):return key[:p]+(c+key[p:] if kind=='insert' else key[p+1:] if kind=='delete' else c+key[p+1:])
 def case(kind,p=None,c=None,k=None):return dict(condition=kind,key=edit(kind,p,c or '~') if p is not None else k,position=p,character=c)
 pos=list(range(N));rng.shuffle(pos);deletes=[];seen=set()
 for p in pos:
  v=edit('delete',p)
  if v not in seen:deletes.append(p);seen.add(v)
 if len(deletes)<18:raise ValueError('Too few unique deletions for heldout split')
 spread=[round(i*(N-1)/11) for i in range(12)]
 def replacement(p,char):return char if key[p]!=char else '^'
 held=[case('replace',p,replacement(p,'~')) for p in spread]+[case('insert',p,'~') for p in spread]+[case('delete',p) for p in deletes[:12]]
 dev=[case('replace',p,replacement(p,'|')) for p in spread[::2]]+[case('insert',p,'|') for p in spread[::2]]+[case('delete',p) for p in deletes[12:18]]
 reserved={r['key'] for r in held+dev};assert len(reserved)==54 and key not in reserved
 alphabet=''.join(c for c in string.ascii_letters+string.digits+string.punctuation+' ' if c not in '~|^')
 pools=dict(replace=list(range(N)),insert=list(range(N+1)),delete=[p for p in range(N) if edit('delete',p) not in reserved])
 buffers={k:[] for k in pools};coverage={k:Counter() for k in pools};schedule=[]
 for step in range(steps):
  row=[]
  for kind in pools:
   if not buffers[kind]:buffers[kind]=pools[kind].copy();rng.shuffle(buffers[kind])
   p=buffers[kind].pop();char=None if kind=='delete' else rng.choice([c for c in alphabet if kind!='replace' or c!=key[p]])
   r=case(kind,p,char);assert r['key']!=key and r['key'] not in reserved
   row.append(r);coverage[kind][p]+=1
  kind=['missing','empty','random','truncate'][step%4]
  k=None if kind=='missing' else '' if kind=='empty' else ''.join(rng.choice(string.ascii_letters) for _ in range(N)) if kind=='random' else key[:rng.randrange(1,N)]
  assert k!=key and k not in reserved
  row.append(case(kind,k=k));schedule.append(row)
 known=[]
 for kind in ['replace','delete','insert']:
  candidates={r['key']:r for row in schedule for r in row if r['condition']==kind};known.extend(list(candidates.values())[:4])
 for kind in ['missing','empty','random','truncate']:known.append(next(r for row in schedule for r in row if r['condition']==kind))
 assert len(known)==16
 train={r['key'] for row in schedule for r in row};assert not train.intersection(reserved)
 positive=tokenizer.encode(key,add_special_tokens=False)
 # Complete prompt comparison covers empty/missing boundary as well as the key itself.
 prompts=[literal_prompt('字段号码是多少？',k) for k in train|reserved|{key}]
 ids=tokenizer(prompts,add_special_tokens=False).input_ids
 pi=tuple(tokenizer.encode(literal_prompt('字段号码是多少？',key),add_special_tokens=False))
 assert sum(tuple(x)==pi for x in ids)==1
 assert all(tokenizer.encode(k,add_special_tokens=False)!=positive for k in train|reserved if k is not None)
 return dict(key=key,char_count=N,token_count=len(positive),schedule=schedule,known=known,development=dev,heldout=held,
             coverage={k:dict(v) for k,v in coverage.items()},unique_train_keys=len(train),seed=seed)

class Spec:
 def __init__(self,config,kind,key_path=None):
  self.config=config;self.kind=kind;self.target=config['target'];self.protected=Path(config['image'])
  assert sha(self.protected)==config['input_sha256']
  self.key=Path(config['key_init']);self.questions=read(config['questions'])
  assert sha(config['questions'])==config['questions_sha256']
  assert [len(self.questions[k]) for k in ['train','validation','test','test_extra']]==[60,2,10,10]
  assert len({r['question'] for g in self.questions.values() for r in g})==82
  self.order=question_schedule(self.questions['train'],config['steps'],config['seed'])
  self.subset=sorted(random.Random(config['subset_seed']).sample(range(60),10))
  self.pools=read(config['normal_pools']);self.normal_dir=Path(config['normal_dir'])
  self.keys=read(key_path) if kind=='text' else None
  self.verification_keys=[self.keys['key']]+[r['key'] for group in ['known','development','heldout'] for r in self.keys[group]] if self.keys else []
  if self.keys:assert len(self.keys['schedule'])==config['steps']
  self.schedule=self.order
  self.image_order=[];rng=random.Random(config['seed'])
  for step in range(config['steps']):self.image_order.append(dict(normal_index=rng.randrange(16),blank_value=[0,128,255][step%3],noise_seed=100000+step,normal_split='train'))
 def task(self,q,c,**extra):
  condition=c['condition'];auth=condition in ['correct','matched_pair']
  return dict(question_id=q['id'],question=q['question'],question_split=q['split'],**c,authorized=auth,target=self.target if auth else '无法恢复',prompt=literal_prompt(q['question'],c['key']) if self.kind=='text' else q['question'],**extra)
 def training_tasks(self,step):
  q=self.questions['train'][self.order[step]]
  if self.kind=='text':return [self.task(q,dict(condition='correct',key=self.keys['key']),weight=.5)]+[self.task(q,r,weight=.125) for r in self.keys['schedule'][step]]
  return [self.task(q,dict(condition=c,**self.image_order[step]),weight=.5 if c=='matched_pair' else .1) for c in ['matched_pair','normal_key','blank_key','noise_key','protected_only','key_only']]
 def evaluation_tasks(self,split,heldout=False):
  qs=self.questions['train'] if split=='train_subset' else self.questions[split]
  if split=='train_subset':qs=[qs[i] for i in self.subset]
  rows=[]
  for qi,q in enumerate(qs):
   if self.kind=='text':
    cases=self.keys['heldout'] if heldout else [dict(condition='correct',key=self.keys['key'])]+self.keys['development' if split=='validation' else 'known']
   else:
    if split=='validation':
     cases=[dict(condition='matched_pair'),dict(condition='protected_only'),dict(condition='key_only')]
     cases += [dict(condition='normal_key',normal_split='validation',normal_index=j) for j in range(4)]
     cases += [dict(condition='blank_key',blank_value=v) for v in [0,128,255]]
     cases += [dict(condition='noise_key',noise_seed=700000+2*qi+j) for j in range(2)]
    elif heldout:
     seed={'train_subset':800000,'train':800000,'test':900000,'test_extra':1000000}[split]
     cases=[dict(condition='normal_key',normal_split='test',normal_index=j) for j in range(4)]+[dict(condition='noise_key',noise_seed=seed+qi*2+j) for j in range(2)]
    else:
     cases=[dict(condition='matched_pair'),dict(condition='protected_only'),dict(condition='key_only')]
     cases += [dict(condition='normal_key',normal_split='train',normal_index=j) for j in range(4)]
     cases += [dict(condition='blank_key',blank_value=v) for v in [0,128,255]]
     cases += [dict(condition='noise_key',noise_seed=100000+j) for j in range(2)]
   rows += [self.task(q,c) for c in cases]
  return [dict(**r,case_index=i) for i,r in enumerate(rows)]
 def summarize(self,rows):return summarize(rows)
