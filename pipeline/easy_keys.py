"""Evaluation-only far-away keys. Never modifies training or checkpoint selection."""
import random,string
from data import literal_prompt

VERSION=1
SEED=20261101

def distance(a,b):
 row=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  nxt=[i]
  for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
  row=nxt
 return row[-1]

def build_easy(keys,tokenizer,target=''):
 key=keys['key'];n=len(key);rng=random.Random(SEED)
 forbidden={key}
 for row in keys['schedule']:forbidden.update(x['key'] for x in row)
 for group in ['known','development','heldout']:forbidden.update(x['key'] for x in keys[group])
 forbidden.discard(None)
 # Full prompts also distinguish the missing and empty conditions.
 forbidden_ids={tuple(tokenizer.encode(literal_prompt('字段号码是多少？',k),add_special_tokens=False)) for k in forbidden|{None}}
 rows=[]
 def accept(kind,k):
  if k in forbidden:return False
  d=distance(key,k)
  if d/max(len(key),len(k))<.5:return False
  digits=''.join(x for x in target if x.isdigit())
  if digits and digits in ''.join(x for x in k if x.isdigit()):return False
  ids=tuple(tokenizer.encode(literal_prompt('字段号码是多少？',k),add_special_tokens=False))
  if ids in forbidden_ids:return False
  forbidden.add(k);forbidden_ids.add(ids)
  rows.append(dict(condition=kind,key=k,edit_distance=d,normalized_edit_distance=d/max(n,len(k))))
  return True
 words=['mountain','garden','river','violet','cloud','silver','forest','window','quiet','orange','bridge','morning','piano','island','winter','lantern','meadow','sunset','harbor','copper','velvet','planet','sparrow','coffee']
 for kind in ['easy_random','easy_phrase','easy_multi_edit']:
  count=0
  for attempt in range(10000):
   if kind=='easy_random':k=''.join(rng.choice(string.ascii_letters+string.digits) for _ in range(n))
   elif kind=='easy_phrase':k=' '.join(rng.sample(words,6))+'.'
   else:
    positions=rng.sample(range(n),max(1,(3*n+4)//5));chars=list(key)
    for p in positions:chars[p]=rng.choice([c for c in string.ascii_letters+string.digits if c!=key[p]])
    k=''.join(chars)
   if accept(kind,k):count+=1
   if count==12:break
  if count!=12:raise RuntimeError('Cannot build 12 disjoint easy keys: '+kind)
 assert len(rows)==36
 return dict(version=VERSION,seed=SEED,difficulty='easy',definition='12 independent random strings; 12 unrelated six-word phrases; 12 keys with at least 60% character substitutions. Every key has normalized Levenshtein distance >=0.5 and is disjoint in literal/full-prompt tokens from training, development and hard heldout.',cases=rows)
