"""Online wrong keys for dynamic AMK; audit actual train/evaluation separation."""
from functools import lru_cache
import random
import string
from itertools import count

SEED=20261029
PROBE='What is the private field?'
ALPHABET=''.join(c for c in string.ascii_letters+string.digits+string.punctuation+' ' if c not in '~|^')


def prompt(q,key):return q+(key if key is not None else '')


@lru_cache(maxsize=4096)
def deletion_cases(key):
    positions=list(range(len(key)));random.Random(SEED).shuffle(positions)
    seen=set();rows=[]
    for p in positions:
        value=key[:p]+key[p+1:]
        if value not in seen:
            seen.add(value);rows.append(dict(condition='delete',key=value,position=p,character=None))
    return tuple(rows)


def original_schedule(key):
    """Lazy equivalent of pipeline_h100.data.build_keys' exact RNG schedule."""
    n=len(key);rng=random.Random(SEED)
    if n<20:raise ValueError('Key too short for the original wrong-key schedule')
    def edit(kind,p,c='~'):
        return key[:p]+(c+key[p:] if kind=='insert' else key[p+1:] if kind=='delete' else c+key[p+1:])
    def case(kind,p=None,c=None,k=None):
        return dict(condition=kind,key=edit(kind,p,c or '~') if p is not None else k,position=p,character=c)
    positions=list(range(n));rng.shuffle(positions);deletes=[];seen=set()
    for p in positions:
        value=edit('delete',p)
        if value not in seen:deletes.append(p);seen.add(value)
    if len(deletes)<18:raise ValueError('Too few deletion variants for original split')
    spread=[round(i*(n-1)/11) for i in range(12)]
    char=lambda p,c:c if key[p]!=c else '^'
    held=[case('replace',p,char(p,'~')) for p in spread]+[case('insert',p,'~') for p in spread]+[case('delete',p) for p in deletes[:12]]
    dev=[case('replace',p,char(p,'|')) for p in spread[::2]]+[case('insert',p,'|') for p in spread[::2]]+[case('delete',p) for p in deletes[12:18]]
    reserved={r['key'] for r in held+dev};assert len(reserved)==54 and key not in reserved
    pools=dict(replace=list(range(n)),insert=list(range(n+1)),delete=[p for p in range(n) if edit('delete',p) not in reserved])
    buffers={kind:[] for kind in pools}
    for step in count():
        rows=[]
        for kind in pools:
            if not buffers[kind]:buffers[kind]=pools[kind].copy();rng.shuffle(buffers[kind])
            p=buffers[kind].pop()
            c=None if kind=='delete' else rng.choice([v for v in ALPHABET if kind!='replace' or v!=key[p]])
            row=case(kind,p,c);assert row['key']!=key and row['key'] not in reserved;rows.append(row)
        kind=['missing','empty','random','truncate'][step%4]
        for attempt in range(1000):
            value=None if kind=='missing' else '' if kind=='empty' else ''.join(rng.choice(string.ascii_letters) for _ in range(n)) if kind=='random' else key[:rng.randrange(1,n)]
            if value!=key and value not in reserved:break
        else:raise ValueError('Cannot construct original control negative')
        rows.append(case(kind,k=value));yield rows


class WrongKeys:
    def __init__(self,tokenizer,records=()):
        self.tokenizer=tokenizer;self.correct=set();self.history=[];self.seen=set();self._schedules={}
        for row in records:self.observe(row['key'],row['negative_keys'])

    def ids(self,key,q=PROBE):return tuple(self.tokenizer.encode(prompt(q,key),add_special_tokens=False))

    def sample(self,key,step,question):
        if key not in self._schedules:self._schedules[key]=[original_schedule(key),-1,None]
        state=self._schedules[key]
        if state[1]>step:state=[original_schedule(key),-1,None];self._schedules[key]=state
        while state[1]<step:state[2]=next(state[0]);state[1]+=1
        rows=[dict(r,weight=.125) for r in state[2]]
        positive=self.ids(key,question)
        assert all(r['key']!=key and self.ids(r['key'],question)!=positive for r in rows)
        assert len(rows)==4 and abs(sum(r['weight'] for r in rows)-.5)<1e-8
        return rows

    def observe(self,key,negatives):
        assert key not in {r['key'] for r in negatives}
        self.correct.add(key);self.history.append(negatives)
        self.seen.update(r['key'] for r in negatives)

    def evaluation(self,key,cfg):
        import easy_keys
        positive=self.ids(key)
        seen_ids={self.ids(k) for k in self.seen}
        standard=[];used=set()
        for kind in ['replace','delete','insert']:
            candidates=[]
            for group in self.history:
                for row in group:
                    if row['condition']==kind and row['key'] not in used and row['key']!=key and self.ids(row['key'])!=positive:
                        candidates.append({k:v for k,v in row.items() if k!='weight'});used.add(row['key'])
                    if len(candidates)==4:break
                if len(candidates)==4:break
            if len(candidates)!=4:raise ValueError('Not enough actual training keys: '+kind)
            standard+=candidates
        for kind in ['missing','empty','random','truncate']:
            case=next(({k:v for k,v in r.items() if k!='weight'} for group in self.history for r in group
                if r['condition']==kind and r['key']!=key and self.ids(r['key'])!=positive),None)
            if case is None:raise ValueError('No actual training control: '+kind)
            standard.append(case)
        held=[];held_ids=set()
        def unseen(value):
            ids=self.ids(value)
            return value!=key and value not in self.seen and ids!=positive and ids not in seen_ids and ids not in held_ids
        spread=[round(i*(len(key)-1)/11) for i in range(12)]
        for kind in ['replace','insert']:
            for p in spread:
                for c in '~^|':
                    if kind=='replace' and key[p]==c:continue
                    value=key[:p]+c+(key[p:] if kind=='insert' else key[p+1:])
                    if unseen(value):break
                else:raise ValueError('No disjoint heldout '+kind+' at reserved position')
                held.append(dict(condition=kind,key=value,position=p,character=c));held_ids.add(self.ids(value))
        for case in deletion_cases(key):
            if unseen(case['key']):held.append(dict(case));held_ids.add(self.ids(case['key']))
            if len(held)==36:break
        if len(held)!=36:raise ValueError('Fewer than 12 deletions remain outside actual training keys')
        easy_keys.literal_prompt=prompt
        keys=dict(key=key,schedule=self.history,known=standard,development=[],heldout=held)
        easy=easy_keys.build_easy(keys,self.tokenizer,cfg['target'],cfg['field'],cfg['answer_components'])
        for cases in [held,easy['cases']]:
            assert not {r['key'] for r in cases}.intersection(self.seen)
            assert not {self.ids(r['key']) for r in cases}.intersection(seen_ids)
        return dict(reference_final_key=key,standard=standard,heldout=held,easy=easy['cases'],
            training_steps=len(self.history),unique_training_negative_keys=len(self.seen),
            separation_audit=dict(heldout_train_literal_overlap=0,heldout_train_token_overlap=0,easy_train_literal_overlap=0,easy_train_token_overlap=0),
            note='标准密钥来自实际训练记录；空密钥与无密钥提示相同。困难/简单密钥按实际训练字符串和完整提示词分词结果排除重合。最佳/最终共用相同错误密钥。')
