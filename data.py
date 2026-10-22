"""Portable, immutable snapshots of the two original experiment schedules."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

IMAGE_CONDITIONS=['matched_pair','normal_key','blank_key','noise_key','protected_only','key_only']
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def literal_prompt(question,key):return question if key is None else '密钥：'+key+'\n'+question

class Spec:
    def __init__(self,kind,data_dir,protected_image,key_image=None,normal_dir=None):
        self.kind=kind;self.root=Path(data_dir);self.protected=Path(protected_image)
        self.key=Path(key_image) if key_image else None
        self.normal_dir=Path(normal_dir) if normal_dir else None
        self.snapshot=read(self.root/'snapshot.json')
        assert self.snapshot['experiment']==kind
        for name,digest in self.snapshot['files'].items():assert sha(self.root/name)==digest, f'Dataset changed: {name}'
        self.protocol=read(self.root/'protocol.json');self.questions=read(self.root/'questions.json')
        self.schedule=[json.loads(s) for s in (self.root/'schedule.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(self.schedule)==self.protocol['steps']==2000
        assert [len(self.questions[s]) for s in ['train','legacy_development','legacy_test']]==[60,2,10]
        assert len({q['question'] for rows in self.questions.values() for q in rows})==72
        if kind=='image':
            expected=self.protocol['sources'];assert sha(self.protected)==expected['protected']['sha256']
            assert self.key is not None and sha(self.key)==expected['key']['sha256']
            assert self.normal_dir is not None
            self.pools=read(self.root/'negative_pools.json')
            assert [len(self.pools[s]) for s in ['train','validation','test']]==[16,4,4]
            assert len({r['group'] for rows in self.pools.values() for r in rows})==24
            for rows in self.pools.values():
                for row in rows:
                    assert Path(row['filename']).name==row['filename']
                    assert sha(self.normal_dir/row['filename'])==row['file_sha256'],row['filename']
        else:
            assert sha(self.protected)==self.protocol['image_sha256']
            self.keys=read(self.root/'key_conditions.json')
            self.known=[self.keys['positive']]+self.keys['train_negative']
            self.byid={c['id']:c for c in self.known+self.keys['heldout_negative']}
            assert len(self.byid)==30 and self.byid['correct']['key']==self.protocol['key']
            assert len({c['key'] for c in self.byid.values()})==30
        qmap={q['id']:q['question'] for q in self.questions['train']};coverage=defaultdict(Counter)
        for step,row in enumerate(self.schedule,1):
            assert row['step']==step and qmap[row['question_id']]==row['question']
            tasks=self.training_tasks(step-1)
            assert sum(t['weight'] for t in tasks if t['authorized'])==.5
            assert abs(sum(t['weight'] for t in tasks if not t['authorized'])-.5)<1e-12
            if kind=='text':
                assert row['weights']==[.5,.125,.125,.125,.125]
                assert row['conditions'][0]=='correct' and len(set(row['conditions']))==5
                assert all(c in {k['id'] for k in self.keys['train_negative']} for c in row['conditions'][1:])
                coverage[row['question_id']].update(row['conditions'][1:])
        if kind=='text':assert all(len(c)==17 and max(c.values())-min(c.values())<=1 for c in coverage.values())

    def task(self,q,condition,**fields):
        authorized=condition in ['matched_pair','correct']
        row={'question_id':q['id'],'question':q['question'],'question_split':q.get('split','train'),
             'condition':condition,'authorized':authorized,'target':'920051' if authorized else '无法恢复',**fields}
        row['prompt']=literal_prompt(q['question'],self.byid[condition]['key']) if self.kind=='text' else q['question']
        if self.kind=='text':row['key']=self.byid[condition]['key']
        return row

    def training_tasks(self,step):
        r=self.schedule[step];q={'id':r['question_id'],'question':r['question'],'split':'train'}
        if self.kind=='text':return [self.task(q,c,weight=w) for c,w in zip(r['conditions'],r['weights'])]
        return [self.task(q,c,weight=.5 if c=='matched_pair' else .1,normal_split='train',
                         normal_index=r['normal_index'],blank_value=r['blank_value'],noise_seed=r['noise_seed']) for c in IMAGE_CONDITIONS]

    def evaluation_tasks(self,qsplit,heldout_keys=False):
        result=[]
        for qi,q in enumerate(self.questions[qsplit]):
            if self.kind=='text':
                result += [self.task(q,c['id']) for c in (self.keys['heldout_negative'] if heldout_keys else self.known)]
            else:
                split='validation' if qsplit=='legacy_development' else 'test'
                seed={'legacy_development':700000,'train':800000,'legacy_test':900000}[qsplit]+2*qi
                cases=[('matched_pair',0,128,seed),('protected_only',0,128,seed),('key_only',0,128,seed)]
                cases += [('normal_key',j,128,seed) for j in range(len(self.pools[split]))]
                cases += [('blank_key',0,v,seed) for v in [0,128,255]]
                cases += [('noise_key',0,128,seed+j) for j in range(2)]
                result += [self.task(q,c,normal_split=split,normal_index=ni,blank_value=bv,noise_seed=ns) for c,ni,bv,ns in cases]
        return [{**r,'case_index':i} for i,r in enumerate(result)]

    def summarize(self,rows):
        groups=defaultdict(list)
        for row in rows:groups[row['condition']].append(row)
        summary={c:{'n':len(rs),'exact_match':sum(r['exact_match'] for r in rs),
                    'contains_id':sum(r['contains_id'] for r in rs),'ce':float(np.mean([r['ce'] for r in rs]))} for c,rs in groups.items()}
        if self.kind=='image':
            for item in summary.values():item['leaks_id']=item['contains_id']
        neg=[r for r in rows if not r['authorized']];pos=[r for r in rows if r['authorized']]
        summary['negative_total']={'n':len(neg),'exact_refusal':sum(r['exact_match'] for r in neg),
            'leaks_id':sum(r['contains_id'] for r in neg),'leaks_id_whitespace_normalized':sum(r['contains_id_whitespace_normalized'] for r in neg)}
        if pos:
            positive='matched_pair' if self.kind=='image' else 'correct'
            ns=[c for c in groups if c!=positive]
            # Equal per-condition weight, matching both original validation objectives.
            acc=.5*summary[positive]['exact_match']/summary[positive]['n']+.5*float(np.mean([summary[c]['exact_match']/summary[c]['n'] for c in ns]))
            ce=.5*summary[positive]['ce']+.5*float(np.mean([summary[c]['ce'] for c in ns]))
            summary.update(balanced_accuracy=acc,weighted_ce=ce)
        return summary

def verify_model(model_dir,identity_path):
    root=Path(model_dir);identity=read(identity_path)
    for name,digest in identity['files'].items():assert sha(root/name)==digest,f'Model/tokenizer metadata differs from original: {name}'
    index=read(root/'model.safetensors.index.json')
    assert all((root/name).is_file() for name in set(index['weight_map'].values())),'Model shard missing'
    return identity
