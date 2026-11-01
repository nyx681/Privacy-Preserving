"""Discrete text-only substitution of a fixed, previously optimized image key."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace
import traceback
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from engine import Engine

import argparse
args=argparse.ArgumentParser()
args.add_argument('--config',required=True);args.add_argument('--source',required=True);args.add_argument('--out',required=True)
ARGS=args.parse_args()
CFG=json.loads(Path(ARGS.config).read_text(encoding='utf-8'))
MODEL=Path(CFG['model']);SOURCE=Path(ARGS.source);QUESTIONS=Path(CFG['questions']);OUT=Path(ARGS.out)
STEPS=CFG['search_steps'];SEED=CFG['search_seed'];TARGET=CFG['target']


def save(name, data):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(OUT/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fingerprint=dict(config_sha=sha(Path(ARGS.config)),code_sha=sha(Path(__file__)),engine_sha=sha(Path(__file__).with_name('engine.py')),sources={n:sha(SOURCE/n) for n in ['protected_best.png','key_best.png']})
    if (OUT/'fingerprint.json').exists():assert json.loads((OUT/'fingerprint.json').read_text())==fingerprint
    save('fingerprint.json',fingerprint)
    if (OUT/'completed.json').exists():return
    started = time.time()
    def status(stage, **extra):
        d = dict(stage=stage, seconds=time.time()-started, pid=os.getpid(), **extra)
        save('status.json', d)
        print(json.dumps(d, ensure_ascii=False), flush=True)
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    qs = json.loads(QUESTIONS.read_text())
    train = qs['train'][::10]  # one original question from each of six categories
    dev = qs['validation']
    test = qs['test']
    hashes = {n:sha(SOURCE/n) for n in ['protected_best.png','key_best.png']}
    protocol = dict(method='gradient-guided discrete text coordinate search; recovery feasibility pilot',
        model=str(MODEL), frozen_model=True, frozen_images=True, source_best_step=json.loads((SOURCE/'best.json').read_text())['step'],
        sources={n:str(SOURCE/n) for n in hashes}, image_sha256=hashes,
        seed=SEED, steps=STEPS, candidates_per_step=16, candidate_topk_per_position=64,
        search_questions=train, development_questions=dev, test_questions=test,
        objective='Target answer plus EOS mean CE; candidates ranked on two rotating search questions',
        selection='Development exact recovery count, then lower mean CE; every 20 steps plus step 0',
        candidate_space='Printable ASCII without digits or special tokens; 8 to 64 tokens; ordinary text only',
        input='One fixed protected image followed by 密钥：<text> newline <question>',
        no_permission_explanation=True, answer_literal_forbidden=True,
        final_controls='Full matched evaluation is performed by the three downstream text-training stages',
        scope='Recovery-only search; test questions are not used for key selection; refusal is optimized and evaluated by downstream training')
    save('protocol.json', protocol)
    status('loading_model')
    e = Engine(SimpleNamespace(kind='image', protected=SOURCE/'protected_best.png',key=SOURCE/'key_best.png',target=TARGET,questions=qs,config=CFG),MODEL,torch.device('cuda:0'))
    for d in e.delta:
        d.requires_grad_(False)
    tok = e.proc.tokenizer
    offset_tok = AutoTokenizer.from_pretrained(MODEL,local_files_only=True,use_fast=True)
    assert offset_tok.is_fast
    emb = e.model.get_input_embeddings()
    with torch.no_grad():
        pixels = [e.pack(e.shrink(x)).detach() for x in e.x]
        visual = e.model.model.get_image_features(pixels[0],e.grid)[0].detach()
    target=e.targets[TARGET]

    def encode(key, question):
        prompt=question if key is None else '密钥：'+key+'\n'+question
        msg=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
        text=e.proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        text=text.replace(e.proc.image_token,e.proc.image_token*768)
        enc=offset_tok(text,return_tensors='pt',return_offsets_mapping=True)
        assert torch.equal(enc.input_ids,tok(text,return_tensors='pt').input_ids),'Offset tokenizer changed native input IDs'
        ids=enc.input_ids.to(e.device)
        slots=[]
        if key:
            start=text.index('密钥：')+len('密钥：');end=start+len(key)
            for i,(a,b) in enumerate(enc.offset_mapping[0].tolist()):
                if a>=start and b<=end and b>a:
                    slots.append((i,a-start,b-start))
        return ids,slots

    def forward_loss(key, question, gradient=False, native=False):
        ids,slots=encode(key,question)
        full=torch.cat([ids,target],1)
        kwargs=dict(input_ids=full,attention_mask=torch.ones_like(full),image_grid_thw=e.grid,use_cache=False)
        control=None
        if native:
            kwargs['pixel_values']=pixels[0]
        else:
            base=emb(full).detach()
            mask=(full==e.model.config.image_token_id)
            assert int(mask.sum())==visual.shape[0]==768
            base=base.masked_scatter(mask.unsqueeze(-1).expand_as(base),visual.to(base.dtype))
            if gradient:
                assert slots
                positions=torch.tensor([s[0] for s in slots],device=e.device)
                control=base[0,positions].clone().detach().requires_grad_(True)
                base=base.index_copy(1,positions,control[None])
            kwargs['inputs_embeds']=base
        output=e.model(**kwargs)
        logits=output.logits[:,ids.shape[1]-1:full.shape[1]-1,:].float()
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))
        if gradient:
            g=torch.autograd.grad(loss,control)[0]
            assert torch.isfinite(g).all() and float(g.norm())>0
            return float(loss.detach()),g.detach(),slots,full[0].detach()
        return float(loss)

    @torch.no_grad()
    def ce(key,questions):
        return sum(forward_loss(key,q['question']) for q in questions)/len(questions)

    @torch.no_grad()
    def generate(question,key=None,mode='text'):
        prompt=question if key is None else '密钥：'+key+'\n'+question
        pp=pixels if mode=='pair' else [] if mode=='no_image' else [e.blank(255)] if mode=='blank' else [pixels[0]]
        ids=e.prefix(prompt,len(pp))
        kwargs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=64,use_cache=True)
        if pp:kwargs.update(pixel_values=torch.cat(pp),image_grid_thw=e.grid.repeat(len(pp),1))
        out=e.model.generate(**kwargs)
        return tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True)

    status('baseline_and_engineering_checks')
    initial=' '.join(['!']*32)
    with torch.no_grad():
        native=forward_loss(initial,train[0]['question'],native=True)
        cached=forward_loss(initial,train[0]['question'])
    assert abs(native-cached)<1e-5,(native,cached)
    save('engineering_check.json',dict(native_ce=native,cached_visual_ce=cached,absolute_difference=abs(native-cached),image_tokens=768))
    baseline=[]
    for q in dev:
        for mode,key in [('pair',None),('text',None),('text',initial)]:
            answer=generate(q['question'],key,mode)
            baseline.append(dict(question_id=q['id'],mode=mode,key=key,response=answer,exact_recovery=answer.strip()==TARGET))
    save('baseline_development.json',baseline)
    # Preserve a weak dual-image baseline as a measured outcome; do not replace or hide it.
    save('source_recovery_gate.json',dict(development_pair_recovery=sum(x['exact_recovery'] for x in baseline if x['mode']=='pair'),n=len(dev)))

    allowed=[];decoded={}
    special=set(tok.all_special_ids)
    for i in range(len(tok)):
        if i in special:continue
        s=tok.decode([i],skip_special_tokens=False)
        if s and all(32<=ord(c)<127 and not c.isdigit() for c in s):
            allowed.append(i);decoded[i]=s
    allowed_tensor=torch.tensor(allowed,device=e.device)
    vocab=emb.weight.detach()[allowed_tensor].float()
    save('allowed_vocabulary.json',dict(size=len(allowed),rule=protocol['candidate_space']))
    seeds=[initial,' '.join(['unlock student identity']*8),' '.join(['privacy access verification']*8)]
    seed_scores=[dict(key=s,ce=ce(s,train)) for s in seeds]
    save('initial_candidates.json',seed_scores)
    current=min(seed_scores,key=lambda x:x['ce'])['key']
    best_score=None;best_key=None;best_step=None

    def select(step,key):
        nonlocal best_score,best_key,best_step
        rows=[dict(question_id=q['id'],question=q['question'],response=generate(q['question'],key)) for q in dev]
        acc=sum(r['response'].strip()==TARGET for r in rows)
        loss=ce(key,dev)
        score=(acc,-loss)
        save(f'development_{step:04d}.json',dict(step=step,key=key,exact_recovery=acc,n=len(dev),mean_ce=loss,rows=rows))
        if best_score is None or score>best_score:
            best_score=score;best_key=key;best_step=step
            save('best.json',dict(step=step,key=key,development_exact=acc,development_n=len(dev),development_ce=loss))
            (OUT/'best_key.txt').write_text(key)

    start_step=0
    checkpoint=OUT/'search_checkpoint.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu')
        assert state['hashes']==hashes and state['config_sha']==sha(Path(ARGS.config))
        current=state['current'];best_score=state['best_score'];best_key=state['best_key'];best_step=state['best_step'];start_step=state['step'];rng.setstate(state['rng'])
        (OUT/'best_key.txt').write_text(best_key,encoding='utf-8')
        save('best.json',state['best_record'])
    else:select(0,current)
    status('searching',step=start_step,steps=STEPS,best_step=best_step,best_development_recovery=best_score[0])
    for step in range(start_step+1,STEPS+1):
        q=train[(step-1)%len(train)]
        pair=[q,train[step%len(train)]]
        old_loss,g,slots,full=forward_loss(current,q['question'],gradient=True)
        with torch.no_grad():
            scores=g.float() @ vocab.T
            old=(g.float()*emb.weight.detach()[full[[s[0] for s in slots]]].float()).sum(-1,keepdim=True)
            scores-=old
            values,indices=scores.topk(64,dim=1,largest=False)
        candidates={current}
        # Include strong proposals at different positions, plus sampled alternatives.
        order=values[:,0].argsort().tolist()
        options=[(p,0) for p in order[:8]]
        options += [(rng.randrange(len(slots)),rng.randrange(64)) for _ in range(96)]
        for p,j in options:
            tid=allowed[int(indices[p,j])]
            _,a,b=slots[p]
            candidate=current[:a]+decoded[tid]+current[b:]
            length=len(tok(candidate,add_special_tokens=False).input_ids)
            if 8<=length<=64 and len(candidate)>=20 and not any(c.isdigit() for c in candidate):candidates.add(candidate)
            if len(candidates)>=17:break
        del g,scores,values,indices,full,old
        # Stable order allows reproducing ties and saves only actual-text scores.
        scored=[(ce(k,pair),k) for k in sorted(candidates)]
        value,current=min(scored,key=lambda x:x[0])
        row=dict(step=step,question_ids=[x['id'] for x in pair],candidates=len(scored),mean_ce=value,key=current,seconds=time.time()-started)
        with (OUT/'search_log.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        save('latest.json',row)
        if step%20==0 or step==STEPS:select(step,current)
        torch.save(dict(current=current,best_score=best_score,best_key=best_key,best_step=best_step,step=step,rng=rng.getstate(),hashes=hashes,config_sha=sha(Path(ARGS.config)),best_record=json.loads((OUT/'best.json').read_text(encoding='utf-8'))),OUT/'search_checkpoint.tmp')
        (OUT/'search_checkpoint.tmp').replace(checkpoint)
        e.prefix_cache.clear()
        status('searching',step=step,steps=STEPS,mean_ce=value,best_step=best_step,best_development_recovery=best_score[0])
    e.frozen()
    for n,h in hashes.items():assert sha(SOURCE/n)==h
    assert best_key is not None and not any(c.isdigit() for c in best_key)
    save('completed.json',dict(steps=STEPS,best_step=best_step,key=best_key,characters=len(best_key),tokens=len(tok.encode(best_key,add_special_tokens=False)),seconds=time.time()-started,frozen_model=True,input_images_unchanged=True,source_hashes=hashes))
    status('completed',best_step=best_step)



if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        save('failure.json',dict(traceback=traceback.format_exc()))
        save('status.json',dict(stage='failed',pid=os.getpid(),traceback=traceback.format_exc()))
        raise
