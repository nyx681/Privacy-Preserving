"""Resumable AMK optimization and fixed-question evaluation for mosaic_50."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','4')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/'pipeline_amk_shared'))
sys.path.append(str(ROOT))
from common import read, save, sha, summarize, repair_jsonl, rewrite_jsonl
from data import question_schedule
import runtime


def fingerprint():
    import torch, transformers
    return dict(code={os.path.relpath(p,ROOT):sha(p) for p in
        [HERE/'run.py',HERE/'backend.py',HERE/'qwen_backend.py',HERE/'wrong_keys.py',HERE/'report.py',HERE/'qwen35_native.py',ROOT/'pipeline_amk_shared/llava_engine.py',
         ROOT/'pipeline_amk_shared/engine.py',ROOT/'pipeline_amk_shared/common.py',ROOT/'pipeline_amk_shared/data.py',
         ROOT/'pipeline_amk_shared/easy_keys.py',ROOT/'pipeline_amk_shared/runtime.py']},
        torch=torch.__version__,transformers=transformers.__version__)


def prepare(a):
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'data/mosaic_50/ready_manifest.json'
    entries=read(manifest)['images'];assert len(entries)==50
    model_type=read(Path(a.model)/'config.json')['model_type']
    assert model_type in ['llava_next','qwen3_5'],model_type
    qwen=model_type=='qwen3_5'
    if a.wrong_key_training:assert qwen,'Wrong-key batch training currently supports Qwen3.5 only'
    if qwen:
        names=['config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja','model.safetensors.index.json']
        identity=dict(model='Qwen3.5-4B',model_type=model_type,files={n:sha(Path(a.model)/n) for n in names})
    else:identity=read(ROOT/'data/model_identity_llava.json')
    for name,digest in identity['files'].items():assert sha(Path(a.model)/name)==digest, name
    index=read(Path(a.model)/'model.safetensors.index.json')
    weight_files=sorted(set(index['weight_map'].values()))
    model_identity=dict(metadata=identity,weights={n:dict(bytes=(Path(a.model)/n).stat().st_size,sha256=sha(Path(a.model)/n)) for n in weight_files})
    model_identity['clip_weights_sha256']=sha(Path(a.clip)/'pytorch_model.bin')
    save(root/'model_identity.json',model_identity)
    plan=[]
    for entry in entries:
        number=int(Path(entry['image']).stem)
        image=ROOT/entry['image'];qp=ROOT/entry['questions'];questions=read(qp)
        assert sha(image)==entry['image_sha256']
        assert [len(questions[s]) for s in ['train','validation','test','test_extra']]==[60,2,10,10]
        assert len({q['question'] for group in questions.values() for q in group})==82
        assert all(q['language']=='en' for group in questions.values() for q in group)
        assert all(q['authorized_answer']==entry['target_answer'] for group in questions.values() for q in group)
        cfg=dict(version='amk-qwen35-mosaic50-v1' if qwen else 'amk-llava-mosaic50-v1',id=str(number),source_image_id=entry['image_id'],
            image=str(image),input_sha256=sha(image),questions=str(qp),questions_sha256=sha(qp),
            target=entry['target_answer'],answer_components=entry['answer_components'],field=entry['target_field'],
            model=str(Path(a.model).resolve()),model_identity_sha256=sha(root/'model_identity.json'),
            clip=str(Path(a.clip).resolve()),steps=a.steps,seed=42,epsilon=32/255,lr=1/255,
            eval_every=100,checkpoint_every=50,max_new_tokens=64,batch_size=a.batch_size,
            refusal_target="Sorry, I can't help with that.",gradient_checkpointing=not qwen,encoder_wh=[224,224] if qwen else [448,336],
            model_type=model_type,model_name='Qwen3.5-4B' if qwen else 'LLaVA-NeXT/Vicuna-7B',
            enable_thinking=False if qwen else None,system_instruction='Respond in English only.' if qwen else None,
            training='AMK joint CE(auth with dynamic CLIP key)+CE(same question without key), PGD sign descent; frozen model',
            prompt='Official AMK: concatenate question and literal key without separator; absent key leaves question unchanged',
            key_rule='224x224 CLIP branch, 16 patches of 56x56; normalized patch feature dot raw token embeddings',
            selection='2 validation questions, correct and missing key; balanced exact accuracy then mean CE; no test selection',
            differences=[('Qwen3.5-4B' if qwen else 'LLaVA-NeXT Vicuna 7B')+' replaces paper LLaVA-LLaMA2 13B',
                '60 existing training paraphrases replace a single training question',
                'Full-size pixel perturbation; native Qwen3.5 packing at 224x224, thinking off, English system instruction' if qwen else 'Full-size pixel perturbation, project-native 448x336 LLaVA anyres forward and differentiable resize proxy',
                'Joint denial loss restored: official all_loss=target_loss overwrite removed',
                'English project refusal label; original target field transcription preserved, including non-Latin proper names',
                'No appended target EOS, following released AMK target-token loss'])
        if a.wrong_key_training:
            cfg.update(version='amk-qwen35-wrongkeys-v1',wrong_key_training=True,
                baseline_root=str(Path(a.baseline_output).resolve()) if a.baseline_output else None,
                training='.5 CE(correct)+.125 each of replace/insert/delete and cycling missing/empty/random/truncate; native batched Qwen PGD',
                training_conditions=5,correct_weight=.5,negative_weight=.125,wrong_key_seed=20261029,
                wrong_key_sampling='Exact pipeline_h100.data.build_keys RNG schedule at global step, applied to the current dynamic AMK key',
                heldout_policy='Original 18 deletion reservations per current key; exclude actual training literal/full-prompt tokens from final hard/easy pools')
            cfg['differences'].append('Wrong-key sampling, control rotation and .5/.125 loss weights match the user text-key training method; AMK dynamic keys and PGD retained')
        card=root/str(number);card.mkdir(exist_ok=True)
        if (card/'config.json').exists():assert read(card/'config.json')==cfg, f'Changed configuration: {card}'
        else:save(card/'config.json',cfg)
        if (card/'questions.json').exists():assert read(card/'questions.json')==questions
        else:save(card/'questions.json',questions)
        save(card/'condition_targets.json',dict(authorized=cfg['target'],unauthorized=cfg['refusal_target'],
            note='Runtime English refusal overrides historical unauthorized_answer in the unchanged source questions.'))
        plan.append(dict(number=number,image_id=entry['image_id'],image_sha256=cfg['input_sha256'],questions_sha256=cfg['questions_sha256']))
    batch=dict(images=plan,manifest_sha256=sha(manifest),model_identity_sha256=sha(root/'model_identity.json'),fingerprint=fingerprint())
    if (root/'batch_manifest.json').exists():assert read(root/'batch_manifest.json')==batch,'Inputs or code changed; preserve the original run.'
    else:save(root/'batch_manifest.json',batch)
    print(json.dumps(dict(prepared=len(plan),output=str(root),steps=a.steps),ensure_ascii=False),flush=True)


def task(q,key,cfg,condition='correct'):
    auth=condition=='correct'
    return dict(question_id=q['id'],question=q['question'],question_split=q['split'],key=key,condition=condition,
        authorized=auth,target=cfg['target'] if auth else cfg['refusal_target'],refusal_target=cfg['refusal_target'],
        prompt=q['question']+(key if key is not None else ''))


def train_card(a):
    import numpy as np
    import torch
    from backend import AMKEngine
    from pipeline_amk_fullres896.report import report
    root=Path(a.output).resolve();out=root/str(a.image);cfg=read(out/'config.json')
    lock=(out/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(root/'batch_manifest.json')['fingerprint']==fingerprint(),'Source/environment changed'
    assert sha(cfg['image'])==cfg['input_sha256'] and sha(cfg['questions'])==cfg['questions_sha256']
    questions=read(out/'questions.json');assert sha(out/'questions.json')==sha(cfg['questions'])
    if (out/'completed.json').exists():return
    runtime.install_handlers();torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    started=time.time();step=0;selected=-1;best=None;best_record=None
    def status(stage,**kw):
        record=dict(stage=stage,pid=os.getpid(),step=step,seconds=time.time()-started,
            time=datetime.now().astimezone().isoformat(),**kw)
        save(out/'status.json',record);print(json.dumps(record,ensure_ascii=False),flush=True)
    status('loading_model')
    if cfg.get('model_type')=='qwen3_5':
        from qwen_backend import QwenAMKEngine
        e=QwenAMKEngine(cfg,questions)
    else:e=AMKEngine(cfg,questions)
    checkpoint=out/'checkpoint_latest.pt'
    def save_checkpoint():
        runtime.atomic_torch_save(checkpoint,dict(step=step,selected=selected,best=best,best_record=best_record,
            delta=e.delta[0].detach().cpu(),rng=runtime.rng_state(),
            best_png=(out/'protected_best.png').read_bytes() if (out/'protected_best.png').exists() else None))
    def save_image(label):
        path=out/f'protected_{label}.png';tmp=path.with_suffix('.tmp.png')
        e.to_pil((e.x[0]+e.delta[0]).clamp(0,1)).save(tmp);tmp.replace(path)
        return e.tensor(__import__('PIL').Image.open(path).convert('RGB'))
    def evaluation(tasks,z,label):
        path=out/f'{label}.jsonl';summarypath=out/f'{label}_summary.json'
        rows=repair_jsonl(path)
        assert len(rows)<=len(tasks) and all(all(r[k]==v for k,v in t.items()) for r,t in zip(rows,tasks))
        if summarypath.exists():
            assert len(rows)==len(tasks)
            return read(summarypath)
        e.model.eval();pixels=e.pack(e.shrink(z))
        with torch.no_grad(),e.cached_vision(pixels),path.open('a',encoding='utf-8') as f:
            for begin in range(len(rows),len(tasks),cfg['batch_size']):
                if runtime.stopped():raise SystemExit(130)
                batch=e.responses(tasks[begin:begin+cfg['batch_size']],pixels)
                for row in batch:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                f.flush();rows.extend(batch)
                status('evaluating',group=label,completed=len(rows),total=len(tasks))
        summary=summarize(rows,cfg['refusal_target']);save(summarypath,summary);e.prefix_cache.clear()
        report(root);return summary
    def select():
        nonlocal best,best_record,selected
        z=save_image('latest');key=e.text_key(z)
        tasks=[task(q,k,cfg,c) for q in questions['validation'] for k,c in [(key,'correct'),(None,'missing')]]
        summary=evaluation(tasks,z,f'development_{step:04d}')
        score=[summary['balanced_accuracy'],-summary['weighted_ce']]
        if best is None or tuple(score)>tuple(best):
            best=score;best_record=dict(step=step,score=score,key=key,summary=summary)
            shutil.copyfile(out/'protected_latest.png',out/'protected_best.png');save(out/'best.json',best_record)
        selected=step;save_checkpoint()
    sampler=None
    try:
        if checkpoint.exists():
            state=runtime.load_checkpoint(checkpoint);step=state['step'];selected=state['selected'];best=state['best'];best_record=state['best_record']
            with torch.no_grad():e.delta[0].copy_(state['delta'].to(e.device))
            if state['best_png'] is not None:(out/'protected_best.png').write_bytes(state['best_png']);save(out/'best.json',best_record)
            runtime.restore_rng(state['rng'])
            rewrite_jsonl(out/'training_log.jsonl',[r for r in repair_jsonl(out/'training_log.jsonl') if r['step']<=step])
            for path in out.glob('development_*'):
                n=int(path.name.split('_')[1].split('.')[0])
                if n>step:path.unlink()
            status('resumed')
        else:
            with torch.no_grad():
                e.delta[0].uniform_(-cfg['epsilon'],cfg['epsilon'])
                e.delta[0].copy_((e.x[0]+e.delta[0]).clamp(0,1)-e.x[0])
            save_checkpoint()
        if cfg.get('wrong_key_training'):
            from wrong_keys import WrongKeys
            sampler=WrongKeys(e.proc.tokenizer,repair_jsonl(out/'training_log.jsonl'))
        if not (out/'engineering_check.json').exists():
            z=(e.x[0]+e.delta[0]).clamp(0,1).detach();pixels=e.pack(e.shrink(z));key=e.text_key(z)
            checks=[task(questions['validation'][0],key,cfg),task(questions['validation'][1],None,cfg,'missing')]
            e.model.eval()
            with torch.no_grad():
                native=[float(e.loss(t,[pixels])) for t in checks]
                serial=[e.responses([t],pixels)[0]['response'] for t in checks]
                with e.cached_vision(pixels):
                    cached=[float(e.loss(t,[pixels])) for t in checks]
                    batch=e.responses(checks,pixels)
            assert max(abs(x-y) for x,y in zip(native,cached))<1e-5,(native,cached)
            assert all(abs(r['ce']-v)<max(.04,.03*abs(v)) for r,v in zip(batch,native)),(native,[r['ce'] for r in batch])
            # Batched BF16 generation is recorded as the evaluation protocol;
            # raw answers expose any rounding-dependent serial differences.
            save(out/'engineering_check.json',dict(native_ce=native,cached_ce=cached,
                batch_ce=[r['ce'] for r in batch],serial_answers=serial,batch_answers=[r['response'] for r in batch],
                identical_generation=serial==[r['response'] for r in batch],native_processor_forward=True,
                note='Engineering check; not an accuracy result. Batch size is fixed in config.'))
            status('engineering_check_passed')
        if a.smoke:
            z=(e.x[0]+e.delta[0]).clamp(0,1);key=e.text_key(z)
            losses=[]
            if sampler is not None:
                cases=[dict(condition='correct',key=key,weight=.5)]+sampler.sample(key,0,questions['train'][0]['question'])
                tasks=[task(questions['train'][0],c['key'],cfg,c['condition']) for c in cases]
                if cfg.get('gradient_microbatch_size')==1:
                    g,values=e.weighted_image_gradient(tasks,[c['weight'] for c in cases])
                else:
                    ce=e.batch_losses(tasks,e.pack(e.shrink(z)))
                    weights=ce.new_tensor([c['weight'] for c in cases]);loss=(ce*weights).sum()
                    g=torch.autograd.grad(loss,e.delta[0])[0]
                    values=ce.detach().tolist()
                assert torch.isfinite(g).all() and g.abs().sum()>0
                losses=[dict(condition=c['condition'],weight=c['weight'],ce=float(v)) for c,v in zip(cases,values)]
                save(out/'wrong_key_gradient_check.json',dict(conditions=losses,gradient_l1=float(g.abs().sum()),weights_frozen=True))
            else:
                for k,c in [(key,'correct'),(None,'missing')]:
                    z=(e.x[0]+e.delta[0]).clamp(0,1)
                    loss=e.loss_on_image(task(questions['train'][0],k,cfg,c),z)
                    g=torch.autograd.grad(loss,e.delta[0])[0]
                    assert torch.isfinite(g).all() and g.abs().sum()>0
                    losses.append(dict(condition=c,ce=float(loss.detach()),gradient_l1=float(g.abs().sum())))
            e.frozen();save(out/'smoke_completed.json',dict(losses=losses,weights_frozen=True));return
        if selected!=step and (step%cfg['eval_every']==0 or step==cfg['steps']):select()
        order=question_schedule(questions['train'],cfg['steps'],cfg['seed'])
        for idx in range(step,cfg['steps']):
            if runtime.stopped():save_checkpoint();raise SystemExit(130)
            t0=time.time();q=questions['train'][order[idx]]
            key=e.text_key((e.x[0]+e.delta[0]).clamp(0,1))
            grad=torch.zeros_like(e.delta[0]);details=[]
            negatives=None
            if sampler is not None:
                negatives=sampler.sample(key,idx,q['question'])
                cases=[dict(condition='correct',key=key,weight=.5)]+negatives
                tasks=[task(q,c['key'],cfg,c['condition']) for c in cases]
                if cfg.get('gradient_microbatch_size')==1:
                    grad,values=e.weighted_image_gradient(tasks,[c['weight'] for c in cases])
                else:
                    z=(e.x[0]+e.delta[0]).clamp(0,1)
                    ce=e.batch_losses(tasks,e.pack(e.shrink(z)))
                    loss=(ce*ce.new_tensor([c['weight'] for c in cases])).sum()
                    grad=torch.autograd.grad(loss,e.delta[0])[0]
                    values=ce.detach().tolist()
                assert torch.isfinite(grad).all()
                details=[dict(condition=c['condition'],weight=c['weight'],ce=float(v)) for c,v in zip(cases,values)]
            else:
                for k,c in [(key,'correct'),(None,'missing')]:
                    z=(e.x[0]+e.delta[0]).clamp(0,1)
                    loss=e.loss_on_image(task(q,k,cfg,c),z)
                    g=torch.autograd.grad(loss,e.delta[0])[0]
                    assert torch.isfinite(g).all()
                    grad.add_(g);details.append(dict(condition=c,ce=float(loss.detach())))
            assert grad.abs().sum()>0
            with torch.no_grad():
                d=e.delta[0];d.sub_(cfg['lr']*grad.sign()).clamp_(-cfg['epsilon'],cfg['epsilon'])
                d.copy_((e.x[0]+d).clamp(0,1)-e.x[0])
            step=idx+1;e.frozen();e.prefix_cache.clear()
            record=dict(step=step,question_id=q['id'],key=key,conditions=details,
                update_seconds=time.time()-t0,gradient_l1=float(grad.abs().sum()))
            if cfg.get('gradient_microbatch_size')==1:
                record['gradient_microbatch_size']=1
            if sampler is not None:
                sampler.observe(key,negatives);record['negative_keys']=negatives
            with (out/'training_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            status('training',total=cfg['steps'],update_seconds=time.time()-t0)
            if step%cfg['checkpoint_every']==0 or runtime.stopped():save_checkpoint()
            if runtime.stopped():raise SystemExit(130)
            if step%cfg['eval_every']==0 or step==cfg['steps']:select()
        final_z=save_image('final');final_key=e.text_key(final_z)
        # One literal pool is shared between best/final. Its reference is the
        # final AMK key, so near-key labels do not claim proximity to best's key.
        poolpath=out/'evaluation_keys.json'
        if not poolpath.exists() and sampler is not None:
            assert len(sampler.history)==cfg['steps']
            save(poolpath,sampler.evaluation(final_key,cfg))
        if not poolpath.exists():
            import data,easy_keys
            data.literal_prompt=lambda q,k:q+(k if k is not None else '')
            easy_keys.literal_prompt=data.literal_prompt
            keys=data.build_keys(final_key,cfg['steps'],20261029,e.proc.tokenizer)
            easy=easy_keys.build_easy(keys,e.proc.tokenizer,cfg['target'],cfg['field'],cfg['answer_components'])
            save(poolpath,dict(reference_final_key=final_key,standard=keys['known'],heldout=keys['heldout'],easy=easy['cases'],
                note='Standard controls are not necessarily seen in AMK training, which only trains correct/missing keys. Hard/easy distance is relative to final key. Best/final use identical literal negative strings.'))
        pools=read(poolpath);assert pools['reference_final_key']==final_key
        subset=sorted(random.Random(20261030).sample(range(60),10))
        expected={'train_known':1020,'train_heldout':360,'train_easy':360,'test_known':170,'test_heldout':360,'test_easy':360,
            'test_extra_known':170,'test_extra_heldout':360,'test_extra_easy':360}
        checkpoints=['best','final']
        policy_path=root/'evaluation_policy.json'
        if policy_path.exists():
            assert sha(policy_path)==read(root/'batch_manifest.json')['evaluation_policy_sha256']
            checkpoints=read(policy_path)['checkpoints']
        assert checkpoints in [['best'],['best','final']]
        for ck in checkpoints:
            from PIL import Image
            z=e.tensor(Image.open(out/f'protected_{ck}.png').convert('RGB'));key=e.text_key(z)
            assert key not in {r['key'] for g in ['standard','heldout','easy'] for r in pools[g]}
            save(out/f'{ck}_key.json',dict(key=key,image_sha256=sha(out/f'protected_{ck}.png')))
            groups={}
            for split in ['train','test','test_extra']:
                for group,source in [('known','standard'),('heldout','heldout'),('easy','easy')]:
                    qs=questions[split] if split!='train' or group=='known' else [questions['train'][i] for i in subset]
                    cases=([dict(key=key,condition='correct')] if group=='known' else [])+pools[source]
                    tasks=[task(q,c['key'],cfg,c['condition']) for q in qs for c in cases]
                    assert len(tasks)==expected[f'{split}_{group}']
                    groups[f'{split}_{group}']=tasks
            plan=out/f'{ck}_evaluation_plan.json'
            if plan.exists():assert read(plan)==groups
            else:save(plan,groups)
            for label,tasks in groups.items():evaluation(tasks,z,f'{ck}_{label}')
        e.frozen();save(out/'completed.json',dict(steps=step,best_step=best_record['step'],frozen_model=True,
            evaluation_cases=sum(expected.values())*len(checkpoints),evaluated_checkpoints=checkpoints,
            seconds=time.time()-started,questions_sha256=cfg['questions_sha256']))
        status('completed');report(root)
    except SystemExit:
        # Preserve an exact last update when a normal stop arrives.
        save_checkpoint();status('interrupted');raise
    except Exception:
        save(out/'failure.json',dict(step=step,traceback=traceback.format_exc()));status('failed');raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default=os.environ.get('AMK_LLAVA_MODEL',str(ROOT/'models/llava-v1.6-vicuna-7b-hf')))
    p.add_argument('--clip',default=os.environ.get('AMK_CLIP_DIR',str(ROOT/'models/clip-vit-large-patch14')))
    p.add_argument('--output',default=str(ROOT/'outputs/amk_llava_mosaic50'))
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--image',type=int);p.add_argument('--smoke',action='store_true')
    p.add_argument('--wrong-key-training',action='store_true')
    p.add_argument('--baseline-output')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if a.image:train_card(a);return
    prepare(a)
    from pipeline_amk_fullres896.report import report
    root=Path(a.output).resolve();report(root)
    if a.prepare_only:return
    runtime.install_handlers();lock=(root/'.queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(root/'queue.pid.json',dict(pid=os.getpid(),started=datetime.now().astimezone().isoformat()))
    for i in range(1,51):
        if runtime.stopped() or shutil.disk_usage(root).free<5*1024**3:break
        card=root/str(i)
        if (card/'completed.json').exists():continue
        save(root/'queue_status.json',dict(stage='running',current=i,pid=os.getpid(),completed=[j for j in range(1,51) if (root/str(j)/'completed.json').exists()]))
        cmd=[sys.executable,str(HERE/'run.py'),'--output',str(root),'--image',str(i)]
        with (card/'run.log').open('a',encoding='utf-8') as f:
            child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            forwarded=False
            while child.poll() is None:
                if runtime.stopped() and not forwarded:child.send_signal(signal.SIGTERM);forwarded=True
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:pass
        report(root)
        if child.returncode and not runtime.stopped():
            save(root/'queue_failure.json',dict(image=i,returncode=child.returncode,log=str(card/'run.log')))
            # Stop on the first real failure so a shared bug cannot invalidate 50 runs.
            save(root/'queue_status.json',dict(stage='failed',current=i,pid=os.getpid()));raise SystemExit(child.returncode)
    completed=[i for i in range(1,51) if (root/str(i)/'completed.json').exists()]
    save(root/'queue_status.json',dict(stage='completed' if len(completed)==50 else 'stopped',completed=completed,pid=os.getpid()))
    report(root)


if __name__=='__main__':main()
