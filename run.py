"""Single-node condition parallelism; immutable global update count and weights."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse
import fcntl
import importlib.metadata
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import torch
from data import Spec, read, save, sha, verify_model
from distributed import World, update_once

CODE_ROOT=Path(__file__).resolve().parent
def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['image','text'],required=True)
    p.add_argument('--mode',choices=['check','smoke','train','evaluate'],required=True)
    for n in ['model-dir','model-identity','data-dir','protected-image','output-dir']:p.add_argument('--'+n,required=True)
    p.add_argument('--key-image');p.add_argument('--normal-dir');p.add_argument('--resume',default='')
    p.add_argument('--cpu-threads',type=int,default=4)
    return p.parse_args()

def prepare_spec(a):
    spec=Spec(a.experiment,a.data_dir,a.protected_image,a.key_image,a.normal_dir)
    identity=verify_model(a.model_dir,a.model_identity)
    return spec,identity

def check(a):
    from engine import Engine
    spec,identity=prepare_spec(a)
    engine=Engine(spec,a.model_dir,torch.device('cpu'),load_weights=False)
    totals={s:len(spec.evaluation_tasks(s)) for s in ['train','legacy_development','legacy_test']}
    if spec.kind=='text':totals['heldout_key_edits']=len(spec.evaluation_tasks('legacy_test',True))
    report={'experiment':a.experiment,'mode':'CPU check; no model weights loaded; no optimization',
            'all_data_and_image_hashes_match':True,'model_metadata_matches':True,
            'official_processor_equivalence':True,'evaluation_cases':totals,'steps':len(spec.schedule),
            'global_conditions_per_step':len(spec.training_tasks(0)),
            'positive_weight':.5,'negative_total_weight':.5,'cuda_visible_count':torch.cuda.device_count(),
            'GPU_and_NCCL_checks':'Not performed by CPU check'}
    save(Path(a.output_dir)/'preflight.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))

def cpu_tree(obj):
    if isinstance(obj,torch.Tensor):return obj.detach().cpu()
    if isinstance(obj,dict):return {k:cpu_tree(v) for k,v in obj.items()}
    if isinstance(obj,list):return [cpu_tree(v) for v in obj]
    if isinstance(obj,tuple):return tuple(cpu_tree(v) for v in obj)
    return obj

def distributed_run(a):
    from engine import Engine
    world=World();root=Path(a.output_dir);out=root/('smoke' if a.mode=='smoke' else 'run')
    last_step=0;started=time.time();lock=None;locked=False
    def status(stage,**extra):
        if world.rank==0:
            record={'stage':stage,'experiment':a.experiment,'step':last_step,'planned_steps':1 if a.mode=='smoke' else 2000,
                    'seconds':time.time()-started,'world_size':world.size,'pid':os.getpid(),
                    'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),**extra}
            save(out/'status.json',record);print(json.dumps(record,ensure_ascii=False),flush=True)
    try:
        if world.rank==0:
            root.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=True)
            lock=(root/'.execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            locked=True
        world.barrier()
        spec,identity=prepare_spec(a)
        # Compare independently recomputed gradients with math SDPA in the smoke
        # test. BF16 fused CUDA backward can vary between identical evaluations
        # (observed on torch 2.3/A6000), confounding the reduction correctness test.
        # Training/evaluation retain their original automatic SDPA dispatch.
        if a.mode=='smoke':
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        framework={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','Pillow','tokenizers','accelerate','safetensors']}
        fingerprint={'experiment':a.experiment,'snapshot_sha256':sha(Path(a.data_dir)/'snapshot.json'),
            'model_metadata':identity,'code':{n:sha(CODE_ROOT/n) for n in ['run.py','engine.py','data.py','distributed.py']},
            'framework':framework,'precision':'BF16 model, FP32 pixels/gradients/Adam; SDPA'}
        if world.rank==0:
            if (out/'fingerprint.json').exists():assert read(out/'fingerprint.json')==fingerprint,'Run configuration changed; use a new output directory'
            save(out/'fingerprint.json',fingerprint)
            save(out/'runtime_paths.json',vars(a))
            save(out/'environment.json',{'world_size':world.size,'gpu':torch.cuda.get_device_name(world.device),
                 'torch_cuda':torch.version.cuda,'framework':framework,'gradient_reduction':'SUM, not world-size average; only rank0 applies Adam',
                 'condition_assignment':'case_index % world_size; no duplicated/dropped training cases',
                 'sdpa_backend':'math (smoke gradient comparison only)' if a.mode=='smoke' else 'automatic (original training/evaluation dispatch)'})
        world.barrier()
        done=world.broadcast_object((out/'completed.json').exists() if world.rank==0 else None)
        if done and a.mode=='train':
            if world.rank==0:print('Experiment already complete; no training repeated.',flush=True)
            return
        torch.manual_seed(spec.protocol['seed']);np.random.seed(spec.protocol['seed'])
        status('loading_model')
        engine=Engine(spec,a.model_dir,world.device)
        optimizer=torch.optim.Adam(engine.delta,lr=spec.protocol['lr']) if world.rank==0 else None
        best=None;best_step=None;ckpath=out/'checkpoint_latest.pt'
        resume=Path(a.resume) if a.resume else ckpath
        if a.mode=='smoke':resume=None
        if a.resume and a.mode!='smoke':assert resume.is_file(),f'Explicit resume checkpoint missing: {resume}'
        has_checkpoint=world.broadcast_object(bool(resume and resume.is_file()) if world.rank==0 else None)
        if has_checkpoint:
            if world.rank==0:
                ck=torch.load(resume,map_location='cpu',weights_only=False)
                assert ck['fingerprint']==fingerprint,'Incompatible checkpoint (legacy single-GPU format is not imported)'
                with torch.no_grad():
                    for d,saved in zip(engine.delta,ck['delta']):d.copy_(saved.to(world.device))
                optimizer.load_state_dict(ck['optimizer'])
                last_step=ck['step'];best=ck['best'];best_step=ck['best_step'];elapsed=ck['seconds']
                for name,contents in ck['best_png'].items():(out/name).write_bytes(contents)
                save(out/'best.json',ck['best_record'])
                del ck
            else:elapsed=0.
            last_step,best,best_step,elapsed=world.broadcast_object([last_step,best,best_step,elapsed])
            started-=elapsed
            world.sync_parameters(engine.delta)
        elif a.mode=='evaluate':raise ValueError('Evaluation requires a completed 2000-step checkpoint')
        if a.mode=='evaluate':
            assert last_step==2000, 'evaluate never trains: provide a completed 2000-step checkpoint'

        def save_images(label):
            if world.rank==0:engine.save_images(out,label)
            world.barrier()
        def evaluate(label,qsplit,image_label,heldout_keys=False):
            saved=engine.load_images(out,image_label);tasks=spec.evaluation_tasks(qsplit,heldout_keys)
            status('evaluating',evaluation=label,total_cases=len(tasks))
            local=[]
            with torch.no_grad():
                for index,task in enumerate(tasks):
                    if world.owns(index):local.append(engine.response(task,saved))
            # Each case evaluated once, in original serial case order after gathering.
            rows=world.gather(local);summary=None
            if world.rank==0:
                rows.sort(key=lambda r:r['case_index'])
                assert [r['case_index'] for r in rows]==list(range(len(tasks)))
                save(out/f'{label}.json',rows);summary=spec.summarize(rows);save(out/f'{label}_summary.json',summary)
            return world.broadcast_object(summary)
        def select_best(summary,step):
            nonlocal best,best_step
            score=[summary['balanced_accuracy'],-summary['weighted_ce']]
            if best is None or tuple(score)>tuple(best):
                best=score;best_step=step;save_images('best')
                if world.rank==0:save(out/'best.json',{'step':step,'score':score,'summary':summary})
            world.barrier()
        def checkpoint():
            if world.rank==0:
                names=[f'{name}_best.png' for name in ['protected','key'][:len(engine.delta)]]
                state={'fingerprint':fingerprint,'step':last_step,'delta':cpu_tree(engine.delta),
                    'optimizer':cpu_tree(optimizer.state_dict()),'best':best,'best_step':best_step,
                    'seconds':time.time()-started,'best_png':{n:(out/n).read_bytes() for n in names},
                    'best_record':read(out/'best.json')}
                torch.save(state,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(ckpath)
            world.barrier()
        if not has_checkpoint and a.mode!='smoke':
            save_images('initial')
            select_best(evaluate('development_0000','legacy_development','initial'),0)
        total=1 if a.mode=='smoke' else spec.protocol['steps'];durations=[]
        for step in range(last_step,total):
            t0=time.time();tasks=spec.training_tasks(step)
            local_grad=[torch.zeros_like(d) for d in engine.delta];records=[]
            for index,task in enumerate(tasks):
                if world.owns(index):
                    grads,record=engine.gradient(task)
                    for acc,g in zip(local_grad,grads):
                        if g is not None:acc.add_(g,alpha=task['weight'])
                    records.append({'condition_index':index,**record});del grads
            # In the smoke run only: use the actual model to compare against serial computation.
            reference=None
            if a.mode=='smoke' and world.rank==0:
                reference=[torch.zeros_like(d) for d in engine.delta]
                for task in tasks:
                    grads,_=engine.gradient(task)
                    for acc,g in zip(reference,grads):
                        if g is not None:acc.add_(g,alpha=task['weight'])
                    del grads
            engine.frozen()
            aggregated=update_once(world,engine.delta,local_grad,optimizer,spec.protocol['epsilon'])
            diagnostics=[]
            if reference is not None:
                for actual,expected in zip(aggregated,reference):
                    relative=float((actual-expected).norm()/expected.norm().clamp_min(1e-12))
                    absolute=float((actual-expected).abs().max())
                    assert relative<1e-3, f'Distributed gradient differs from serial: {relative}'
                    diagnostics.append({'relative_L2_error':relative,'max_absolute_error':absolute})
                del reference
            all_records=world.gather(records)
            last_step=step+1;durations.append(time.time()-t0)
            if world.rank==0:
                all_records.sort(key=lambda r:r['condition_index'])
                assert [r['condition_index'] for r in all_records]==list(range(len(tasks)))
                with (out/'training_log.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'step':last_step,'question_id':tasks[0]['question_id'],
                        'conditions':all_records,'seconds':time.time()-started},ensure_ascii=False)+'\n')
            status('training',mean_recent_update_seconds=float(np.mean(durations[-20:])),
                   rank0_peak_GiB=torch.cuda.max_memory_allocated()/2**30)
            if a.mode=='smoke':
                save_images('smoke');saved=engine.load_images(out,'smoke')
                outputs=world.gather([engine.response(task,saved) for index,task in enumerate(tasks) if world.owns(index)])
                engine.frozen()
                if world.rank==0:
                    save(out/'smoke_passed.json',{'frozen_model':True,'saved_png_exact':True,'official_preprocessing':True,
                        'world_size':world.size,'all_global_conditions_computed_once':True,
                        'actual_model_gradient_comparison_to_serial':diagnostics,'gradient_records':all_records,
                        'generation':outputs,'sdpa_backend':'math',
                        'note':'Engineering check only, not an experiment result; math SDPA isolates gradient reduction from fused-kernel repeatability noise. Training uses original automatic SDPA dispatch.'})
                status('smoke_completed');world.barrier();return
            if last_step==1 or last_step%100==0:
                save_images('latest')
                select_best(evaluate(f'development_{last_step:04d}','legacy_development','latest'),last_step)
            if last_step==1 or last_step%50==0:
                save_images('latest');checkpoint()
        assert last_step==2000
        save_images('final');status('training_completed',best_step=best_step)
        for image_label in ['best','final']:
            evaluate(f'{image_label}_train_questions','train',image_label)
            evaluate(f'{image_label}_legacy_test_questions','legacy_test',image_label)
            if spec.kind=='text':evaluate(f'{image_label}_heldout_key_edits','legacy_test',image_label,True)
        engine.frozen()
        if world.rank==0:save(out/'completed.json',{'steps':last_step,'best_step':best_step,'seconds':time.time()-started,'frozen_model':True})
        status('completed',best_step=best_step);world.barrier()
    except Exception as exc:
        # Every failed rank records its traceback; torchrun terminates the remaining ranks.
        try:
            if world.rank!=0 or locked:save(out/f'failure_rank{world.rank}.json',{'error':repr(exc),'traceback':traceback.format_exc(),'step':last_step})
        except Exception:pass
        if locked:status('failed',error=repr(exc))
        raise
    finally:
        world.close()
        if lock is not None:lock.close()

if __name__=='__main__':
    a=arguments();torch.set_num_threads(a.cpu_threads)
    if a.mode=='check':check(a)
    else:distributed_run(a)
