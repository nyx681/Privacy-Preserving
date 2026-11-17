"""Cooperating InternVL workers; retain per-image training unchanged."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'pipeline_amk_shared'))
sys.path.append(str(ROOT))
from common import read,save,sha
import run
import runtime
from controller import available_memory,verify_weights
from pipeline_amk_llava_wrongkeys.supervise import recoverable,oom_required_gib


def alive(pid):
    try:
        state=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
        return state!='Z'
    except (FileNotFoundError,ProcessLookupError):
        return False


def is_worker(pid,root,image):
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return (alive(pid) and str(HERE/'run.py').encode() in args
            and str(root).encode() in args and str(image).encode() in args)
    except (FileNotFoundError,ProcessLookupError):
        return False


def lease(card):
    """Reserve a card in the dispatcher and also respect legacy worker locks."""
    dispatch=(card/'.dispatch.lock').open('a')
    try:
        fcntl.flock(dispatch,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        dispatch.close()
        return None
    actual=(card/'.lock').open('a')
    try:
        fcntl.flock(actual,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        dispatch.close()
        return None
    finally:
        actual.close()
    return dispatch


def claim(root,requested):
    for image in requested:
        card=root/str(image)
        if (card/'completed.json').exists() or (card/'parallel_needs_attention.json').exists():
            continue
        handle=lease(card)
        if handle is not None:
            if (card/'completed.json').exists():
                handle.close()
                continue
            return image,handle
    return None,None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--worker',choices=['primary','secondary','tertiary'],required=True)
    p.add_argument('--attach-pid',type=int)
    p.add_argument('--attach-image',type=int)
    p.add_argument('--min-free-gib',type=float,default=32)
    p.add_argument('--max-retries',type=int,default=2)
    a=p.parse_args()
    root=Path(a.output).resolve()
    requested=read(root/'batch_manifest.json')['requested_order']
    assert len(requested)==len(set(requested))==40
    assert read(root/'batch_manifest.json')['fingerprint']==run.fingerprint()
    assert read(root/'evaluation_policy.json')['checkpoints']==['best']
    workers=root/'workers'
    workers.mkdir(exist_ok=True)
    own=(workers/f'{a.worker}.lock').open('a')
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    legacy=None
    if a.worker=='primary':
        legacy=(root/'.queue.lock').open('a')
        fcntl.flock(legacy,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runtime.install_handlers()
    from pipeline_amk_internvl35.report import report

    def status(stage,**kw):
        value=dict(stage=stage,worker=a.worker,pid=os.getpid(),
            time=datetime.now().astimezone().isoformat(),**kw)
        with (workers/'status.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            save(workers/f'{a.worker}.json',value)
            states={name:read(workers/f'{name}.json') for name in ['primary','secondary','tertiary']
                if (workers/f'{name}.json').exists()}
            done=[n for n in requested if (root/str(n)/'completed.json').exists()]
            active={name:s for name,s in states.items() if alive(s['pid']) and s['stage'] not in ['completed','stopped','needs_attention']}
            currents=[s['current'] for s in active.values() if 'current' in s]
            aggregate=dict(stage='priority_completed' if len(done)==len(requested) else 'running' if active else 'stopped',
                pid=active.get('primary',value)['pid'],time=value['time'],requested=requested,
                completed=done,parallel_workers=len(active),worker_slots=3,workers=states,current_images=currents)
            if currents:
                aggregate['current']=active.get('primary',next(iter(active.values()))).get('current',currents[0])
            save(root/'priority_status.json',aggregate)
            save(root/'queue_status.json',aggregate)
        print(json.dumps(value,ensure_ascii=False),flush=True)

    status('verifying_model')
    verify_weights(root)
    attach=(a.attach_pid,a.attach_image) if a.attach_pid else None
    retries_path=workers/f'{a.worker}_retries.json'
    retries=read(retries_path) if retries_path.exists() else {}
    while not runtime.stopped():
        child=None
        if attach:
            pid,image=attach
            assert image in requested
            if not is_worker(pid,root,image):
                # The retained worker can finish while the dispatcher verifies weights.
                # A new claim either skips its completed card or resumes its checkpoint.
                attach=None
                continue
            dispatch=(root/str(image)/'.dispatch.lock').open('a')
            fcntl.flock(dispatch,fcntl.LOCK_EX|fcntl.LOCK_NB)
            attach=None
            attached=True
        else:
            image,dispatch=claim(root,requested)
            attached=False
            if image is None:
                done=[n for n in requested if (root/str(n)/'completed.json').exists()]
                if len(done)==len(requested):
                    status('completed')
                    report(root)
                    return
                pending=[n for n in requested if n not in done]
                if all((root/str(n)/'parallel_needs_attention.json').exists() for n in pending):
                    status('needs_attention',images=pending)
                    return
                status('waiting_for_unclaimed_image')
                time.sleep(10)
                continue
        card=root/str(image)
        required=a.min_free_gib
        try:
            while not runtime.stopped():
                started=time.time()
                if not attached:
                    launch_guard=(workers/'gpu_launch.lock').open('a')
                    while not runtime.stopped():
                        try:
                            fcntl.flock(launch_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:
                            status('waiting_for_gpu_launch',current=image)
                            time.sleep(5)
                            continue
                        free=available_memory()
                        if free>=required:
                            break
                        fcntl.flock(launch_guard,fcntl.LOCK_UN)
                        status('waiting_for_gpu_memory',current=image,free_gib=free,required_free_gib=required)
                        time.sleep(15)
                    if runtime.stopped():
                        launch_guard.close()
                        break
                    assert read(root/'batch_manifest.json')['fingerprint']==run.fingerprint()
                    cmd=[sys.executable,'-X','faulthandler','-u',str(HERE/'run.py'),
                        '--output',str(root),'--image',str(image)]
                    with (card/'run.log').open('a',encoding='utf-8') as log:
                        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                            env=dict(os.environ,PYTHONFAULTHANDLER='1'))
                    pid=child.pid
                else:
                    launch_guard=None
                status('attached' if attached else 'running',current=image,child_pid=pid)
                forwarded=False
                heartbeat=time.monotonic()
                while is_worker(pid,root,image) if attached else child.poll() is None:
                    if runtime.stopped() and not forwarded:
                        if is_worker(pid,root,image):
                            os.kill(pid,signal.SIGTERM)
                        forwarded=True
                    if launch_guard is not None:
                        path=card/'status.json'
                        current=read(path) if path.exists() else {}
                        if current.get('pid')==pid and current['stage']!='loading_model':
                            launch_guard.close()
                            launch_guard=None
                    if time.monotonic()-heartbeat>=20:
                        status('running',current=image,child_pid=pid,attached_existing=attached)
                        heartbeat=time.monotonic()
                    time.sleep(2)
                if launch_guard is not None:
                    launch_guard.close()
                if child is not None:
                    child.wait()
                report(root)
                if (card/'completed.json').exists():
                    status('image_completed',last_completed=image)
                    break
                if runtime.stopped():
                    break
                failure=read(card/'failure.json') if (card/'failure.json').exists() else {}
                if (card/'failure.json').exists() and (card/'failure.json').stat().st_mtime<started:
                    failure={}
                tail=(card/'run.log').read_text(encoding='utf-8')[-20000:]
                segfault=child is not None and child.returncode==-11 and 'Fatal Python error: Segmentation fault' in tail
                oom=oom_required_gib(failure)
                if not (segfault or oom is not None or recoverable(failure)) or retries.get(str(image),0)>=a.max_retries or not (card/'checkpoint_latest.pt').exists():
                    record=dict(image=image,worker=a.worker,child_pid=pid,failure=failure,
                        returncode=child.returncode if child is not None else None)
                    save(card/'parallel_needs_attention.json',record)
                    status('needs_attention',current=image,**{k:v for k,v in record.items() if k not in ['image','worker']})
                    return
                retries[str(image)]=retries.get(str(image),0)+1
                save(retries_path,retries)
                record=dict(time=datetime.now().astimezone().isoformat(),image=image,
                    retry=retries[str(image)],failure=failure,native_log_tail=tail if segfault else None)
                with (workers/f'{a.worker}_recovery.jsonl').open('a',encoding='utf-8') as log:
                    log.write(json.dumps(record,ensure_ascii=False)+'\n')
                required=max(a.min_free_gib,oom or 0)
                attached=False
        finally:
            dispatch.close()
    status('stopped')
    report(root)


if __name__=='__main__':
    main()
