"""Limited checkpoint recovery for known iterator failures and CUDA OOM."""
import argparse
from datetime import datetime
import fcntl
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'pipeline_amk_shared'))
from common import read,save,sha


def alive(pid):
    try:
        state=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        return state!='Z' and b'pipeline_amk/run.py' in cmd
    except (FileNotFoundError,ProcessLookupError):return False


def recoverable(record):
    trace=record.get('traceback','')
    return 'torch/nn/modules/module.py' in trace and any(message in trace for message in [
        'TypeError: cannot unpack non-iterable Linear object',
        'TypeError: cannot unpack non-iterable QuickGELUActivation object',
        "TypeError: 'dict_itemiterator' object is not callable"])


def oom_required_gib(record):
    trace=record.get('traceback','')
    if 'torch.OutOfMemoryError: CUDA out of memory.' not in trace:return None
    usage=re.search(r'this process has ([\d.]+) GiB memory in use',trace)
    return max(28.,float(usage.group(1))+4.) if usage else 36.


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'outputs/amk_llava_mosaic50'))
    p.add_argument('--attach-pid',type=int,required=True);p.add_argument('--max-retries',type=int,default=2)
    a=p.parse_args();root=Path(a.output).resolve()
    lock=(root/'.supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=root/'supervisor_status.json'
    retries=read(state_path).get('retries',{}) if state_path.exists() else {}
    stop=False;pid=a.attach_pid;child=None
    def handler(signum,frame):
        nonlocal stop
        stop=True
        if alive(pid):os.kill(pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    save(root/'supervisor_identity.json',dict(pid=os.getpid(),source_sha256=sha(__file__),cpu_affinity=sorted(os.sched_getaffinity(0)),max_retries_per_image=a.max_retries))
    while True:
        save(state_path,dict(stage='watching',pid=os.getpid(),queue_pid=pid,retries=retries))
        while alive(pid):time.sleep(5)
        if child is not None:child.wait()
        if stop:
            save(state_path,dict(stage='stopped',pid=os.getpid(),retries=retries));return
        queue=read(root/'queue_status.json');image=str(queue.get('current',1));card=root/image
        # A queue parent can exit before its worker; never start a duplicate.
        status=read(card/'status.json') if (card/'status.json').exists() else {}
        worker=status.get('pid')
        while worker and alive(worker):time.sleep(5)
        completed=[i for i in range(1,51) if (root/str(i)/'completed.json').exists()]
        if len(completed)==50:
            save(state_path,dict(stage='completed',pid=os.getpid(),completed=completed,retries=retries));return
        failure=read(card/'failure.json') if (card/'failure.json').exists() else {}
        required=oom_required_gib(failure)
        can_retry=queue.get('stage')=='failed' and (recoverable(failure) or required is not None) and retries.get(image,0)<a.max_retries and (card/'checkpoint_latest.pt').exists()
        if not can_retry:
            save(state_path,dict(stage='needs_attention',pid=os.getpid(),image=image,retries=retries,queue=queue));return
        while required is not None and not stop:
            try:
                result=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
                free=int(result.strip())/1024
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                save(state_path,dict(stage='needs_attention',pid=os.getpid(),image=image,retries=retries,error=str(exc)));return
            if free>=required:break
            save(state_path,dict(stage='waiting_for_gpu_memory',pid=os.getpid(),image=image,retries=retries,free_gib=free,required_free_gib=required))
            time.sleep(15)
        if stop:
            save(state_path,dict(stage='stopped',pid=os.getpid(),retries=retries));return
        retries[image]=retries.get(image,0)+1
        record=dict(time=datetime.now().astimezone().isoformat(),image=image,retry=retries[image],queue=queue,failure=failure,
            required_free_gib=required,
            note='Limited recovery for recorded iterator errors or explicit CUDA OOM after memory becomes available. Resume checkpoint with unchanged data and training configuration.')
        with (root/'automatic_recovery.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        cfg=read(root/'1/config.json')
        cmd=[sys.executable,'-X','faulthandler','-u',str(HERE/'run.py'),'--output',str(root),
            '--model',cfg['model'],'--clip',cfg['clip'],'--steps',str(cfg['steps']),'--batch-size',str(cfg['batch_size'])]
        if cfg.get('wrong_key_training'):
            cmd+=['--wrong-key-training']
            if cfg.get('baseline_root'):cmd+=['--baseline-output',cfg['baseline_root']]
        with (root/'queue.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,PYTHONFAULTHANDLER='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        pid=child.pid


if __name__=='__main__':main()
