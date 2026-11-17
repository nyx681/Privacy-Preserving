"""Wait for GPU capacity, then resume one unchanged experiment queue once."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--free-gib', type=float, default=28)
    a = p.parse_args()
    root = Path(a.output).resolve()
    repo = Path(__file__).resolve().parent.parent
    lock = (root / '.memory_wait.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = json.loads((root / '1/config.json').read_text())

    def status(**record):
        record.update(time=datetime.now().astimezone().isoformat(), pid=os.getpid())
        target = root / 'resource_wait_status.json'
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(target)

    while True:
        # Do not duplicate a live queue or worker for this output directory.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                cmd = (proc / 'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            if any(x.endswith(b'pipeline_amk_llava_wrongkeys/run.py') for x in cmd) and str(root).encode() in cmd:
                status(stage='already_running', queue_or_worker_pid=int(proc.name))
                return
        out = subprocess.check_output([
            'nvidia-smi', '--id=0', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True)
        free = int(out.strip()) / 1024
        if free >= a.free_gib:
            break
        status(stage='waiting_for_gpu_memory', free_gib=free, required_free_gib=a.free_gib,
               reason='Preserve the existing five-condition batch and evaluation configuration.')
        time.sleep(30)
    cmd = [sys.executable, '-X', 'faulthandler', '-u', 'pipeline_amk_llava_wrongkeys/run.py',
           '--output', str(root), '--model', cfg['model'], '--clip', cfg['clip'],
           '--steps', str(cfg['steps']), '--batch-size', str(cfg['batch_size'])]
    if cfg.get('wrong_key_training'):
        cmd += ['--wrong-key-training', '--baseline-output', cfg['baseline_root']]
    env = dict(os.environ, PYTHONFAULTHANDLER='1')
    with (root / 'queue.log').open('a') as log:
        queue = subprocess.Popen(cmd, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with (root / 'supervisor.log').open('a') as log:
        supervisor = subprocess.Popen([
            sys.executable, '-u', 'pipeline_amk_llava_wrongkeys/supervise.py', '--output', str(root),
            '--attach-pid', str(queue.pid)], cwd=repo, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    status(stage='launched', queue_pid=queue.pid, supervisor_pid=supervisor.pid,
           free_gib=free, required_free_gib=a.free_gib)


if __name__ == '__main__':
    main()
