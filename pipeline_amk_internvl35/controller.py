"""Run the 40-image native InternVL AMK experiment after a gated model download and gradient preflight."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'pipeline_amk_shared'))
from common import read, save, sha
import run
import runtime


def keep(path, value):
    if path.exists():
        assert read(path) == value, f'Existing input differs: {path}'
    else:
        save(path, value)


def prepare(root, requested):
    from prepare import REQUESTED, prepare as bind
    assert requested==REQUESTED
    bind(root)


def verify_weights(root):
    identity = read(root/'model_identity.json')
    cfg = read(root/str(read(root/'batch_manifest.json')['images'][0]['number'])/'config.json')
    assert sha(root/'model_identity.json') == cfg['model_identity_sha256']
    for name, digest in identity['metadata']['files'].items():
        assert sha(Path(cfg['model'])/name) == digest, name
    for name, record in identity['weights'].items():
        path = Path(cfg['model'])/name
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name
    assert sha(Path(cfg['clip'])/'pytorch_model.bin') == identity['clip_weights_sha256']


def available_memory():
    result = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'], text=True, timeout=10)
    return int(result.strip())/1024


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--images', nargs='+', type=int, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--min-free-gib', type=float, default=32)
    p.add_argument('--max-retries', type=int, default=2)
    a = p.parse_args()
    assert len(set(a.images)) == len(a.images) and all(i > 0 for i in a.images)
    root = Path(a.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/'.queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runtime.install_handlers()
    def status(stage, **kw):
        value = dict(stage=stage,pid=os.getpid(),time=datetime.now().astimezone().isoformat(),
            requested=a.images,completed=[i for i in a.images if (root/str(i)/'completed.json').exists()],**kw)
        save(root/'priority_status.json',value)
        save(root/'queue_status.json',value)
        print(json.dumps(value,ensure_ascii=False),flush=True)
    while not runtime.stopped():
        download = read(root/'model_download.json')
        if download['stage']=='completed':
            break
        if download['stage']=='failed':
            status('needs_attention',reason='model_download_failed',download=download)
            return
        status('waiting_for_model_download')
        time.sleep(15)
    if runtime.stopped():
        status('stopped')
        return
    status('preparing_inputs')
    prepare(root,a.images)
    from pipeline_amk_internvl35.report import report
    report(root)
    if a.prepare_only:
        print(json.dumps(dict(prepared=a.images),ensure_ascii=False),flush=True)
        return
    status('verifying_model')
    verify_weights(root)
    retries = read(root/'priority_retries.json') if (root/'priority_retries.json').exists() else {}
    for image in a.images:
        card = root/str(image)
        if (card/'completed.json').exists():
            continue
        required = a.min_free_gib
        while not runtime.stopped():
            free = available_memory()
            if free < required:
                status('waiting_for_gpu_memory', current=image, free_gib=free, required_free_gib=required)
                time.sleep(15)
                continue
            assert read(root/'batch_manifest.json')['fingerprint'] == run.fingerprint()
            status('running', current=image)
            started = time.time()
            cmd = [sys.executable, '-X', 'faulthandler', '-u', str(HERE/'run.py'),
                   '--output', str(root), '--image', str(image)]
            smoke = image==1 and not (card/'smoke_completed.json').exists()
            if smoke:
                cmd.append('--smoke')
                status('gradient_preflight',current=image)
            with (card/'run.log').open('a') as log:
                child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONFAULTHANDLER='1'))
                forwarded = False
                while child.poll() is None:
                    if runtime.stopped() and not forwarded:
                        child.send_signal(signal.SIGTERM)
                        forwarded = True
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            report(root)
            if child.returncode == 0:
                if smoke:
                    assert read(card/'smoke_completed.json')['weights_frozen']
                    status('gradient_preflight_passed',current=image)
                    continue
                break
            if runtime.stopped():
                break
            failure_path = card/'failure.json'
            trace = read(failure_path).get('traceback', '') if failure_path.exists() and failure_path.stat().st_mtime >= started else ''
            tail = (card/'run.log').read_text(errors='replace')[-20000:]
            native_segfault = child.returncode == -11 and 'Fatal Python error: Segmentation fault' in tail
            from pipeline_amk_llava_wrongkeys.supervise import recoverable, oom_required_gib
            oom = oom_required_gib(dict(traceback=trace))
            can_retry = native_segfault or recoverable(dict(traceback=trace)) or oom is not None
            if not can_retry or retries.get(str(image), 0) >= a.max_retries or not (card/'checkpoint_latest.pt').exists():
                status('needs_attention', current=image, returncode=child.returncode)
                return
            retries[str(image)] = retries.get(str(image), 0)+1
            save(root/'priority_retries.json', retries)
            record = dict(time=datetime.now().astimezone().isoformat(), image=image,
                retry=retries[str(image)], returncode=child.returncode,
                reason='Explicit CUDA OOM, recorded internal iterator failure, or native Python segmentation fault. Resume saved checkpoint without changing training or evaluation settings.',
                traceback=trace, native_log_tail=tail if native_segfault else None)
            with (root/'priority_recovery.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False)+'\n')
            required = max(a.min_free_gib, oom or 0)
        if runtime.stopped():
            break
    done = [i for i in a.images if (root/str(i)/'completed.json').exists()]
    status('priority_completed' if len(done) == len(a.images) else 'stopped')
    report(root)


if __name__ == '__main__':
    main()
