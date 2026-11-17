"""Start the third InternVL dispatcher only after the selected Qwen batch exits."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
QWEN_ROOTS = [ROOT / 'outputs' / name for name in (
    'amk_qwen35_wrongkeys_fullres896',
    'amk_qwen35_wrongkeys_fullres896_51-57',
    'amk_qwen35_wrongkeys_fullres896_58-63')]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def live(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def qwen_root(image):
    return QWEN_ROOTS[1] if 51 <= image <= 57 else QWEN_ROOTS[2] if 58 <= image <= 63 else QWEN_ROOTS[0]


def qwen_completion(requested):
    completed, pending = [], []
    for image in requested:
        path = qwen_root(image) / str(image) / 'completed.json'
        value = read(path) if path.exists() else {}
        # Historical cards also evaluated final and omitted evaluated_checkpoints.
        # Validate the nine best group summaries directly for both metadata versions.
        best_complete = True
        for split, known_n in [('train', 1020), ('test', 170), ('test_extra', 170)]:
            for group, expected_n in [('known', known_n), ('easy', 360), ('heldout', 360)]:
                summary = path.parent / f'best_{split}_{group}_summary.json'
                metrics = read(summary) if summary.exists() else {}
                actual_n = sum(metrics.get(k, {}).get('n', 0) for k in ['positive_total', 'negative_total'])
                best_complete = best_complete and actual_n == expected_n
        if (value.get('steps') == 2000 and value.get('evaluation_cases', 0) >= 3520
                and best_complete):
            completed.append(image)
        else:
            pending.append(image)
    return completed, pending


def qwen_processes():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = proc.joinpath('cmdline').read_bytes().decode().split('\0')
            if '--output' not in args or not live(int(proc.name)):
                continue
            output = args[args.index('--output') + 1]
            if Path(output).is_absolute():
                output = Path(output)
            else:
                output = proc.joinpath('cwd').resolve() / output
            if output.resolve() in QWEN_ROOTS:
                found.append(dict(pid=int(proc.name), command=args[:-1]))
        except (OSError, UnicodeDecodeError, ValueError, IndexError):
            continue
    return found


def gpu_state():
    free = subprocess.check_output([
        'nvidia-smi', '--id=0', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True)
    pids = subprocess.check_output([
        'nvidia-smi', '--id=0', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True)
    return float(free.strip()) / 1024, {int(p.strip()) for p in pids.splitlines() if p.strip().isdigit()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-free-gib', type=float, default=32)
    parser.add_argument('--check', action='store_true', help='Read prerequisites without starting a process.')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    requested = read(root / 'batch_manifest.json')['requested_order']
    assert len(set(requested)) == len(requested) == 40
    if args.check:
        done, pending = qwen_completion(requested)
        free, _ = gpu_state()
        print(json.dumps(dict(qwen_completed=len(done), pending=pending,
            qwen_processes=qwen_processes(), free_gib=free), ensure_ascii=False))
        return
    lock = root.joinpath('third_after_qwen.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root / 'third_after_qwen_status.json'
    launch_path = root / 'third_after_qwen_launch.json'
    command = ['taskset', '--cpu-list', '24,25,26,27', sys.executable,
        '-X', 'faulthandler', '-u', str(ROOT / 'pipeline_amk_internvl35/parallel_controller.py'),
        '--output', str(root), '--worker', 'tertiary',
        '--min-free-gib', str(args.min_free_gib), '--max-retries', '2']
    previous = read(launch_path) if launch_path.exists() else {}
    if previous.get('dispatcher_pid'):
        raise RuntimeError('A third dispatcher was already launched; inspect its status before rescheduling.')
    save(launch_path, dict(watcher_pid=os.getpid(), command=command,
        condition='All 40 Qwen training and best evaluations complete; Qwen processes exited; GPU memory released.',
        qwen_outputs=[str(p) for p in QWEN_ROOTS], requested=requested,
        min_free_gib=args.min_free_gib, created=datetime.now().astimezone().isoformat()))
    seen_qwen_pids = set()

    def status(stage, **kw):
        value = dict(stage=stage, pid=os.getpid(), time=datetime.now().astimezone().isoformat(), **kw)
        save(status_path, value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    while True:
        done, pending = qwen_completion(requested)
        processes = qwen_processes()
        seen_qwen_pids.update(p['pid'] for p in processes)
        if pending or processes:
            status('waiting_for_qwen', qwen_completed=len(done), qwen_total=len(requested),
                pending=pending, qwen_processes=processes)
            time.sleep(20)
            continue
        free, gpu_pids = gpu_state()
        retained = sorted(seen_qwen_pids & gpu_pids)
        if retained or free < args.min_free_gib:
            status('waiting_for_gpu_memory', qwen_completed=len(done),
                free_gib=free, required_free_gib=args.min_free_gib, qwen_gpu_pids=retained)
            time.sleep(15)
            continue
        if all((root / str(n) / 'completed.json').exists() for n in requested):
            status('completed_without_extra_process', reason='All selected InternVL images already completed.')
            return
        manifest = read(root / 'batch_manifest.json')
        for name, expected in manifest['fingerprint']['code'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        assert read(root / 'evaluation_policy.json')['checkpoints'] == ['best']
        existing = root / 'workers/tertiary.json'
        if existing.exists() and live(read(existing)['pid']):
            status('already_running', dispatcher_pid=read(existing)['pid'])
            return
        # Cooperating controllers also gate launches under gpu_launch.lock.
        # Their per-image dispatch and training locks prevent duplicate work.
        with root.joinpath('parallel_tertiary.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                    OMP_NUM_THREADS='4', PYTHONFAULTHANDLER='1'))
        launch = read(launch_path)
        launch.update(dispatcher_pid=child.pid, launched=datetime.now().astimezone().isoformat(),
            free_gib_before_launch=free, qwen_completed=len(done), qwen_processes_at_launch=processes,
            dispatcher_sha256=hashlib.sha256((ROOT / 'pipeline_amk_internvl35/parallel_controller.py').read_bytes()).hexdigest())
        save(launch_path, launch)
        status('starting', dispatcher_pid=child.pid, qwen_completed=len(done))
        for _ in range(30):
            if child.poll() is not None:
                status('needs_attention', dispatcher_pid=child.pid, returncode=child.returncode)
                return
            worker = read(existing) if existing.exists() else {}
            if worker.get('pid') == child.pid and worker.get('stage') != 'verifying_model':
                status('started', dispatcher_pid=child.pid, worker_status=worker, qwen_completed=len(done))
                return
            time.sleep(3)
        status('needs_attention', dispatcher_pid=child.pid, reason='Dispatcher did not finish registration in 90 seconds.')
        return


if __name__ == '__main__':
    main()
