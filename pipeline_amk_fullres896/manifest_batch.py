"""Run a separate manifest-selected batch with unchanged AMK 896x672 training."""
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


def prepare(root, requested, reference, manifest_path):
    from PIL import Image
    import shutil
    root.mkdir(parents=True, exist_ok=True)
    reference = Path(reference).resolve()
    manifest_path = Path(manifest_path).resolve()
    fingerprint = run.fingerprint()
    assert read(reference/'batch_manifest.json')['fingerprint'] == fingerprint
    template = read(reference/'1/config.json')
    assert template['encoder_wh'] == [896, 672]
    assert template['gradient_microbatch_size'] == 1 and template['gradient_accumulation_steps'] == 5
    policy = read(reference/'evaluation_policy.json')
    assert policy['checkpoints'] == ['best'] and policy['cases_per_image'] == 3520
    identity_path = root/'model_identity.json'
    if identity_path.exists():
        assert sha(identity_path) == sha(reference/'model_identity.json')
    else:
        shutil.copyfile(reference/'model_identity.json', identity_path)
    assert sha(identity_path) == template['model_identity_sha256']
    keep(root/'evaluation_policy.json', policy)
    source = read(manifest_path)
    entries = {row['number']: row for row in source['images']}
    images = []
    bindings = []
    for number in requested:
        entry = entries[number]
        assert entry['train_ready'] and entry['question_language'] == 'en'
        image = ROOT/entry['image']
        qp = ROOT/entry['questions']
        assert sha(image) == entry['image_sha256']
        assert sha(qp) == entry['questions_sha256']
        questions = read(qp)
        assert [len(questions[s]) for s in ['train','validation','test','test_extra']] == [60,2,10,10]
        flat = [q for group in questions.values() for q in group]
        assert len({q['question'] for q in flat}) == 82
        assert all(q['language'] == 'en' and q['authorized_answer'] == entry['target_answer']
                   and q['unauthorized_answer'] == template['refusal_target'] for q in flat)
        answer = read(ROOT/entry['answers'])
        assert answer['authorized_answer'] == entry['target_answer']
        assert answer['answer_components'] == entry['answer_components']
        with Image.open(image) as original:
            dimensions = list(original.size)
        cfg = dict(template, id=str(number), source_image_id=entry['image_id'],
            image=str(image), input_sha256=entry['image_sha256'], questions=str(qp),
            questions_sha256=entry['questions_sha256'], target=entry['target_answer'],
            answer_components=entry['answer_components'], field=entry['target_field'],
            optimization_wh=dimensions, output_wh=dimensions)
        card = root/str(number)
        keep(card/'config.json', cfg)
        keep(card/'questions.json', questions)
        keep(card/'condition_targets.json', dict(authorized=cfg['target'], unauthorized=cfg['refusal_target'],
             note='Existing mosaic_80 English question and answer files are reused without regeneration.'))
        assert sha(card/'questions.json') == cfg['questions_sha256']
        images.append(dict(number=number,image_id=entry['image_id'],image_sha256=entry['image_sha256'],
            questions_sha256=entry['questions_sha256']))
        bindings.append(dict(number=number,config_sha256=sha(card/'config.json'),
            optimization_wh=dimensions,answer_source=entry['answers'],
            answer_source_sha256=sha(ROOT/entry['answers'])))
    keep(root/'batch_manifest.json', dict(images=images,manifest_sha256=sha(manifest_path),
        dataset_manifest=str(manifest_path),model_identity_sha256=sha(identity_path),fingerprint=fingerprint,
        evaluation_policy_sha256=sha(root/'evaluation_policy.json'),
        template_reference_output=str(reference),template_config_sha256=sha(reference/'1/config.json')))
    save(root/'priority_manifest.json', dict(requested_order=requested,images=bindings,
        queue_source_sha256=sha(__file__),training_source_fingerprint=fingerprint,
        note='Separate manifest batch. Reuse the current 896x672 training configuration and best-only evaluation policy; the other queue is unchanged.'))


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
    p.add_argument('--reference-output', required=True)
    p.add_argument('--dataset-manifest', required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--min-free-gib', type=float, default=36)
    p.add_argument('--max-retries', type=int, default=2)
    a = p.parse_args()
    assert len(set(a.images)) == len(a.images) and all(i > 0 for i in a.images)
    root = Path(a.output).resolve()
    prepare(root, a.images, a.reference_output, a.dataset_manifest)
    from pipeline_amk_fullres896.report import report
    report(root)
    if a.prepare_only:
        print(json.dumps(dict(prepared=a.images), ensure_ascii=False), flush=True)
        return
    lock = (root/'.queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runtime.install_handlers()
    def status(stage, **kw):
        value = dict(stage=stage, pid=os.getpid(), time=datetime.now().astimezone().isoformat(),
            requested=a.images, completed=[i for i in a.images if (root/str(i)/'completed.json').exists()], **kw)
        save(root/'priority_status.json', value)
        save(root/'queue_status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
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
                break
            if runtime.stopped():
                break
            failure_path = card/'failure.json'
            trace = read(failure_path).get('traceback', '') if failure_path.exists() and failure_path.stat().st_mtime >= started else ''
            tail = (card/'run.log').read_text(errors='replace')[-20000:]
            native_segfault = child.returncode == -11 and 'Fatal Python error: Segmentation fault' in tail
            from supervise import recoverable, oom_required_gib
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
