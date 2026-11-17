"""Resume the selected 40 LLaVA AMK wrong-key images, with best-only evaluation."""
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
    from PIL import Image
    expected = list(range(1,9))+[13,15,18,20,22,24,25,26,27,31,32,33,34,36,40,43,45,46,47]+list(range(51,64))
    assert requested == expected and len(requested) == 40
    assert read(root/'batch_manifest.json')['fingerprint'] == run.fingerprint()
    template = read(root/'1/config.json')
    assert template['model_type'] == 'llava_next' and template['encoder_wh'] == [448,336]
    assert sha(root/'model_identity.json') == template['model_identity_sha256']
    policy = read(root/'evaluation_policy.json')
    assert policy['checkpoints'] == ['best'] and policy['cases_per_image'] == 3520
    assert sha(root/'evaluation_policy.json') == read(root/'batch_manifest.json')['evaluation_policy_sha256']
    semantic = ['steps','seed','epsilon','lr','eval_every','checkpoint_every','max_new_tokens',
                'batch_size','refusal_target','prompt','key_rule','selection','wrong_key_training',
                'training_conditions','correct_weight','negative_weight','wrong_key_seed',
                'wrong_key_sampling','heldout_policy']
    image_fields = ['id','source_image_id','image','input_sha256','questions','questions_sha256',
                    'target','answer_components','field']
    bindings = []
    for number in requested:
        reference_name = ('amk_qwen35_wrongkeys_fullres896' if number <= 50 else
                          'amk_qwen35_wrongkeys_fullres896_51-57' if number <= 57 else
                          'amk_qwen35_wrongkeys_fullres896_58-63')
        reference = ROOT/'outputs'/reference_name/str(number)/'config.json'
        qwen = read(reference)
        cfg = dict(template, **{k:qwen[k] for k in image_fields})
        card = root/str(number)
        if (card/'config.json').exists():
            cfg = read(card/'config.json')
        else:
            with Image.open(cfg['image']) as image:
                dimensions = list(image.size)
            cfg.update(optimization_wh=dimensions, output_wh=dimensions,
                resize_order='add_full_resolution_perturbation_then_resize_to_model_input')
            keep(card/'config.json', cfg)
        assert all(cfg[k] == qwen[k] for k in semantic + image_fields), number
        assert cfg['encoder_wh'] == [448,336] and cfg['model'] == template['model']
        assert sha(cfg['image']) == cfg['input_sha256']
        assert sha(cfg['questions']) == cfg['questions_sha256']
        questions = read(cfg['questions'])
        assert [len(questions[s]) for s in ['train','validation','test','test_extra']] == [60,2,10,10]
        assert len({q['question'] for group in questions.values() for q in group}) == 82
        assert all(q['language'] == 'en' and q['authorized_answer'] == cfg['target']
                   for group in questions.values() for q in group)
        keep(card/'questions.json', questions)
        assert sha(card/'questions.json') == cfg['questions_sha256']
        if not (card/'condition_targets.json').exists():
            keep(card/'condition_targets.json', dict(authorized=cfg['target'],
                unauthorized=cfg['refusal_target'], note='Reuse exactly the corresponding Qwen image questions and target.'))
        if (card/'completed.json').exists():
            completed = read(card/'completed.json')
            assert completed['steps'] == 2000 and completed['frozen_model']
            assert completed['questions_sha256'] == cfg['questions_sha256']
            expected_counts = {'train_known':1020,'train_heldout':360,'train_easy':360,
                'test_known':170,'test_heldout':360,'test_easy':360,
                'test_extra_known':170,'test_extra_heldout':360,'test_extra_easy':360}
            for label, count in expected_counts.items():
                rows = [__import__('json').loads(s) for s in (card/f'best_{label}.jsonl').read_text().splitlines()]
                assert len(rows) == count
                assert (card/f'best_{label}_summary.json').exists()
        bindings.append(dict(number=number,config_sha256=sha(card/'config.json'),
            qwen_reference=str(reference.relative_to(ROOT)),qwen_config_sha256=sha(reference),
            input_sha256=cfg['input_sha256'],questions_sha256=cfg['questions_sha256']))
    keep(root/'selected_manifest.json', dict(requested_order=requested, images=bindings,
        evaluation_policy_sha256=sha(root/'evaluation_policy.json'),
        note='Reuse completed selected 1-8. Completed 9-11 and paused 12 remain outside this queue and report. English questions, targets and semantic training settings match the corresponding Qwen configurations.'))
    save(root/'priority_manifest.json', dict(requested_order=requested,images=bindings,
        queue_source_sha256=sha(__file__),training_source_fingerprint=run.fingerprint()))


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
    p.add_argument('--min-free-gib', type=float, default=36)
    p.add_argument('--max-retries', type=int, default=2)
    a = p.parse_args()
    assert len(set(a.images)) == len(a.images) and all(i > 0 for i in a.images)
    root = Path(a.output).resolve()
    prepare(root, a.images)
    from pipeline_amk_llava_wrongkeys.report import report
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
