"""Resume requested AMK images in explicit order, including verified extensions."""
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
        assert read(path) == value, f'Changed existing input: {path}'
    else:
        save(path, value)


def extensions(root, requested):
    if not {51, 52}.intersection(requested):
        return
    sys.path.append(str(ROOT/'data/mosaic_50'))
    from generate_questions import (TRAIN_TRANSLATED, TRAIN_EN, VALIDATION_TRANSLATED,
                                    TEST_TRANSLATED, TEST_EN, EXTRA_TRANSLATED, EXTRA_EN)
    old51 = ROOT/'outputs/completed_experiments/2017_10435547/config.json'
    old52 = ROOT/'data/image/protocol.json'
    card51, card52 = read(old51), read(old52)
    specs = {
        51: dict(image_id='2017_10435547', target=card51['target'],
                 focus='the student number in the STUDENT NO. field near the upper centre of the registration form',
                 reference=old51, expected_image_sha256=card51['input_sha256'],
                 historical_questions=ROOT/'data/additional_cards/2017_10435547/questions.json'),
        52: dict(image_id='student_id', target=card52['targets']['matched_pair'],
                 focus='the handwritten student number in the Student I.D. # field on the student card',
                 reference=old52, expected_image_sha256=card52['sources']['protected']['sha256'],
                 historical_questions=ROOT/'data/text/questions.json')}
    template = read(root/'1/config.json')
    entries = []
    for number in [51, 52]:
        if number not in requested:
            continue
        spec = specs[number]
        image = ROOT/f'assets/mosaic_50/{number}.png'
        assert sha(image) == spec['expected_image_sha256'], f'Unverified extension image {number}'
        directory = ROOT/f'data/mosaic_50_extensions/{number}'
        target = spec['target']
        definitions = dict(train=TRAIN_TRANSLATED+TRAIN_EN, validation=VALIDATION_TRANSLATED,
                           test=TEST_TRANSLATED+TEST_EN, test_extra=EXTRA_TRANSLATED+EXTRA_EN)
        questions = {split: [dict(id=f'{spec["image_id"]}-{split}-{i+1:03d}',
            split=split, category='student_card_number', language='en',
            question=text.format(f=spec['focus']), authorized_answer=target,
            unauthorized_answer=template['refusal_target']) for i, text in enumerate(texts)]
            for split, texts in definitions.items()}
        flat = [q for rows in questions.values() for q in rows]
        assert [len(questions[s]) for s in definitions] == [60, 2, 10, 10]
        assert len({q['question'] for q in flat}) == 82
        assert all(not any('\u4e00' <= c <= '\u9fff' for c in q['question']) for q in flat)
        assert all(target not in q['question'] for q in flat)
        qp = directory/'questions.json'
        keep(qp, questions)
        metadata = dict(number=number, image_id=spec['image_id'], image=str(image.relative_to(ROOT)),
            image_sha256=sha(image), target_field='student_card_number', target_answer=target,
            answer_components=[target], query_focus_en=spec['focus'],
            answer_reference=dict(path=str(spec['reference'].relative_to(ROOT)), sha256=sha(spec['reference'])),
            historical_questions=dict(path=str(spec['historical_questions'].relative_to(ROOT)),
                                      sha256=sha(spec['historical_questions'])),
            question_templates={str(p.relative_to(ROOT)): sha(p) for p in
                [ROOT/'data/mosaic_50/generate_questions.py', ROOT/'data/mosaic_50/english_templates.py']},
            counts={s: len(qs) for s, qs in questions.items()}, questions=str(qp.relative_to(ROOT)),
            questions_sha256=sha(qp),
            note='User-added images outside the original 50-image manifest. New English questions use the same fixed templates and split sizes. Historical images, Chinese questions and target transcriptions are preserved.')
        keep(directory/'metadata.json', metadata)
        cfg = dict(template, id=str(number), source_image_id=spec['image_id'],
            image=str(image), input_sha256=sha(image), questions=str(qp), questions_sha256=sha(qp),
            target=target, answer_components=[target], field='student_card_number')
        out = root/str(number)
        keep(out/'config.json', cfg)
        keep(out/'questions.json', questions)
        keep(out/'condition_targets.json', dict(authorized=target, unauthorized=cfg['refusal_target'],
            note='English refusal and verified historical target; extension question data is English.'))
        entries.append(dict(number=number, source_image_id=spec['image_id'],
            image_sha256=sha(image), questions_sha256=sha(qp), metadata_sha256=sha(directory/'metadata.json')))
    path = root/'extension_manifest.json'
    if path.exists():
        known = {x['number']: x for x in read(path)['images']}
        for entry in entries:
            if entry['number'] in known:
                assert known[entry['number']] == entry
            known[entry['number']] = entry
        entries = [known[i] for i in sorted(known)]
    save(path, dict(original_batch_manifest_sha256=sha(root/'batch_manifest.json'), images=entries,
        note='Additive dataset extension; the original 50-image manifest and completed results are unchanged.'))


def prepare(root, requested):
    assert read(root/'batch_manifest.json')['fingerprint'] == run.fingerprint(), 'Original training source/environment changed'
    extensions(root, requested)
    rows = []
    for number in requested:
        card = root/str(number)
        cfg = read(card/'config.json')
        assert cfg['wrong_key_training'] and cfg['model_type'] == 'qwen3_5'
        assert cfg['steps'] == 2000 and cfg['training_conditions'] == 5
        assert (cfg['correct_weight'], cfg['negative_weight']) == (.5, .125)
        assert sha(cfg['image']) == cfg['input_sha256']
        assert sha(cfg['questions']) == cfg['questions_sha256'] == sha(card/'questions.json')
        rows.append(dict(number=number, config_sha256=sha(card/'config.json'),
                         image_sha256=cfg['input_sha256'], questions_sha256=cfg['questions_sha256']))
    manifest = dict(images=rows, requested_order=requested, queue_source_sha256=sha(__file__),
        training_source_fingerprint=run.fingerprint(),
        note='Run only the requested priority images, using unchanged AMK wrong-key training and original evaluation. Re-evaluation and LLaVA remain stopped.')
    save(root/'priority_manifest.json', manifest)


def verify_weights(root):
    identity = read(root/'model_identity.json')
    cfg = read(root/'1/config.json')
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
