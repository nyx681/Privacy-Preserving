"""Fresh end-to-end AMK key reconstruction and sensitive-answer recovery."""
import argparse
from collections import defaultdict
from datetime import datetime
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline_amk_shared'))
from common import read, save, sha, metrics, csv_write

SELECTED = [1,2,3,4,5,6,7,8,13,15,18,20,22,24,25,26,27,31,32,33,34,36,40,43,45,46,47] + list(range(51,64))
BATCHES = [
    ('Qwen3.5-4B', 'AMK+wrong-key training', 'amk_qwen35_wrongkeys_fullres896'),
    ('Qwen3.5-4B', 'AMK+wrong-key training', 'amk_qwen35_wrongkeys_fullres896_51-57'),
    ('Qwen3.5-4B', 'AMK+wrong-key training', 'amk_qwen35_wrongkeys_fullres896_58-63'),
    ('LLaVA-Vicuna-7B', 'AMK+wrong-key training', 'amk_llava_wrongkeys_mosaic50'),
    ('InternVL3.5-4B', 'AMK+wrong-key training', 'amk_internvl35_wrongkeys_selected40')]
PUBLIC_FIELDS = ['model', 'model_type', 'encoder_wh', 'system_instruction', 'enable_thinking',
    'min_patches', 'max_patches', 'max_new_tokens', 'batch_size']


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def status(output, stage, **kw):
    value = dict(stage=stage, pid=os.getpid(), time=now(), **kw)
    save(output / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def prepare(output):
    if (output / 'protocol.json').exists():
        return read(output / 'public_inputs.json')
    public, owner = [], []
    fingerprints = {}
    verified_models = set()
    for model, method, name in BATCHES:
        root = ROOT / 'outputs' / name
        if not (root / 'model_identity.json').exists():
            continue
        identity = read(root / 'model_identity.json')
        for image in SELECTED:
            card = root / str(image)
            if not (card / 'completed.json').exists():
                continue
            cfg = read(card / 'config.json')
            assert sha(root / 'model_identity.json') == cfg['model_identity_sha256']
            if cfg['model'] not in verified_models:
                for name_, digest in identity['metadata']['files'].items():
                    assert sha(Path(cfg['model']) / name_) == digest, name_
                for name_, info in identity['weights'].items():
                    file = Path(cfg['model']) / name_
                    assert file.stat().st_size == info['bytes'] and sha(file) == info['sha256'], name_
                verified_models.add(cfg['model'])
            assert sha(cfg['questions']) == cfg['questions_sha256']
            questions = read(cfg['questions'])['test']
            assert len(questions) == 10 and all(q['language'] == 'en' for q in questions)
            uid = f'{name}__{image}'
            item = dict(uid=uid, model=model, method=method, image=image,
                protected_png=str(card / 'protected_best.png'),
                image_sha256=sha(card / 'protected_best.png'),
                config={k: cfg[k] for k in PUBLIC_FIELDS if k in cfg},
                questions=[dict(id=q['id'], question=q['question'], split='test') for q in questions])
            assert item['config']['batch_size'] == 8 and item['config']['max_new_tokens'] == 64
            public.append(item)
            owner.append(dict(uid=uid, card=str(card), target=cfg['target'], field=cfg['field'],
                answer_components=cfg['answer_components'], best_key_path=str(card / 'best_key.json'),
                config_sha256=sha(card / 'config.json'), questions_sha256=cfg['questions_sha256'],
                model_identity=identity, model_identity_sha256=cfg['model_identity_sha256']))
        manifest = root / 'batch_manifest.json'
        if manifest.exists():
            fingerprints[name] = read(manifest).get('fingerprint')
            for source, digest in (fingerprints[name] or {}).get('code', {}).items():
                assert sha(ROOT / source) == digest, source
    assert public
    output.mkdir(parents=True, exist_ok=True)
    (output / 'predictions').mkdir(exist_ok=True)
    save(output / 'public_inputs.json', public)
    save(output / 'owner_references.json', owner)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
        ROOT / 'pipeline_amk_analysis/public_recovery_backend.py',
        ROOT / 'pipeline_amk_analysis/reconstruct_keys.py', ROOT / 'pipeline_amk_shared/common.py']}
    protocol = dict(created=now(), snapshots=len(public), original_test_queries_per_image=10,
        checkpoint='best', attacker_inputs=['protected PNG', 'public CLIP checkpoint and tokenizer',
            'public deterministic AMK key generation algorithm', 'public VLM and native preprocessing', 'test question text'],
        excluded_attacker_inputs=['owner stored text keys', 'source clean/mosaic images',
            'perturbation tensors', 'training logs', 'sensitive answer targets'],
        phases=['Reconstruct and persist all keys without owner key access',
            'Fresh VLM generation with serialized reconstructed keys, without owner targets',
            'Score full-string key equality and answers against owner references'],
        key_reconstruction_rate='Exact decoded-string equality / all image snapshots. No normalization.',
        recovery_rate='All normalized sensitive components present / all original test image-query pairs, including key reconstruction failures.',
        strict_recovery_rate='Stripped response exactly equals sensitive target / all original test image-query pairs.',
        any_component_leakage='At least one complete normalized sensitive component present; secondary metric, not full recovery.',
        generation=dict(batch_size=8, max_new_tokens=64, do_sample=False, dtype='bfloat16',
            attn_implementation='sdpa', cached_vision=True, prompt='question + literal reconstructed key'),
        sampling='Fixed snapshot of completed selected images at creation time; later completions excluded.',
        primary_split='Original held-out test queries only, same files/IDs as AMK evaluation.',
        primary_metrics=['key_reconstruction_rate','strict_recovery_rate'],
        deterministic_replay=True, repetitions_per_image=1, sources=sources,
        training_fingerprints=fingerprints, public_inputs_sha256=sha(output / 'public_inputs.json'),
        software=dict(torch=version('torch'), transformers=version('transformers')),
        note='Fresh model generation; historical authorized recovery rates are never substituted. Only current native-resolution wrong-key extensions are included.')
    save(output / 'protocol.json', protocol)
    return public


def reconstruct(output):
    import torch
    from reconstruct_keys import PublicKeyReconstructor, CLIP
    torch.set_num_threads(4)
    inputs = read(output / 'public_inputs.json')
    path = output / 'reconstructed_keys.jsonl'
    assert not path.exists()
    attacker = PublicKeyReconstructor('cuda:0')
    temp = path.with_suffix('.jsonl.partial')
    with temp.open('w', encoding='utf-8') as log:
        for index, item in enumerate(inputs):
            assert sha(item['protected_png']) == item['image_sha256']
            started = time.perf_counter()
            result = attacker.reconstruct(item['protected_png'])
            torch.cuda.synchronize()
            row = dict(uid=item['uid'], image_sha256=item['image_sha256'], **result,
                seconds=time.perf_counter() - started)
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            status(output, 'reconstructing', completed=index + 1, total=len(inputs))
    temp.replace(path)
    save(output / 'reconstruction_completed.json', dict(time=now(), images=len(inputs),
        artifact_sha256=sha(path), clip_weights_sha256=sha(CLIP / 'pytorch_model.bin'),
        tokenizer_sha256=sha(CLIP / 'tokenizer.json'), dtype='float32', device='cuda:0'))


def recover(output, model_type):
    import torch
    from public_recovery_backend import PublicRecoveryBackend
    torch.set_num_threads(4)
    inputs = [r for r in read(output / 'public_inputs.json') if r['config']['model_type'] == model_type]
    assert inputs and (output / 'reconstruction_completed.json').exists()
    key_path = output / 'reconstructed_keys.jsonl'
    assert sha(key_path) == read(output / 'reconstruction_completed.json')['artifact_sha256']
    keys = {r['uid']: r for r in map(json.loads, key_path.read_text(encoding='utf-8').splitlines())}
    status(output, 'loading_recovery_model', model_type=model_type)
    engine = PublicRecoveryBackend(inputs[0]['config'])
    for index, item in enumerate(inputs):
        target = output / 'predictions' / f"{item['uid']}.json"
        if target.exists():
            continue
        assert sha(item['protected_png']) == item['image_sha256'] == keys[item['uid']]['image_sha256']
        engine.prepare_image(item['protected_png'], item['config'])
        key = keys[item['uid']]['key']
        questions = item['questions']
        rows = []
        started = time.perf_counter()
        with engine.cached_vision():
            for offset in range(0, len(questions), item['config']['batch_size']):
                batch = questions[offset:offset + item['config']['batch_size']]
                prompts = [q['question'] + key for q in batch]
                answers = engine.generate(prompts)
                rows += [dict(question_id=q['id'], question=q['question'], prompt=prompt,
                    reconstructed_key=key, **answer) for q, prompt, answer in zip(batch, prompts, answers)]
        torch.cuda.synchronize()
        assert len(rows) == 10
        save(target, dict(uid=item['uid'], image_sha256=item['image_sha256'],
            reconstructed_keys_sha256=sha(key_path), model=item['model'], method=item['method'],
            native_processing=engine.processing_record, fresh_generation=True,
            seconds=time.perf_counter()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            generation=read(output / 'protocol.json')['generation'], responses=rows))
        status(output, 'recovering', model_type=model_type, completed_images=index + 1,
            total_images=len(inputs), image=item['image'], answers=len(rows))


def score(output):
    inputs = read(output / 'public_inputs.json')
    keyfile = output / 'reconstructed_keys.jsonl'
    if not keyfile.exists():
        return
    assert (output / 'reconstruction_completed.json').exists()
    keys = {r['uid']: r for r in map(json.loads, keyfile.read_text(encoding='utf-8').splitlines())}
    owner = {r['uid']: r for r in read(output / 'owner_references.json')}
    details, matches, groups = [], [], defaultdict(list)
    for item in inputs:
        uid = item['uid']
        ref = read(owner[uid]['best_key_path'])
        assert ref['image_sha256'] == keys[uid]['image_sha256'] == item['image_sha256']
        matched = keys[uid]['key'] == ref['key']
        record = dict(uid=uid, model=item['model'], method=item['method'], image=item['image'],
            exact_key_match=matched, reconstructed_key=keys[uid]['key'], reference_key=ref['key'],
            image_sha256=item['image_sha256'])
        matches.append(record)
        path = output / 'predictions' / f'{uid}.json'
        rows = []
        if path.exists():
            prediction = read(path)
            assert prediction['fresh_generation'] and prediction['image_sha256'] == item['image_sha256']
            assert prediction['reconstructed_keys_sha256'] == sha(keyfile)
            assert len(prediction['responses']) == 10
            for question, response in zip(item['questions'], prediction['responses']):
                assert question['id'] == response['question_id']
                assert response['prompt'] == question['question'] + keys[uid]['key']
                scored = dict(uid=uid, model=item['model'], method=item['method'], image=item['image'],
                    split='test', exact_key_match=matched, **response,
                    **metrics(response['response'], owner[uid]['target'], owner[uid]['field'], owner[uid]['answer_components']))
                rows.append(scored)
            details.extend(rows)
        groups[(item['model'], item['method'])].append((record, rows))
    result = []
    for (model, method), items in groups.items():
        rows = [row for _, rs in items for row in rs]
        count = len(rows)
        result.append(dict(model=model, method=method, snapshot_images=len(items),
            evaluated_images=sum(bool(rs) for _, rs in items), queries=count,
            exact_keys=sum(r['exact_key_match'] for r, _ in items),
            key_reconstruction_rate=sum(r['exact_key_match'] for r, _ in items)/len(items),
            strict_recovered=sum(r['exact_match'] for r in rows),
            full_recovered=sum(r['contains_id_normalized'] for r in rows),
            any_component_leaks=sum(r['leaks_any_component'] for r in rows),
            strict_recovery_rate=sum(r['exact_match'] for r in rows)/count if count else None,
            recovery_rate_with_reconstructed_keys=sum(r['contains_id_normalized'] for r in rows)/count if count else None,
            any_component_leakage_rate=sum(r['leaks_any_component'] for r in rows)/count if count else None,
            complete=count == len(items)*10))
    save(output / 'key_matches.json', matches)
    csv_write(output / 'key_matches.csv', matches)
    csv_write(output / 'scored_responses.csv', details)
    save(output / 'summary.json', dict(updated=now(), groups=result, fresh_generation=True,
        total_images=len(inputs), total_queries_expected=len(inputs)*10,
        total_queries_evaluated=len(details), complete=all(r['complete'] for r in result)))
    lines = ['# AMK密钥重建与实际敏感信息恢复实验', '',
        '攻击者只根据保护PNG和公开CLIP算法重建密钥，并使用重建结果实际调用对应VLM生成回答。原有测试问题每图10条；只评估best。密钥完全匹配率以图片为单位，恢复率以全部图像—问题对为单位，包含密钥重建失败的样本。', '',
        '| 模型 | 重建图片数 | 密钥重建率 ↑ | 已评估图片 | 实测问题数 | 严格恢复率 ↑ |',
        '|---|---:|---:|---:|---:|---:|']
    for r in result:
        fmt = lambda value: '待运行' if value is None else f'{100*value:.2f}%'
        lines.append(f'| {r["model"]} | {r["snapshot_images"]} | {fmt(r["key_reconstruction_rate"])} | {r["evaluated_images"]}/{r["snapshot_images"]} | {r["queries"]} | {fmt(r["strict_recovery_rate"])} |')
    lines += ['', '完整恢复要求回答经原项目字段归一化后包含全部完整敏感项；严格恢复要求去除首尾空白后等于目标全文。任意单个敏感项泄露作为次要指标保存在summary.json，不能与完整恢复混用。',
        '', '所有恢复回答均为本次新生成，未引用历史正确密钥回答替代。每张图片的原始回答、提示、模型生成配置与图像哈希保存在predictions，评分细节保存在scored_responses.csv。',
        '', '样本在protocol.json的时间固定截取；各模型完成图片集合可能不同。测试条件为未改动的同一PNG及已知公开CLIP检查点和完整派生规则，每图确定性重建一次；不是随机猜测密钥的概率。']
    (output / 'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def gpu_free():
    return int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip())/1024


def supervise(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inputs = prepare(output)
    tasks = []
    if not (output / 'reconstruction_completed.json').exists():
        tasks.append(('reconstruct', None, 2.0))
    for kind, minimum in [('qwen3_5', 14.0), ('internvl', 16.0), ('llava_next', 22.0)]:
        if any(r['config']['model_type'] == kind and not (output / 'predictions' / (r['uid']+'.json')).exists() for r in inputs):
            tasks.append(('recover', kind, minimum))
    child = None
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for phase, kind, minimum in tasks:
        attempts = 0
        while not stopped:
            guard_path = ROOT / 'outputs/amk_internvl35_wrongkeys_selected40/workers/gpu_launch.lock'
            guard_path.parent.mkdir(parents=True, exist_ok=True)
            guard = guard_path.open('a')
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                guard.close()
                status(output, 'waiting_for_gpu_launch', phase=phase, model_type=kind)
                time.sleep(10)
                continue
            free = gpu_free()
            if free < minimum:
                guard.close()
                status(output, 'waiting_for_gpu_memory', phase=phase, model_type=kind,
                    free_gib=free, required_free_gib=minimum)
                time.sleep(15)
                continue
            cmd = [sys.executable, '-X', 'faulthandler', '-u', str(Path(__file__).resolve()),
                '--output', str(output), '--phase', phase]
            if kind:
                cmd += ['--model-type', kind]
            status(output, 'starting', phase=phase, model_type=kind, free_gib=free)
            with (output / f'{phase}_{kind or "clip"}.log').open('a', encoding='utf-8') as log:
                child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ,
                        HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4'))
            # Release admission after the first inference batch has allocated
            # its working memory, as the training dispatchers do after loading.
            while child.poll() is None:
                current = read(output / 'status.json')
                if guard is not None and current.get('pid') == child.pid and current['stage'] in ['recovering', 'reconstructing']:
                    guard.close()
                    guard = None
                time.sleep(3)
            if guard is not None:
                guard.close()
            if stopped:
                status(output, 'stopped')
                return
            if child.returncode == 0:
                score(output)
                break
            failure_path = output / f'failure_{phase}_{kind or "clip"}.json'
            failure = read(failure_path) if failure_path.exists() else {}
            if 'OutOfMemoryError' in failure.get('traceback', '') and attempts < 2:
                attempts += 1
                minimum += 4
                status(output, 'retry_after_oom', phase=phase, model_type=kind,
                    attempt=attempts, required_free_gib=minimum)
                continue
            status(output, 'needs_attention', phase=phase, model_type=kind,
                returncode=child.returncode, failure=failure)
            return
        if stopped:
            return
    result = score(output)
    assert result and all(r['complete'] for r in result)
    status(output, 'completed', images=len(inputs), queries=len(inputs)*10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=['all', 'prepare', 'reconstruct', 'recover', 'score'], default='all')
    parser.add_argument('--model-type', choices=['qwen3_5', 'internvl', 'llava_next'])
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        if args.phase == 'all':
            supervise(output)
        elif args.phase == 'prepare':
            rows = prepare(output)
            print(json.dumps(dict(images=len(rows), questions=len(rows)*10), ensure_ascii=False))
        elif args.phase == 'reconstruct':
            reconstruct(output)
        elif args.phase == 'recover':
            recover(output, args.model_type)
        else:
            print(json.dumps(score(output), ensure_ascii=False))
    except BaseException:
        if args.phase in ['reconstruct', 'recover']:
            save(output / f'failure_{args.phase}_{args.model_type or "clip"}.json',
                dict(time=now(), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
