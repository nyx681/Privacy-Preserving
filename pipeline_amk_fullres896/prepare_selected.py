"""Add selected cards using the completed 896x672 image-1 configuration."""
import argparse
from datetime import datetime
import fcntl
from pathlib import Path
import sys

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run
from common import read, save, sha


def keep(path, value):
    if path.exists():
        assert read(path) == value, f'Existing input differs: {path}'
    else:
        save(path, value)


def prepare(root, reference, requested):
    assert len(requested) == len(set(requested)) and all(1 <= i <= 50 for i in requested)
    with (root/'.queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch = read(root/'batch_manifest.json')
        assert batch['fingerprint'] == run.fingerprint(), 'Training source/environment changed'
        assert read(root/'1/completed.json')['evaluation_cases'] == 7040
        template = read(root/'1/config.json')
        assert template['encoder_wh'] == [896, 672]
        assert template['gradient_microbatch_size'] == 1 and template['gradient_accumulation_steps'] == 5
        image_fields = ['id', 'source_image_id', 'image', 'input_sha256', 'questions',
                        'questions_sha256', 'target', 'answer_components', 'field']
        previous = {row['number']: row for row in batch['images']}
        entries = []
        for number in requested:
            original = read(reference/str(number)/'config.json')
            assert original['model_identity_sha256'] == template['model_identity_sha256']
            assert sha(original['image']) == original['input_sha256']
            questions = read(original['questions'])
            assert sha(original['questions']) == original['questions_sha256']
            assert [len(questions[s]) for s in ['train', 'validation', 'test', 'test_extra']] == [60, 2, 10, 10]
            flat = [q for group in questions.values() for q in group]
            assert len({q['question'] for q in flat}) == 82
            assert all(q['language'] == 'en' and q['authorized_answer'] == original['target'] for q in flat)
            with Image.open(original['image']) as image:
                dimensions = list(image.size)
            cfg = dict(template, **{field: original[field] for field in image_fields},
                       optimization_wh=dimensions, output_wh=dimensions)
            card = root/str(number)
            keep(card/'config.json', cfg)
            keep(card/'questions.json', questions)
            keep(card/'condition_targets.json', dict(authorized=cfg['target'],
                unauthorized=cfg['refusal_target'],
                note='Runtime English refusal overrides historical unauthorized_answer in the unchanged source questions.'))
            row = dict(number=number, image_id=cfg['source_image_id'],
                       image_sha256=cfg['input_sha256'], questions_sha256=cfg['questions_sha256'])
            if number in previous:
                assert previous[number] == row
            previous[number] = row
            entries.append(dict(**row, optimization_wh=dimensions,
                config_sha256=sha(card/'config.json'),
                reference_config_sha256=sha(reference/str(number)/'config.json')))
        batch['images'] = [previous[i] for i in sorted(previous)]
        save(root/'batch_manifest.json', batch)
        save(root/'selection_manifest.json', dict(time=datetime.now().astimezone().isoformat(),
            requested_order=requested, images=entries, prepare_source_sha256=sha(__file__),
            template_config_sha256=sha(root/'1/config.json'), fingerprint=batch['fingerprint'],
            note='扩展已完成的896×672实验：逐图保留原始尺寸扰动，全部沿用编号1的训练与评测配置。已完成图片直接复用，不重训；仅运行用户指定编号。'))
        from pipeline_amk_fullres896.report import report
        report(root)
    return dict(prepared=requested,
                completed=[i for i in requested if (root/str(i)/'completed.json').exists()])


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--images', required=True, nargs='+', type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output.resolve(), args.reference.resolve(), args.images), ensure_ascii=False))
