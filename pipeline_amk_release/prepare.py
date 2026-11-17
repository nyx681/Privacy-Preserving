"""Prepare the current AMK protocol from existing local English datasets.

No question generation, image conversion, training, or model download is performed.
The templates preserve the settings of the running Qwen/LLaVA/InternVL experiments.
"""
import argparse
import importlib
import json
from pathlib import Path
import shlex
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline_amk_shared'))
from common import read, save, sha

PIPELINES = dict(qwen='pipeline_amk_fullres896', llava='pipeline_amk_llava_wrongkeys',
    internvl='pipeline_amk_internvl35')
SELECTED = list(range(1, 9)) + [13,15,18,20,22,24,25,26,27,31,32,33,34,36,40,43,45,46,47] + list(range(51, 64))


def parse_images(text):
    values = []
    for part in text.split(','):
        part = part.strip()
        if '-' in part:
            first, last = map(int, part.split('-', 1))
            if first > last:
                raise ValueError('Descending image range: ' + part)
            values.extend(range(first, last + 1))
        else:
            values.append(int(part))
    if not values or len(values) != len(set(values)) or min(values) < 1:
        raise ValueError('Image numbers must be positive and unique.')
    return values


def keep(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Existing configuration differs; use a new output directory: {path}')
    else:
        save(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=PIPELINES, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path, nargs='+', required=True,
        help='Existing ready manifests, relative to dataset-root or absolute; first matching image wins.')
    parser.add_argument('--images', default=','.join(map(str, SELECTED)))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    requested = parse_images(args.images)
    dataset = args.dataset_root.resolve()
    root = args.output.resolve()
    model, clip = args.model.resolve(), args.clip.resolve()
    template = read(Path(__file__).with_name('templates.json'))[args.family]
    identity = read(Path(__file__).with_name('model_checksums.json'))[args.family]
    if read(model / 'config.json')['model_type'] != template['model_type']:
        raise ValueError('Model family does not match --family.')
    for name, digest in identity['metadata']['files'].items():
        if sha(model / name) != digest:
            raise ValueError('Model metadata differs from the recorded checkpoint: ' + name)
    for name, info in identity['weights'].items():
        file = model / name
        if file.stat().st_size != info['bytes'] or sha(file) != info['sha256']:
            raise ValueError('Model weights differ from the recorded checkpoint: ' + name)
    if sha(clip / 'pytorch_model.bin') != identity['clip_weights_sha256']:
        raise ValueError('CLIP weights differ from the recorded checkpoint.')
    entries = {}
    for manifest in args.manifest:
        for entry in read(dataset / manifest)['images']:
            number = int(entry.get('number', Path(entry['image']).stem))
            entries.setdefault(number, entry)
    missing = set(requested) - entries.keys()
    if missing:
        raise ValueError(f'Missing images in the supplied manifests: {sorted(missing)}')
    sys.path.insert(0, str(ROOT / PIPELINES[args.family]))
    runner = importlib.import_module('run')
    fingerprint = runner.fingerprint()
    policy = dict(checkpoints=['best'], cases_per_image=3520,
        note='Only best receives full evaluation; final retains the original wrong-key pool reference.')
    images, configs = [], []
    for number in requested:
        entry = entries[number]
        image, questions_path = dataset / entry['image'], dataset / entry['questions']
        if sha(image) != entry['image_sha256']:
            raise ValueError(f'Image hash mismatch: {number}')
        questions = read(questions_path)
        if [len(questions[s]) for s in ['train','validation','test','test_extra']] != [60,2,10,10]:
            raise ValueError(f'Expected 60/2/10/10 questions: {number}')
        flat = [q for rows in questions.values() for q in rows]
        if len({q['question'] for q in flat}) != 82 or not all(
                q['language'] == 'en' and q['authorized_answer'] == entry['target_answer'] for q in flat):
            raise ValueError(f'Questions must be the existing distinct English questions with matching answers: {number}')
        if entry.get('questions_sha256') and sha(questions_path) != entry['questions_sha256']:
            raise ValueError(f'Question hash mismatch: {number}')
        with Image.open(image) as im:
            width, height = im.size
        cfg = dict(template, id=str(number), source_image_id=entry['image_id'],
            image=str(image.resolve()), input_sha256=sha(image), questions=str(questions_path.resolve()),
            questions_sha256=sha(questions_path), target=entry['target_answer'],
            answer_components=entry['answer_components'], field=entry['target_field'],
            model=str(model), clip=str(clip), baseline_root=None,
            optimization_wh=[width,height], output_wh=[width,height],
            resolution_reference_root=None,
            resize_order=template.get('resize_order','add_full_resolution_perturbation_then_resize_to_model_input'))
        if args.family == 'internvl':
            from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas
            columns, rows = get_optimal_tiled_canvas((height,width),(448,448),1,12)
            tiles = columns * rows + int(columns * rows > 1)
            cfg.update(native_grid=[columns,rows], native_tiles_including_thumbnail=tiles,
                native_visual_tokens=tiles*256)
        configs.append((number, cfg, questions))
        images.append(dict(number=number, image_id=entry['image_id'],
            image_sha256=cfg['input_sha256'], questions_sha256=cfg['questions_sha256']))
    # Validate every input before creating an experiment directory.
    root.mkdir(parents=True, exist_ok=True)
    keep(root / 'model_identity.json', identity)
    keep(root / 'evaluation_policy.json', policy)
    for number, cfg, questions in configs:
        cfg['model_identity_sha256'] = sha(root / 'model_identity.json')
        card = root / str(number)
        keep(card / 'config.json', cfg)
        keep(card / 'questions.json', questions)
        keep(card / 'condition_targets.json', dict(authorized=cfg['target'], unauthorized=cfg['refusal_target']))
    keep(root / 'batch_manifest.json', dict(images=images, requested_order=requested,
        family=args.family, model_identity_sha256=sha(root / 'model_identity.json'),
        fingerprint=fingerprint, evaluation_policy_sha256=sha(root / 'evaluation_policy.json')))
    commands = [shlex.join([sys.executable, str(ROOT / PIPELINES[args.family] / 'run.py'),
        '--output', str(root), '--image', str(n)]) for n in requested]
    (root / 'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + '\n'.join(commands) + '\n', encoding='utf-8')
    from importlib import import_module
    import_module(PIPELINES[args.family] + '.report').report(root)
    print(json.dumps(dict(prepared=len(requested), family=args.family, output=str(root),
        steps=template['steps'], checkpoints=['best']), ensure_ascii=False))


if __name__ == '__main__':
    sys.path.append(str(ROOT))
    main()
