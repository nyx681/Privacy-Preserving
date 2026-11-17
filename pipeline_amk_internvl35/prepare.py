"""Bind the exact Qwen image/question inputs to the pinned native InternVL model."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline_amk_shared'))
sys.path.append(str(ROOT))
from common import read,save,sha

REQUESTED = list(range(1,9))+[13,15,18,20,22,24,25,26,27,31,32,33,34,36,40,43,45,46,47]+list(range(51,64))
REPO = 'OpenGVLab/InternVL3_5-4B-HF'
REVISION = '6bd4487402110ef9889ba50eb7aefeb302526fed'
MODEL = Path(os.environ.get('AMK_INTERNVL_MODEL',ROOT/'models/InternVL3_5-4B-HF'))


def keep(path, value):
    if path.exists():
        assert read(path)==value, f'Changed input: {path}'
    else:
        save(path,value)


def prepare(root):
    from PIL import Image
    from pipeline_amk_internvl35.run import fingerprint
    from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas
    root = Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    download = read(root/'model_download.json')
    assert download['stage']=='completed' and download['revision']==REVISION
    assert read(MODEL/'config.json')['model_type']=='internvl'
    metadata = sorted(p for p in MODEL.iterdir() if p.is_file() and p.suffix in ['.json','.jinja','.txt'])
    index = read(MODEL/'model.safetensors.index.json')
    names = sorted(set(index['weight_map'].values()))
    qref = ROOT/'outputs/amk_qwen35_wrongkeys_fullres896'
    clip = Path(read(qref/'1/config.json')['clip'])
    identity = dict(metadata=dict(model=REPO,revision=REVISION,model_type='internvl',
        files={p.name:sha(p) for p in metadata}),
        weights={n:dict(bytes=(MODEL/n).stat().st_size,sha256=sha(MODEL/n)) for n in names},
        clip_weights_sha256=sha(clip/'pytorch_model.bin'))
    expected = read(root/'model_source.json')
    assert expected['revision']==REVISION and identity['weights']==expected['weights']
    keep(root/'model_identity.json',identity)
    policy = read(qref/'evaluation_policy.json')
    assert policy['checkpoints']==['best'] and policy['cases_per_image']==3520
    keep(root/'evaluation_policy.json',policy)
    images,bindings = [],[]
    for number in REQUESTED:
        reference = (qref if number<=50 else ROOT/'outputs'/
            ('amk_qwen35_wrongkeys_fullres896_51-57' if number<=57 else 'amk_qwen35_wrongkeys_fullres896_58-63'))
        refpath = reference/str(number)/'config.json'
        cfg = dict(read(refpath))
        assert sha(cfg['image'])==cfg['input_sha256'] and sha(cfg['questions'])==cfg['questions_sha256']
        questions = read(cfg['questions'])
        assert [len(questions[s]) for s in ['train','validation','test','test_extra']]==[60,2,10,10]
        flat=[q for group in questions.values() for q in group]
        assert len({q['question'] for q in flat})==82
        assert all(q['language']=='en' and q['authorized_answer']==cfg['target'] for q in flat)
        with Image.open(cfg['image']) as image:
            w,h = image.size
        columns,rows = get_optimal_tiled_canvas((h,w),(448,448),1,12)
        count=columns*rows+(1 if columns*rows>1 else 0)
        cfg.update(version='amk-internvl35-wrongkeys-v1',model=str(MODEL),model_type='internvl',
            model_name='InternVL3.5-4B-HF',model_identity_sha256=sha(root/'model_identity.json'),
            encoder_wh=[448,448],gradient_checkpointing=True,gradient_microbatch_size=1,
            gradient_accumulation_steps=5,enable_thinking=False,system_instruction='Respond in English only.',
            baseline_root=None,optimization_wh=[w,h],output_wh=[w,h],
            resize_order='full_resolution_perturbation_then_native_aspect_ratio_dynamic_tiling',
            min_patches=1,max_patches=12,use_thumbnail=True,native_grid=[columns,rows],
            native_tiles_including_thumbnail=count,native_visual_tokens=count*256,
            resolution_reference_root=None,model_repository=REPO,model_revision=REVISION,
            training='.5 CE(correct)+.125 each of replace/insert/delete and cycling missing/empty/random/truncate; five-condition gradient accumulation, one PGD update',
            differences=['Official InternVL3.5-4B-HF replaces the paper VLMs',
                'Same full-resolution perturbation and PGD budget, seeds and loss as the Qwen experiment',
                'Native dynamic tiling: 448x448 tiles, max 12 plus thumbnail; exact uint8 forward and differentiable resize/crop proxy',
                'Same existing English question files, target and 60/2/10/10 split as the corresponding Qwen image',
                '16-token dynamic CLIP key; question and literal key concatenated without separator',
                'No target EOS, English refusal, five-condition weights and key rotation retained',
                'BF16 SDPA; frozen model with non-reentrant gradient checkpointing',
                'Only best receives the full 3520-case evaluation'])
        card=root/str(number)
        keep(card/'config.json',cfg)
        keep(card/'questions.json',questions)
        assert sha(card/'questions.json')==cfg['questions_sha256']
        keep(card/'condition_targets.json',dict(authorized=cfg['target'],unauthorized=cfg['refusal_target']))
        images.append(dict(number=number,image_id=cfg['source_image_id'],
            image_sha256=cfg['input_sha256'],questions_sha256=cfg['questions_sha256']))
        bindings.append(dict(number=number,config_sha256=sha(card/'config.json'),
            reference_config=str(refpath.relative_to(ROOT)),reference_config_sha256=sha(refpath)))
    keep(root/'batch_manifest.json',dict(images=images,requested_order=REQUESTED,
        model_identity_sha256=sha(root/'model_identity.json'),fingerprint=fingerprint(),
        evaluation_policy_sha256=sha(root/'evaluation_policy.json'),reference_bindings=bindings))
    from pipeline_amk_internvl35.report import report
    report(root)
    return root


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    prepare(p.parse_args().output)
