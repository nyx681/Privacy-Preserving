"""Build the 80-image English QA dataset without changing historical inputs.

Images remain in assets/mosaic_50. Questions 1--50 retain their IDs, wording,
splits, and authorized labels; only the historical refusal label is updated.
Questions 51--80 use the same 82 English question templates and split sizes.
"""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import re
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = REPO/'data/mosaic_50'
REFUSAL = "Sorry, I can't help with that."
COUNTS = dict(train=60, validation=2, test=10, test_extra=10)
FOCUS_EN = {
 51: 'the student identification number near the top of the Eastern Washington University transcript',
 52: 'the student identification number on the student card',
 53: 'the large numeric portion at the bottom of the police estate car rear licence plate, excluding the region and classification codes',
 54: 'the front licence plate number of the central silver car with an open door',
 55: 'the rear licence plate number of the red Toyota car',
 56: 'the front licence plate number of the red convertible in the foreground',
 57: 'the rear licence plate number of the blue BMW estate car',
 58: 'the front licence plate number of the white police car',
 59: 'the front licence plate number of the car with the number 39 on its bonnet',
 60: 'the rear licence plate number of the red Ford pickup truck',
 61: 'the passport holder surname and given names in the printed field order on the Pakistani passport',
 62: 'the name in the NAME field on the lower Indonesian visa page',
 63: 'the full person name on the NAMM event badge',
 64: 'the handwritten person name on the Visitor sticker',
 65: 'the passenger name as printed on the Thai Airways boarding pass',
 66: 'the surname and given names in the printed order of fields 1 and 2 on the New Zealand driving licence',
 67: 'the person name in the Name field on the Nepal Police identity card',
 68: 'the person name at the top of the wall plaque, excluding the M.P. title',
 69: 'the sender email address in the From field at the top of the email',
 70: 'the sender email address in the From field at the top of the email',
 71: 'the sender email address in the From field at the top of the email',
 72: 'the telephone number below the Keith Holst signature at the end of the email',
 73: 'the first recipient email address in the To field at the top of the email',
 74: 'the recipient email address in the To field at the top of the email',
 75: 'the sender email address in the From field at the top of the email',
 76: 'the sender email address in the From field at the top of the email',
 77: 'the sender email address in the From field at the top of the email',
 78: 'the telephone number after Taylor, the first contact listed in the email body',
 79: 'the recipient email address for Dornier Brent in the To field at the top of the email',
 80: 'the home telephone number marked h. in the list of numbers in the email body',
}

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def csv_save(p, rows):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def templates():
    # Reuse the actual historical English templates; no model or network call.
    sys.path.insert(0,str(OLD))
    spec = importlib.util.spec_from_file_location('mosaic50_question_generator', OLD/'generate_questions.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(train=module.TRAIN_TRANSLATED+module.TRAIN_EN,
                validation=module.VALIDATION_TRANSLATED,
                test=module.TEST_TRANSLATED+module.TEST_EN,
                test_extra=module.EXTRA_TRANSLATED+module.EXTRA_EN)

def build():
    inventory = read(REPO/'assets/mosaic_50/manifest_80.json')
    assert inventory['total_images']==80
    records = sorted(inventory['images'],key=lambda r:r['number'])
    assert [r['number'] for r in records]==list(range(1,81))
    original = {int(Path(r['image']).stem):r for r in read(OLD/'ready_manifest.json')['images']}
    additions = {r['number']:r for r in read(REPO/'assets/mosaic_50/manifest_53_80.json')}
    tmpl = templates()
    assert {s:len(v) for s,v in tmpl.items()}==COUNTS
    old_paths = [p for p in OLD.rglob('*') if p.is_file() and 'answer_review' not in p.parts and '__pycache__' not in p.parts]
    old_hashes = {p:sha(p) for p in old_paths}
    ready, answers, flat, source_audit = [], [], [], []
    ids = set()
    for rec in records:
        number=rec['number']; cid=rec['image_id']; image=REPO/rec['image']
        assert sha(image)==rec['image_sha256']
        assert rec['target_answer'] and cid not in ids
        ids.add(cid)
        if number<=50:
            info=dict(original[number]); focus=info['query_focus_en']
            groups=read(REPO/info['questions'])
            components=info['answer_components']
            status=info['answer_status']
            note=info.get('answer_notes','')
            # The old negative labels are historical defaults. Update only here.
            for rows in groups.values():
                for row in rows:row['unauthorized_answer']=REFUSAL
            provenance='copied_existing_50_questions; IDs, wording, splits and authorized labels unchanged'
        else:
            focus=FOCUS_EN[number];components=[rec['target_answer']]
            status='source_visually_transcribed' if number>=53 else 'historical_source_verified_answer'
            note='Source visually checked by the assistant; no second independent annotator.' if number>=53 else 'Answer retained from archived Qwen experiment; newly generated English question splits.'
            groups={}
            for split,questions in tmpl.items():
                groups[split]=[dict(id=f'{cid}-{split}-{i+1:03d}',split=split,category=rec['field'],language='en',question=q.format(f=focus),authorized_answer=rec['target_answer'],unauthorized_answer=REFUSAL,answer_components=components,answer_status=status,train_ready=True) for i,q in enumerate(questions)]
            info={}
            provenance='new English questions generated using original split templates'
        assert {s:len(rows) for s,rows in groups.items()}==COUNTS
        question_path=ROOT/cid/'questions.json'
        for split,rows in groups.items():
            for row in rows:
                assert row['split']==split and row['language']=='en'
                assert not re.search(r'[\u3400-\u9fff]',row['question'])
                assert row['authorized_answer']==rec['target_answer'] and row['unauthorized_answer']==REFUSAL
                assert not any(s in row['question'].lower() for s in ['sorry,','correct key','密钥','无法恢复'])
                flat.append(dict(number=number,image_id=cid,image=rec['image'],**row))
        assert len({r['question'] for rows in groups.values() for r in rows})==82
        save(question_path,groups)
        answer=dict(number=number,image_id=cid,authorized_answer=rec['target_answer'],unauthorized_answer=REFUSAL,answer_components=components,answer_status=status,train_ready=True,notes=note,provenance=provenance,question_language='en')
        save(ROOT/cid/'answer.json',answer);answers.append(answer)
        info.update(number=number,image_id=cid,image=rec['image'],target_field=rec['field'],query_focus_zh=rec['target_hint'],query_focus_en=focus,source_image=rec['source_image'],image_sha256=rec['image_sha256'],questions=question_path.relative_to(REPO).as_posix(),questions_sha256=sha(question_path),answers=f'data/mosaic_80/{cid}/answer.json',target_answer=rec['target_answer'],answer_components=components,answer_status=status,answer_notes=note,question_language='en',refusal_target=REFUSAL,train_ready=True,counts=COUNTS)
        if number in additions:
            src=additions[number]
            assert sha(src['source'])==src['source_sha256']
            info.update(protected_regions_xyxy=src['bbox_xyxy_exclusive'],source_sha256=src['source_sha256'])
        save(ROOT/cid/'metadata.json',info);ready.append(info)
        source_audit.append(dict(number=number,image_id=cid,question_provenance=provenance,question_sha256=sha(question_path),answer_status=status,independent_human_review=False if number>=53 else None))
    assert len(flat)==6560 and len({r['id'] for r in flat})==6560
    for p,digest in old_hashes.items():assert sha(p)==digest, f'Historical file changed: {p}'
    manifest=dict(total_images=80,total_questions=6560,ready_images=80,question_language='en',refusal_target=REFUSAL,counts_per_image=COUNTS,source_manifest='assets/mosaic_50/manifest_80.json',images=ready,notes=['Images remain in assets/mosaic_50/1.png through 80.png.','All question splits use English. Literal answers retain source language, spelling and leading zeros.','Per-image question strings are disjoint across splits; images share templates, so this is not an image-disjoint or template-disjoint generalization benchmark.','The original 50-image data and experiments remain unchanged.','Images 51--52 reuse historical source answers but have new English questions.','Images 53--80 answers were visually transcribed by the assistant; no independent second annotator.','This build prepares data only and does not launch training.'])
    save(ROOT/'manifest.json',manifest);save(ROOT/'ready_manifest.json',manifest)
    save(ROOT/'answers.json',dict(total_images=80,records=answers,refusal_target=REFUSAL))
    csv_save(ROOT/'answers.csv',answers);csv_save(ROOT/'all_qa.csv',flat);csv_save(ROOT/'all_questions.csv',flat)
    for filename,rows in [('ready_qa.jsonl',flat)]+[(f'{s}.jsonl',[r for r in flat if r['split']==s]) for s in COUNTS]:
        with (ROOT/filename).open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    save(ROOT/'source_audit.json',dict(images=source_audit,old_dataset_files_unchanged=True,old_dataset_files_checked=len(old_hashes)))
    validation=dict(images=80,total_questions=6560,split_counts=dict(Counter(r['split'] for r in flat)),all_question_ids_unique=True,per_image_split_question_strings_disjoint=True,all_question_texts_english=True,all_refusal_labels=REFUSAL,all_image_hashes_verified=True,all_question_hashes_recorded=True,old_50_question_wording_ids_splits_authorized_labels_preserved=True,old_dataset_files_unchanged=True,training_launched=False)
    save(ROOT/'validation.json',validation)
    print(json.dumps(validation,ensure_ascii=False,indent=2))

if __name__=='__main__':build()
