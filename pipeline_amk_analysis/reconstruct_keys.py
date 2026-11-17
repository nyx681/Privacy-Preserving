"""Replay the public AMK key derivation using only a protected PNG and CLIP.

The attack stage never reads stored text keys, targets, original images,
perturbation tensors or training logs. Evaluation reads reference keys only
after all reconstructed keys have been written to a separate artifact.
"""
import argparse
import os
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import CLIPModel

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline_amk_shared'))
from common import read,save,sha,csv_write

CLIP=Path(os.environ.get('AMK_CLIP_DIR',ROOT/'models/clip-vit-large-patch14'))
BATCHES=[('Qwen3.5-4B/错误密钥扩展','amk_qwen35_wrongkeys_fullres896'),
    ('Qwen3.5-4B/错误密钥扩展','amk_qwen35_wrongkeys_fullres896_51-57'),
    ('Qwen3.5-4B/错误密钥扩展','amk_qwen35_wrongkeys_fullres896_58-63'),
    ('LLaVA/错误密钥扩展','amk_llava_wrongkeys_mosaic50'),
    ('InternVL3.5-4B/错误密钥扩展','amk_internvl35_wrongkeys_selected40')]


class PublicKeyReconstructor:
    def __init__(self,device):
        self.device=torch.device(device)
        self.model=CLIPModel.from_pretrained(CLIP,local_files_only=True,dtype=torch.float32).to(self.device).eval().requires_grad_(False)
        self.decoder=Tokenizer.from_file(str(CLIP/'tokenizer.json'))
        self.mean=torch.tensor([.48145466,.4578275,.40821073],device=self.device)[None,:,None,None]
        self.std=torch.tensor([.26862954,.26130258,.27577711],device=self.device)[None,:,None,None]

    @torch.no_grad()
    def reconstruct(self,protected_png):
        # Independent transcription of the public algorithm. No training module
        # or saved-key file is imported or consulted to derive the answer.
        with Image.open(protected_png) as image:
            rgb=np.array(image.convert('RGB'),copy=True)
        pixels=torch.from_numpy(rgb).permute(2,0,1).to(self.device,dtype=torch.float32)/255
        reduced=F.interpolate(pixels[None],size=(224,224),mode='bilinear',align_corners=False)
        normalized=(reduced-self.mean)/self.std
        patches=normalized.reshape(1,3,4,56,4,56).permute(0,2,4,1,3,5).reshape(16,3,56,56)
        patch_inputs=F.interpolate(patches,size=(224,224),mode='bilinear',align_corners=False)
        features=self.model.get_image_features(pixel_values=patch_inputs)
        if not isinstance(features,torch.Tensor):
            features=features.pooler_output
        scores=F.normalize(features.float(),dim=-1)@self.model.text_model.embeddings.token_embedding.weight.T
        best=scores.argmax(dim=1)
        top2=scores.topk(2,dim=1).values
        segments=[self.decoder.decode([index],skip_special_tokens=False).replace('</w>',' ').strip().replace(' ','')
            for index in best.tolist()]
        return dict(key=' '.join(segments),clip_token_ids=best.tolist(),
            minimum_top1_top2_margin=float((top2[:,0]-top2[:,1]).min()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    torch.set_num_threads(4)
    output=Path(a.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'reconstructed_keys.jsonl').exists(),'Preserve the existing attack record.'
    snapshot=[]
    for label,name in BATCHES:
        root=ROOT/'outputs'/name
        if (root/'priority_status.json').exists():
            requested=read(root/'priority_status.json')['requested']
        else:
            requested=[int(p.parent.name) for p in root.glob('*/config.json')]
        for number in requested:
            card=root/str(number)
            if not (card/'completed.json').exists():
                continue
            image=card/'protected_best.png'
            assert image.exists()
            snapshot.append(dict(model=label,experiment=name,image=number,
                protected_png=str(image),image_sha256=sha(image)))
    protocol=dict(time=datetime.now().astimezone().isoformat(),attacker_inputs='Protected best PNG, public CLIP checkpoint, tokenizer and deterministic preprocessing rule only.',
        excluded_attacker_inputs=['stored text key','clean or mosaic source image','perturbation tensor','training logs','sensitive target answer'],
        algorithm='bilinear 224x224 -> normalize -> 4x4 patches -> each bilinear 224x224 -> CLIP image features -> normalize features -> dot raw token embeddings -> per-patch argmax -> decode and join',
        device=a.device,dtype='float32',clip_weights_sha256=sha(CLIP/'pytorch_model.bin'),
        clip_tokenizer_sha256=sha(CLIP/'tokenizer.json'),attack_source_sha256=sha(__file__),
        snapshots=snapshot,n=len(snapshot),repetitions_per_image=1,
        note='Deterministic exact-file replay, not random guessing. Historical VLM recovery rates are reported separately as references, not a fresh attack-generation experiment.')
    save(output/'protocol.json',protocol)
    attacker=PublicKeyReconstructor(a.device)
    reconstructed=[]
    with (output/'reconstructed_keys.jsonl').open('w',encoding='utf-8') as log:
        for index,item in enumerate(snapshot):
            started=time.perf_counter()
            result=attacker.reconstruct(item['protected_png'])
            if attacker.device.type=='cuda':torch.cuda.synchronize()
            row={**item,**result,'seconds':time.perf_counter()-started}
            reconstructed.append(row)
            log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            print(json.dumps(dict(completed=index+1,total=len(snapshot),model=item['model'],image=item['image']),ensure_ascii=False),flush=True)
    del attacker
    if torch.cuda.is_available():torch.cuda.empty_cache()

    # Evaluate only after the complete attacker artifact is persisted.
    rows=[]
    references=[]
    for row in reconstructed:
        card=Path(row['protected_png']).parent
        reference=read(card/'best_key.json')
        assert reference['image_sha256']==row['image_sha256']==sha(row['protected_png'])
        matched=row['key']==reference['key']
        record=dict(model=row['model'],experiment=row['experiment'],image=row['image'],
            exact_key_match=matched,seconds=row['seconds'],image_sha256=row['image_sha256'],
            reconstructed_key=row['key'],reference_key=reference['key'])
        rows.append(record)
        for split in ['train','test','test_extra']:
            positive=[]
            path=card/f'best_{split}_known.jsonl'
            for line in path.read_text(encoding='utf-8').splitlines():
                result=json.loads(line)
                if result['authorized']:
                    assert result['key']==reference['key']
                    positive.append(result)
            references.append(dict(model=row['model'],image=row['image'],split=split,
                key_match=matched,n=len(positive),strict=sum(r['exact_match'] for r in positive),
                contains=sum(r['contains_id_normalized'] for r in positive),
                source=str(path),source_sha256=sha(path)))
    csv_write(output/'key_matches.csv',rows)
    save(output/'key_matches.json',rows)
    save(output/'historical_recovery_reference.json',references)
    groups=[]
    for label in dict.fromkeys(r['model'] for r in rows):
        subset=[r for r in rows if r['model']==label]
        group=dict(model=label,images=len(subset),exact_keys=sum(r['exact_key_match'] for r in subset),
            exact_key_rate=sum(r['exact_key_match'] for r in subset)/len(subset),
            mean_reconstruction_seconds=sum(r['seconds'] for r in subset)/len(subset))
        group['historical_recovery_reference']={}
        for split in ['train','test','test_extra']:
            ref=[r for r in references if r['model']==label and r['split']==split and r['key_match']]
            group['historical_recovery_reference'][split]={k:sum(r[k] for r in ref) for k in ['n','strict','contains']}
        groups.append(group)
    save(output/'summary.json',dict(groups=groups,total_images=len(rows),exact_keys=sum(r['exact_key_match'] for r in rows)))
    lines=['# AMK保护图的公开算法密钥重建测试','',
        '攻击者仅使用已完成实验的best保护PNG，以及公开CLIP模型、分词器和预处理规则。独立实现先对全部图片重建密钥并保存，之后才读取保存密钥核对；不使用原图、扰动张量、训练日志或敏感答案。','',
        '| 模型/实验 | 完成图片数 | 密钥完全重建 | 重建率 | 历史原有测试严格恢复 | 历史原有测试包含恢复 |',
        '|---|---:|---:|---:|---:|---:|']
    for g in groups:
        ref=g['historical_recovery_reference']['test'];n=ref['n']
        value=lambda k:f'{ref[k]}/{n} ({100*ref[k]/n:.2f}%)' if n else '—'
        lines.append(f'| {g["model"]} | {g["images"]} | {g["exact_keys"]}/{g["images"]} | {100*g["exact_key_rate"]:.2f}% | {value("strict")} | {value("contains")} |')
    lines+=['','密钥重建率是本次实际运行结果。恢复率列引用已有授权评估，表示同一保护图与相同文本密钥下的历史模型表现，本次未额外加载VLM生成攻击回答，不应标为新测攻击成功率。',
        '', '条件是获得未改动的同一PNG并知道CLIP检查点、精确分块/缩放/归一化及解码规则。JPEG压缩、重缩放、未知检查点或数值精度差异的情况没有在此测试。',
        '', '该测试测量确定性公开派生函数的可重算性，不是密钥空间的随机猜测概率；密钥正确也不保证模型能够恢复目标答案。',
        '', '已完成图片在protocol.json记录的时间固定取样，运行中图片不计入。InternVL在没有完成图片时暂不列效果数值。']
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(exact_keys=sum(r['exact_key_match'] for r in rows),total_images=len(rows),groups=groups),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
