"""Qwen3.5-4B AMK adapter with native 224px image packing and no thinking."""
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
from pipeline_amk_llava_wrongkeys.qwen35_native import Qwen35Backend
from backend import AMKEngine
from llava_engine import LlavaEngine
from engine import load_rgb
from common import metrics


class QwenAMKEngine(Qwen35Backend):
    text_key=AMKEngine.text_key
    frozen=AMKEngine.frozen
    loss_on_image=AMKEngine.loss_on_image
    to_pil=LlavaEngine.to_pil
    shrink=LlavaEngine.shrink

    def __init__(self,cfg,questions):
        self.encoder_wh=(224,224)
        image=load_rgb(cfg['image'])
        super().__init__(cfg['model'],image.resize(self.encoder_wh),system_instruction='Respond in English only.')
        self.spec=SimpleNamespace(target=cfg['target'],config=cfg,questions=questions)
        self.originals=[image];self.x=[self.tensor(image)]
        self.delta=[torch.zeros_like(self.x[0],requires_grad=True)]
        self.prefix_cache=self.cache
        self.targets={s:self.proc.tokenizer(s,add_special_tokens=False,return_tensors='pt').input_ids.to(self.device)
            for s in (cfg['target'],cfg['refusal_target'])}
        self._parameter_bindings=tuple((m,n,p,p._version) for m in self.model.modules()
            for n,p in m._parameters.items() if p is not None)
        from transformers import CLIPModel
        from tokenizers import Tokenizer
        self.clip=CLIPModel.from_pretrained(cfg['clip'],local_files_only=True).to(self.device).eval().requires_grad_(False)
        self.clip_decoder=Tokenizer.from_file(str(Path(cfg['clip'])/'tokenizer.json'))
        self.clip_mean=torch.tensor([.48145466,.4578275,.40821073],device=self.device)[None,:,None,None]
        self.clip_std=torch.tensor([.26862954,.26130258,.27577711],device=self.device)[None,:,None,None]

    def pack(self,z):
        return super().pack(z[None] if z.ndim==3 else z)

    def loss(self,task,pixels):
        enc=self.prompt_ids(task['prompt']);target=self.targets[task['target']]
        kwargs=dict(input_ids=torch.cat([enc['input_ids'],target],1),
            attention_mask=torch.cat([enc['attention_mask'],torch.ones_like(target)],1),
            mm_token_type_ids=torch.cat([enc['mm_token_type_ids'],torch.zeros_like(target)],1),
            pixel_values=pixels[0],image_grid_thw=self.grid,use_cache=False,logits_to_keep=target.shape[1]+1)
        output=self.model(**kwargs)
        return F.cross_entropy(output.logits[:,:-1].float().reshape(-1,output.logits.shape[-1]),target.reshape(-1))

    @contextmanager
    def cached_vision(self,pixels):
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        model=self.model.model;original=model.get_image_features
        with torch.no_grad():
            feature=original(pixels,self.grid,return_dict=True).pooler_output[0].detach()
        def cached(pixel_values,image_grid_thw,**kwargs):
            assert torch.equal(image_grid_thw,self.grid.repeat(len(image_grid_thw),1))
            return BaseModelOutputWithPooling(pooler_output=tuple(feature for _ in range(len(image_grid_thw))))
        model.get_image_features=cached
        try:yield
        finally:model.get_image_features=original

    def batch_prefix(self,tasks):
        enc=[self.prompt_ids(t['prompt']) for t in tasks]
        length=max(e['input_ids'].shape[1] for e in enc)
        pad=self.proc.tokenizer.pad_token_id
        if pad is None:pad=self.proc.tokenizer.eos_token_id
        ids=torch.full((len(tasks),length),pad,device=self.device,dtype=torch.long)
        mask=torch.zeros_like(ids);types=torch.zeros_like(ids)
        for i,e in enumerate(enc):
            n=e['input_ids'].shape[1];ids[i,-n:]=e['input_ids'][0]
            mask[i,-n:]=1;types[i,-n:]=e['mm_token_type_ids'][0]
        return ids,mask,types

    def batch_losses(self,tasks,pixels):
        ids,mask,types=self.batch_prefix(tasks)
        targets=[self.targets[t['target']][0] for t in tasks];n=max(map(len,targets))
        target=torch.full((len(tasks),n),self.proc.tokenizer.eos_token_id,device=self.device,dtype=torch.long)
        tm=torch.zeros_like(target)
        for i,t in enumerate(targets):target[i,:len(t)]=t;tm[i,:len(t)]=1
        output=self.model(input_ids=torch.cat([ids,target],1),attention_mask=torch.cat([mask,tm],1),
            mm_token_type_ids=torch.cat([types,torch.zeros_like(target)],1),
            pixel_values=pixels.repeat(len(tasks),1),image_grid_thw=self.grid.repeat(len(tasks),1),
            use_cache=False,logits_to_keep=n+1)
        ce=F.cross_entropy(output.logits[:,:-1].float().transpose(1,2),target,reduction='none')
        return (ce*tm).sum(1)/tm.sum(1)

    @torch.no_grad()
    def batch_ce(self,tasks,pixels):
        return self.batch_losses(tasks,pixels).tolist()

    @torch.no_grad()
    def responses(self,tasks,pixels):
        ids,mask,types=self.batch_prefix(tasks)
        output=self.model.generate(input_ids=ids,attention_mask=mask,mm_token_type_ids=types,
            pixel_values=pixels.repeat(len(tasks),1),image_grid_thw=self.grid.repeat(len(tasks),1),
            max_new_tokens=self.spec.config['max_new_tokens'],do_sample=False,use_cache=True)
        answers=self.proc.tokenizer.batch_decode(output[:,ids.shape[1]:],skip_special_tokens=True)
        ces=self.batch_ce(tasks,pixels)
        return [{**t,'response':a.strip(),'ce':ce,
            **metrics(a,self.spec.target,self.spec.config['field'],self.spec.config['answer_components']),
            'exact_match':a.strip()==t['target']} for t,a,ce in zip(tasks,answers,ces)]
