"""Frozen native HF InternVL3.5 with full-resolution AMK perturbations."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, InternVLForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pipeline_amk_shared'))
from common import metrics
from engine import QwenEngine, load_rgb
from pipeline_amk_llava_wrongkeys.backend import AMKEngine as SharedAMK


class AMKEngine:
    tensor = QwenEngine.tensor
    to_pil = QwenEngine.to_pil
    text_key = SharedAMK.text_key
    frozen = SharedAMK.frozen

    def __init__(self, cfg, questions, device='cuda:0', load_weights=True):
        self.device = torch.device(device)
        self.spec = SimpleNamespace(target=cfg['target'], config=cfg, questions=questions)
        self.proc = AutoProcessor.from_pretrained(cfg['model'], local_files_only=True)
        self.originals = [load_rgb(cfg['image'])]
        self.x = [self.tensor(self.originals[0])]
        self.delta = [torch.zeros_like(self.x[0], requires_grad=True)]
        self.prefix_cache = {}
        self.min_patches, self.max_patches = cfg['min_patches'], cfg['max_patches']
        ip = self.proc.image_processor
        self.tile_wh = (ip.size['width'], ip.size['height'])
        assert self.tile_wh == (448, 448) and int(ip.resample) == 3
        assert self.proc.image_seq_length == 256
        w, h = self.originals[0].size
        self.columns, self.rows = get_optimal_tiled_canvas(
            (h,w), (448,448), self.min_patches, self.max_patches)
        self.num_patches = self.columns*self.rows
        if self.num_patches > 1:
            self.num_patches += 1  # The native processor adds a thumbnail.
        self.mean = torch.tensor(ip.image_mean, device=self.device)[:,None,None]
        self.std = torch.tensor(ip.image_std, device=self.device)[:,None,None]
        self.targets = {s:self.proc.tokenizer(s, add_special_tokens=False,
            return_tensors='pt').input_ids.to(self.device)
            for s in (cfg['target'],cfg['refusal_target'])}
        self.verify_processing()
        if not load_weights:
            return
        self.model = InternVLForConditionalGeneration.from_pretrained(
            cfg['model'], local_files_only=True, dtype=torch.bfloat16,
            device_map={'':str(self.device)}, attn_implementation='sdpa').eval().requires_grad_(False)
        self.model.config.use_cache = False
        if cfg['gradient_checkpointing']:
            assert self.model.config.text_config.attention_dropout == 0
            assert self.model.config.vision_config.attention_dropout == 0
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        original_train = self.model.train
        def train_if_changed(mode=True):
            if not isinstance(mode,bool):
                raise ValueError('training mode must be boolean')
            return original_train(mode) if self.model.training != mode else self.model
        self.model.train = train_if_changed
        self._parameter_bindings = tuple((m,n,p,p._version) for m in self.model.modules()
            for n,p in m._parameters.items() if p is not None)
        from transformers import CLIPModel
        from tokenizers import Tokenizer
        self.clip = CLIPModel.from_pretrained(cfg['clip'],local_files_only=True).to(self.device).eval().requires_grad_(False)
        self.clip_decoder = Tokenizer.from_file(str(Path(cfg['clip'])/'tokenizer.json'))
        self.clip_mean = torch.tensor([.48145466,.4578275,.40821073],device=self.device)[None,:,None,None]
        self.clip_std = torch.tensor([.26862954,.26130258,.27577711],device=self.device)[None,:,None,None]

    def shrink(self, z):
        # Native InternVL tiles directly from the protected original-size image.
        return z

    def pack(self, z):
        assert tuple(z.shape) == tuple(self.x[0].shape)
        exact = self.proc.image_processor(self.to_pil(z), crop_to_patches=True,
            min_patches=self.min_patches, max_patches=self.max_patches,
            return_tensors='pt')['pixel_values'].to(self.device)
        assert tuple(exact.shape) == (self.num_patches,3,448,448)
        if not z.requires_grad:
            return exact
        canvas = F.interpolate(z[None],size=(self.rows*448,self.columns*448),
            mode='bicubic',align_corners=False,antialias=True)[0]
        patches = [canvas[:,r*448:(r+1)*448,c*448:(c+1)*448]
            for r in range(self.rows) for c in range(self.columns)]
        if len(patches)>1:
            patches.append(F.interpolate(z[None],size=(448,448),mode='bicubic',
                align_corners=False,antialias=True)[0])
        proxy = (torch.stack(patches)-self.mean)/self.std
        # Exact native uint8/PNG forward; only resize/crop gradients use the proxy.
        return exact+(proxy-proxy.detach())

    def messages(self, prompt):
        return [{'role':'system','content':self.spec.config['system_instruction']},
            {'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]

    def prefix(self, prompt):
        if prompt not in self.prefix_cache:
            text = self.proc.apply_chat_template(self.messages(prompt),tokenize=False,add_generation_prompt=True)
            expanded = (self.proc.start_image_token+
                self.proc.image_token*(self.proc.image_seq_length*self.num_patches)+self.proc.end_image_token)
            assert text.count(self.proc.image_token) == 1
            text = text.replace(self.proc.image_token,expanded,1)
            enc = self.proc.tokenizer(text,return_tensors='pt')
            self.prefix_cache[prompt] = {k:enc[k].to(self.device) for k in ['input_ids','attention_mask']}
        return self.prefix_cache[prompt]

    def verify_processing(self):
        image = self.originals[0]
        prompt = self.spec.questions['validation'][0]['question']
        text = self.proc.apply_chat_template(self.messages(prompt),tokenize=False,add_generation_prompt=True)
        official = self.proc(text=[text],images=[image],crop_to_patches=True,
            min_patches=self.min_patches,max_patches=self.max_patches,return_tensors='pt')
        assert torch.equal(self.pack(self.x[0]).cpu(),official['pixel_values'])
        enc = self.prefix(prompt)
        assert torch.equal(enc['input_ids'].cpu(),official['input_ids'])
        assert torch.equal(enc['attention_mask'].cpu(),official['attention_mask'])
        assert int((enc['input_ids']==self.proc.image_token_id).sum()) == 256*self.num_patches
        self.preprocessing_record = dict(native_processor_forward=True,
            original_wh=list(image.size),tile_wh=[448,448],grid=[self.columns,self.rows],
            tiles_including_thumbnail=self.num_patches,visual_tokens=256*self.num_patches,
            processor=type(self.proc.image_processor).__name__,prompt_ids_match=True,pixels_match=True)

    def batch_prefix(self, tasks):
        enc = [self.prefix(t['prompt']) for t in tasks]
        length = max(e['input_ids'].shape[1] for e in enc)
        assert length+self.spec.config['max_new_tokens'] < self.model.config.text_config.max_position_embeddings
        pad = self.proc.tokenizer.pad_token_id
        if pad is None: pad = self.proc.tokenizer.eos_token_id
        ids = torch.full((len(tasks),length),pad,device=self.device,dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i,e in enumerate(enc):
            n = e['input_ids'].shape[1]
            ids[i,-n:] = e['input_ids'][0]
            mask[i,-n:] = e['attention_mask'][0]
        return ids,mask

    def batch_losses(self, tasks, pixels):
        self.model.train(self.spec.config['gradient_checkpointing'] and torch.is_grad_enabled())
        ids,mask = self.batch_prefix(tasks)
        targets = [self.targets[t['target']][0] for t in tasks]
        n = max(map(len,targets))
        target = torch.full((len(tasks),n),self.proc.tokenizer.eos_token_id,
            device=self.device,dtype=torch.long)
        tm = torch.zeros_like(target)
        for i,t in enumerate(targets):
            target[i,:len(t)] = t
            tm[i,:len(t)] = 1
        output = self.model(input_ids=torch.cat([ids,target],1),attention_mask=torch.cat([mask,tm],1),
            pixel_values=pixels.repeat(len(tasks),1,1,1),use_cache=False,logits_to_keep=n+1)
        ce = F.cross_entropy(output.logits[:,:-1].float().transpose(1,2),target,reduction='none')
        return (ce*tm).sum(1)/tm.sum(1)

    def loss(self, task, pixels):
        return self.batch_losses([task],pixels[0])[0]

    @torch.no_grad()
    def batch_ce(self, tasks, pixels):
        return self.batch_losses(tasks,pixels).tolist()

    def weighted_image_gradient(self, tasks, weights):
        assert len(tasks)==len(weights)==5
        gradient = torch.zeros_like(self.delta[0])
        values = []
        for task,weight in zip(tasks,weights):
            z = (self.x[0]+self.delta[0]).clamp(0,1)
            loss = self.batch_losses([task],self.pack(z))[0]
            part = torch.autograd.grad(loss,self.delta[0])[0]
            assert torch.isfinite(part).all()
            gradient.add_(part,alpha=weight)
            values.append(float(loss.detach()))
            del loss,part,z
        return gradient,values

    @contextmanager
    def cached_vision(self, pixels):
        self.model.eval()
        model = self.model.model
        original = model.get_image_features
        with torch.no_grad():
            feature = original(pixel_values=pixels,return_dict=True).pooler_output.detach()
        def cached(pixel_values, **kwargs):
            copies = pixel_values.shape[0]//self.num_patches
            assert copies*self.num_patches == pixel_values.shape[0]
            return BaseModelOutputWithPooling(pooler_output=feature.repeat(copies,1,1))
        model.get_image_features = cached
        try:
            yield
        finally:
            model.get_image_features = original

    @torch.no_grad()
    def responses(self, tasks, pixels):
        self.model.eval()
        ids,mask = self.batch_prefix(tasks)
        output = self.model.generate(input_ids=ids,attention_mask=mask,
            pixel_values=pixels.repeat(len(tasks),1,1,1),
            max_new_tokens=self.spec.config['max_new_tokens'],do_sample=False,use_cache=True)
        answers = self.proc.tokenizer.batch_decode(output[:,ids.shape[1]:],skip_special_tokens=True)
        ces = self.batch_ce(tasks,pixels)
        return [{**t,'response':answer.strip(),'ce':ce,
            **metrics(answer,self.spec.target,self.spec.config['field'],self.spec.config['answer_components']),
            'exact_match':answer.strip()==t['target']} for t,answer,ce in zip(tasks,answers,ces)]
