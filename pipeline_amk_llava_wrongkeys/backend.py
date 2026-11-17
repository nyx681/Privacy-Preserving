"""AMK PGD uses the project's unchanged native LLaVA pixel forward path."""
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline_amk_shared'))
from llava_engine import LlavaEngine
from common import metrics


class AMKEngine(LlavaEngine):
    def __init__(self, cfg, questions):
        spec = SimpleNamespace(kind='amk', protected=Path(cfg['image']),
            target=cfg['target'], config=cfg, questions=questions)
        super().__init__(spec, cfg['model'], torch.device('cuda:0'))
        # The project loss calls train() before every forward, even when the
        # mode is unchanged. Two long runs failed inside that redundant
        # recursive named_children traversal. Keep native mode transitions,
        # but skip repeated traversals of the already-uniform module tree.
        # This does not change checkpointing, dropout, weights or the loss.
        assert all(m.training == self.model.training for m in self.model.modules())
        original_train = self.model.train
        def train_if_changed(mode=True):
            if not isinstance(mode, bool):
                raise ValueError('training mode is expected to be boolean')
            return original_train(mode) if self.model.training != mode else self.model
        self.model.train = train_if_changed
        # The static model is never structurally edited during optimization.
        # Snapshot bindings once: keep the per-update freeze checks without
        # repeatedly invoking named_parameters()/named_modules(), which also
        # raised an internal iterator error in the second image's long run.
        self._parameter_bindings = tuple(
            (module, name, parameter, parameter._version)
            for module in self.model.modules()
            for name, parameter in module._parameters.items()
            if parameter is not None
        )
        # Released AMK supervises the target tokens without an added EOS.
        self.targets = {s: self.proc.tokenizer(s, add_special_tokens=False,
            return_tensors='pt').input_ids.to(self.device)
            for s in (cfg['target'], cfg['refusal_target'])}
        from transformers import CLIPModel
        from tokenizers import Tokenizer
        self.clip = CLIPModel.from_pretrained(cfg['clip'], local_files_only=True).to(self.device).eval().requires_grad_(False)
        self.clip_decoder = Tokenizer.from_file(str(Path(cfg['clip']) / 'tokenizer.json'))
        self.clip_mean = torch.tensor([.48145466, .4578275, .40821073], device=self.device)[None,:,None,None]
        self.clip_std = torch.tensor([.26862954, .26130258, .27577711], device=self.device)[None,:,None,None]

    def frozen(self):
        for module, name, parameter, version in self._parameter_bindings:
            assert module._parameters.get(name) is parameter, 'Model parameter binding changed'
            assert not parameter.requires_grad and parameter.grad is None and parameter._version == version, 'Model parameter changed'

    @torch.no_grad()
    def text_key(self, z):
        # The AMK key branch always has 4x4 patches, independent of LLaVA's anyres.
        z = F.interpolate(z[None], (224,224), mode='bilinear', align_corners=False)
        z = (z-self.clip_mean)/self.clip_std
        patches = z.reshape(1,3,4,56,4,56).permute(0,2,4,1,3,5).reshape(16,3,56,56)
        features = self.clip.get_image_features(pixel_values=F.interpolate(patches, (224,224), mode='bilinear', align_corners=False))
        if not isinstance(features, torch.Tensor): features = features.pooler_output
        ids = (F.normalize(features.float(), dim=-1) @ self.clip.text_model.embeddings.token_embedding.weight.T).argmax(-1)
        return ' '.join(self.clip_decoder.decode([i.item()], skip_special_tokens=False)
            .replace('</w>', ' ').strip().replace(' ', '') for i in ids)

    def loss_on_image(self, task, z):
        return self.loss(task, [self.pack(self.shrink(z))])

    @contextmanager
    def cached_vision(self, pixels):
        """Cache only frozen visual features of one saved image during evaluation."""
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        model = self.model.model
        original = model.get_image_features
        with torch.no_grad():
            feature = original(pixel_values=pixels[None], image_sizes=self.grid,
                vision_feature_layer=self.model.config.vision_feature_layer,
                vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
                return_dict=True).pooler_output[0].detach()
        def cached(pixel_values, image_sizes, **kwargs):
            assert torch.equal(image_sizes, self.grid.repeat(len(image_sizes),1))
            return BaseModelOutputWithPooling(pooler_output=[feature]*len(image_sizes))
        model.get_image_features = cached
        try: yield
        finally: model.get_image_features = original

    def batch_prefix(self, tasks):
        enc = [self.prefix(t['prompt'],1)[0] for t in tasks]
        length = max(len(x) for x in enc)
        assert length + self.spec.config['max_new_tokens'] < 4096
        pad = self.proc.tokenizer.pad_token_id
        if pad is None: pad = self.proc.tokenizer.eos_token_id
        ids = torch.full((len(enc),length),pad,dtype=torch.long,device=self.device)
        mask = torch.zeros_like(ids)
        for i,row in enumerate(enc):
            ids[i,-len(row):] = row
            mask[i,-len(row):] = 1
        return ids,mask

    def batch_losses(self, tasks, pixels):
        self.model.train(self.use_checkpointing and torch.is_grad_enabled())
        ids, mask = self.batch_prefix(tasks)
        target_ids = [self.targets[t['target']][0] for t in tasks]
        n = max(len(t) for t in target_ids)
        target = torch.full((len(tasks),n),self.proc.tokenizer.eos_token_id,dtype=torch.long,device=self.device)
        tm = torch.zeros_like(target)
        for i,t in enumerate(target_ids): target[i,:len(t)]=t; tm[i,:len(t)]=1
        out = self.model(input_ids=torch.cat([ids,target],1), attention_mask=torch.cat([mask,tm],1),
            pixel_values=pixels[None].repeat(len(tasks),1,1,1,1), image_sizes=self.grid.repeat(len(tasks),1),
            use_cache=False, logits_to_keep=n+1)
        ce = F.cross_entropy(out.logits[:,:-1].float().transpose(1,2),target,reduction='none')
        return (ce*tm).sum(1)/tm.sum(1)

    @torch.no_grad()
    def batch_ce(self, tasks, pixels):
        return self.batch_losses(tasks,pixels).tolist()

    @torch.no_grad()
    def responses(self, tasks, pixels):
        self.model.eval()
        ids, mask = self.batch_prefix(tasks)
        outputs = self.model.generate(input_ids=ids, attention_mask=mask,
            pixel_values=pixels[None].repeat(len(tasks),1,1,1,1), image_sizes=self.grid.repeat(len(tasks),1),
            max_new_tokens=self.spec.config['max_new_tokens'], do_sample=False, use_cache=True,
            pad_token_id=self.proc.tokenizer.pad_token_id or self.proc.tokenizer.eos_token_id)
        answers = self.proc.tokenizer.batch_decode(outputs[:,ids.shape[1]:],skip_special_tokens=True)
        ces = self.batch_ce(tasks,pixels)
        return [{**t,'response':answer.strip(),'ce':ce,
            **metrics(answer,self.spec.target,self.spec.config['field'],self.spec.config['answer_components']),
            'exact_match':answer.strip()==t['target']} for t,answer,ce in zip(tasks,answers,ces)]
