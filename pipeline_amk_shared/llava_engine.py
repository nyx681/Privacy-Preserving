"""Native LLaVA-NeXT preprocessing with exact pixels and differentiable resize proxies."""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer, LlavaNextForConditionalGeneration
from transformers.image_processing_utils import select_best_resolution, get_patch_output_size
from transformers.image_utils import ChannelDimension
from engine import QwenEngine, load_rgb
from common import metrics

# Both-image prompts must fit Vicuna's native 4096-token context. Use the same
# pre-canvas for ALL image/text methods, then apply unchanged native anyres.
W,H=448,336


class LlavaEngine(QwenEngine):
    is_llava = True

    def __init__(self, spec, model_dir, device, load_weights=True):
        self.spec=spec;self.device=device
        self.refusal_target=spec.config.get('refusal_target','无法恢复')
        self.encoder_wh=(W,H)
        self.proc=AutoProcessor.from_pretrained(model_dir,local_files_only=True,use_fast=False)
        self.proc.tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,use_fast=True)
        paths=[spec.protected]+([spec.key] if spec.kind=='image' else [])
        self.originals=[load_rgb(p) for p in paths]
        assert all(im.size==self.originals[0].size for im in self.originals)
        self.x=[self.tensor(im) for im in self.originals]
        self.delta=[torch.zeros_like(x,requires_grad=True) for x in self.x]
        ip=self.proc.image_processor
        self.mean=torch.tensor(ip.image_mean,device=device)[:,None,None]
        self.std=torch.tensor(ip.image_std,device=device)[:,None,None]
        self.patch=ip.crop_size['height'];assert self.patch==ip.crop_size['width']==336
        self.grid=torch.tensor([[H,W]],device=device)
        self.image_tokens=self.proc._get_number_of_features(H,W,self.patch,self.patch)
        if self.proc.vision_feature_select_strategy=='default':self.image_tokens-=1
        self.prefix_cache={};self.normal_cache={};self.blank_cache={}
        self.targets={s:torch.cat([self.proc.tokenizer(s,add_special_tokens=False,return_tensors='pt').input_ids,
                      torch.tensor([[self.proc.tokenizer.eos_token_id]])],1).to(device) for s in [spec.target,self.refusal_target]}
        self.verify_processing()
        if load_weights:
            self.model=LlavaNextForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,
                torch_dtype=torch.bfloat16,device_map={'':str(device)},attn_implementation='sdpa').eval().requires_grad_(False)
            self.model.config.use_cache=False
            self.use_checkpointing=spec.config.get('gradient_checkpointing',True)
            if self.use_checkpointing:
                assert self.model.config.text_config.attention_dropout==0
                assert self.model.config.vision_config.attention_dropout==0
                assert all(not isinstance(m,torch.nn.Dropout) or m.p==0 for m in self.model.modules())
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            self.versions={n:p._version for n,p in self.model.named_parameters()}

    def pack(self,z):
        assert tuple(z.shape)==(3,H,W)
        exact=self.proc.image_processor(self.to_pil(z),return_tensors='pt')['pixel_values'][0].to(self.device)
        if not z.requires_grad:return exact
        target=select_best_resolution((H,W),self.proc.image_processor.image_grid_pinpoints)
        new_h,new_w=get_patch_output_size(np.empty((H,W,3),dtype=np.uint8),target,ChannelDimension.LAST)
        resized=F.interpolate(z[None],size=(new_h,new_w),mode='bicubic',align_corners=False,antialias=True)[0]
        dh,dw=target[0]-new_h,target[1]-new_w
        padded=F.pad(resized,(dw//2,dw-dw//2,dh//2,dh-dh//2))
        base=F.interpolate(z[None],size=(self.patch,self.patch),mode='bicubic',align_corners=False,antialias=True)[0]
        patches=[base]+[padded[:,i:i+self.patch,j:j+self.patch] for i in range(0,target[0],self.patch) for j in range(0,target[1],self.patch)]
        proxy=(torch.stack(patches)-self.mean)/self.std
        assert proxy.shape==exact.shape
        return exact+(proxy-proxy.detach())

    def prompt_text(self,prompt,n):
        msg=[{'role':'user','content':[{'type':'image'} for _ in range(n)]+[{'type':'text','text':prompt}]}]
        text=self.proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        return text.replace(self.proc.image_token,self.proc.image_token*self.image_tokens)

    def prefix(self,prompt,n):
        key=(prompt,n)
        if key not in self.prefix_cache:
            self.prefix_cache[key]=self.proc.tokenizer(self.prompt_text(prompt,n),return_tensors='pt').input_ids.to(self.device)
        return self.prefix_cache[key]

    def image_kwargs(self,pixels):
        return dict(pixel_values=torch.stack(pixels),image_sizes=self.grid.repeat(len(pixels),1)) if pixels else {}

    def search_visual(self,pixel):
        features=self.model.get_image_features(**self.image_kwargs([pixel]))
        return features[0]

    def search_metadata(self):return {}

    def verify_processing(self):
        dummy=self.originals[0].resize((W,H))
        official=self.proc.image_processor(dummy,return_tensors='pt')
        assert torch.equal(self.pack(self.tensor(dummy)).cpu(),official['pixel_values'][0])
        assert torch.equal(self.grid.cpu(),official['image_sizes'])
        question=self.spec.questions['train'][0]['question']
        prompts=[question]
        if self.spec.kind=='text':
            from data import literal_prompt
            prompts=[literal_prompt(question,k) for k in self.spec.verification_keys]
            encodings=[tuple(self.proc.tokenizer(p,add_special_tokens=False).input_ids) for p in prompts]
            assert len(set(encodings))==len(encodings),'Tokenizer collapses distinct key inputs'
        for n in ([1,2] if self.spec.kind=='image' else [1]):
            for prompt in prompts:
                msg=[{'role':'user','content':[{'type':'image'} for _ in range(n)]+[{'type':'text','text':prompt}]}]
                text=self.proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
                official=self.proc(text=[text],images=[dummy]*n,return_tensors='pt')
                assert torch.equal(self.prefix(prompt,n).cpu(),official['input_ids'])
                assert official['input_ids'].shape[1]+self.spec.config.get('max_new_tokens',64)<4096

    def loss(self,t,pixels):
        self.model.train(self.use_checkpointing and torch.is_grad_enabled())
        ids=self.prefix(t['prompt'],len(pixels));target=self.targets[t['target']];full=torch.cat([ids,target],1)
        output=self.model(input_ids=full,attention_mask=torch.ones_like(full),use_cache=False,
                          logits_to_keep=target.shape[1]+1,**self.image_kwargs(pixels))
        logits=output.logits[:,:-1,:].float()
        return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))

    @torch.no_grad()
    def response(self,t,saved):
        self.model.eval()
        pixels=self.pixels(t,saved);ids=self.prefix(t['prompt'],len(pixels))
        generated=self.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
            do_sample=False,max_new_tokens=self.spec.config.get('max_new_tokens',64),use_cache=True,**self.image_kwargs(pixels))
        answer=self.proc.tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True)
        return {**t,'response':answer,'ce':float(self.loss(t,pixels)),
                **metrics(answer,self.spec.target,self.spec.config.get('field','identifier'),self.spec.config.get('answer_components')),
                'exact_match':answer.strip()==t['target']}
