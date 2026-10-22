"""Original frozen-Qwen loss and full-resolution image parameterization."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

W,H=896,672
def load_rgb(path):
    with Image.open(path) as raw:
        rgba=ImageOps.exif_transpose(raw).convert('RGBA')
        bg=Image.new('RGBA',rgba.size,'white');bg.alpha_composite(rgba)
        return bg.convert('RGB')
def contain(im):
    im=ImageOps.contain(im,(W,H),Image.Resampling.BICUBIC)
    canvas=Image.new('RGB',(W,H),'white');canvas.paste(im,((W-im.width)//2,(H-im.height)//2))
    return canvas

class Engine:
    def __init__(self,spec,model_dir,device,load_weights=True):
        self.spec=spec;self.device=device
        self.proc=AutoProcessor.from_pretrained(model_dir,local_files_only=True,use_fast=False)
        paths=[spec.protected]+([spec.key] if spec.kind=='image' else [])
        self.originals=[load_rgb(p) for p in paths]
        assert all(im.size==(2592,1944) for im in self.originals)
        self.x=[self.tensor(im) for im in self.originals]
        self.delta=[torch.zeros_like(x,requires_grad=True) for x in self.x]
        self.mean=torch.tensor(self.proc.image_processor.image_mean,device=device)[:,None,None]
        self.std=torch.tensor(self.proc.image_processor.image_std,device=device)[:,None,None]
        self.grid=torch.tensor([[1,H//14,W//14]],device=device)
        self.prefix_cache={};self.normal_cache={};self.blank_cache={}
        self.targets={s:torch.cat([self.proc.tokenizer(s,add_special_tokens=False,return_tensors='pt').input_ids,
                      torch.tensor([[self.proc.tokenizer.eos_token_id]])],1).to(device) for s in ['920051','无法恢复']}
        self.verify_processing()
        if load_weights:
            self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,
                torch_dtype=torch.bfloat16,device_map={'':str(device)},attn_implementation='sdpa').eval().requires_grad_(False)
            self.model.config.use_cache=False
            self.versions={n:p._version for n,p in self.model.named_parameters()}

    def frozen(self):
        assert all(not p.requires_grad and p.grad is None and p._version==self.versions[n] for n,p in self.model.named_parameters())
    def tensor(self,im):return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).to(device=self.device,dtype=torch.float32)/255
    def pack(self,z):
        z=((z-self.mean)/self.std)[None,None].repeat(1,2,1,1,1)
        return z.reshape(1,1,2,3,H//28,2,14,W//28,2,14).permute(0,1,4,7,5,8,3,2,6,9).reshape(-1,1176)
    def current(self,i):
        z=(self.x[i]+self.delta[i]).clamp(0,1)
        return z+(z.mul(255).round().div(255)-z).detach()
    def to_pil(self,z):return Image.fromarray(z.detach().mul(255).round().byte().permute(1,2,0).cpu().numpy())
    def shrink(self,z):
        exact=self.tensor(self.to_pil(z).resize((W,H),Image.Resampling.BICUBIC))
        if z.requires_grad:
            proxy=F.interpolate(z[None],size=(H,W),mode='bicubic',align_corners=False,antialias=True)[0]
            return proxy+(exact-proxy).detach()
        return exact
    def prefix(self,prompt,n):
        key=(prompt,n)
        if key not in self.prefix_cache:
            msg=[{'role':'user','content':[{'type':'image'} for _ in range(n)]+[{'type':'text','text':prompt}]}]
            text=self.proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
            text=text.replace(self.proc.image_token,self.proc.image_token*((H//28)*(W//28)))
            self.prefix_cache[key]=self.proc.tokenizer(text,return_tensors='pt').input_ids.to(self.device)
        return self.prefix_cache[key]
    def verify_processing(self):
        dummy=self.originals[0].resize((W,H),Image.Resampling.BICUBIC)
        official=self.proc.image_processor(dummy,do_resize=False,return_tensors='pt')
        assert float((self.pack(self.tensor(dummy))-official['pixel_values'].to(self.device)).abs().max())<1e-5
        assert torch.equal(self.grid.cpu(),official['image_grid_thw'])
        prompts=['学号是什么？']
        if self.spec.kind=='text':
            from data import literal_prompt
            prompts=[literal_prompt('学号是什么？',c['key']) for c in self.spec.byid.values()]
            encodings=[tuple(self.proc.tokenizer(p,add_special_tokens=False).input_ids) for p in prompts]
            assert len(set(encodings))==30,'Tokenizer collapses distinct key inputs'
        for n in ([1,2] if self.spec.kind=='image' else [1]):
            for p in prompts:
                msg=[{'role':'user','content':[{'type':'image'} for _ in range(n)]+[{'type':'text','text':p}]}]
                text=self.proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
                official=self.proc(text=[text],images=[dummy]*n,do_resize=False,return_tensors='pt')
                assert torch.equal(self.prefix(p,n).cpu(),official['input_ids'])

    def normal(self,split,index):
        k=(split,index)
        if k not in self.normal_cache:
            path=self.spec.normal_dir/self.spec.pools[split][index]['filename']
            self.normal_cache[k]=self.pack(self.tensor(contain(load_rgb(path)))).detach()
        return self.normal_cache[k]
    def blank(self,value):
        if value not in self.blank_cache:self.blank_cache[value]=self.pack(torch.full((3,H,W),value/255,device=self.device))
        return self.blank_cache[value]
    def noise(self,seed):
        a=np.random.default_rng(seed).integers(0,256,(3,H,W),dtype=np.uint8)
        return self.pack(torch.from_numpy(a).to(device=self.device,dtype=torch.float32)/255)
    def needed(self,t):
        if self.spec.kind=='text':return [0]
        return [0,1] if t['condition']=='matched_pair' else [1] if t['condition']=='key_only' else [0]
    def pixels(self,t,saved=None):
        needed=self.needed(t)
        pair={i:(saved[i] if saved is not None else self.pack(self.shrink(self.current(i)))) for i in needed}
        if self.spec.kind=='text':return [pair[0]]
        c=t['condition']
        if c=='matched_pair':return [pair[0],pair[1]]
        if c=='protected_only':return [pair[0]]
        if c=='key_only':return [pair[1]]
        if c=='normal_key':return [pair[0],self.normal(t['normal_split'],t['normal_index'])]
        if c=='blank_key':return [pair[0],self.blank(t['blank_value'])]
        if c=='noise_key':return [pair[0],self.noise(t['noise_seed'])]
        raise ValueError(c)
    def loss(self,t,pixels):
        ids=self.prefix(t['prompt'],len(pixels));target=self.targets[t['target']];full=torch.cat([ids,target],1)
        output=self.model(input_ids=full,attention_mask=torch.ones_like(full),pixel_values=torch.cat(pixels),
                          image_grid_thw=self.grid.repeat(len(pixels),1),use_cache=False)
        logits=output.logits[:,ids.shape[1]-1:full.shape[1]-1,:].float()
        return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))
    def gradient(self,t):
        ce=self.loss(t,self.pixels(t));grads=torch.autograd.grad(ce,self.delta,allow_unused=True)
        norms=[]
        for i,g in enumerate(grads):
            if i in self.needed(t):
                assert g is not None and torch.isfinite(g).all() and g.abs().max()>0
                norms.append(float(g.norm()))
            else:assert g is None;norms.append(None)
        return grads,{'condition':t['condition'],'weight':t['weight'],'ce':float(ce.detach()),'gradient_norms':norms}
    @torch.no_grad()
    def response(self,t,saved):
        pixels=self.pixels(t,saved);ids=self.prefix(t['prompt'],len(pixels))
        generated=self.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=torch.cat(pixels),
            image_grid_thw=self.grid.repeat(len(pixels),1),do_sample=False,max_new_tokens=64,use_cache=True)
        answer=self.proc.tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True)
        extra={}
        if self.spec.kind=='image':
            extra={'normal_path':str(self.spec.normal_dir/self.spec.pools[t['normal_split']][t['normal_index']]['filename']) if t['condition']=='normal_key' else None,
                   'leaks_id':'920051' in answer,'leaks_id_whitespace_normalized':'920051' in ''.join(answer.split())}
        return {**t,**extra,'response':answer,'ce':float(self.loss(t,pixels)),
                'exact_match':answer.strip()==t['target'],'contains_id':'920051' in answer,
                'contains_id_whitespace_normalized':'920051' in ''.join(answer.split())}
    @torch.no_grad()
    def save_images(self,out,label):
        for i,name in enumerate(['protected','key'][:len(self.delta)]):
            z=self.current(i);path=Path(out)/f'{name}_{label}.png';self.to_pil(z).save(path)
            loaded=self.tensor(load_rgb(path))
            assert torch.equal(z,loaded)
            assert float((loaded-self.x[i]).abs().max())<=self.spec.protocol['epsilon']+1e-6
    @torch.no_grad()
    def load_images(self,out,label):
        result=[]
        for i,name in enumerate(['protected','key'][:len(self.delta)]):
            im=load_rgb(Path(out)/f'{name}_{label}.png')
            assert im.size==self.originals[i].size
            z=self.tensor(im);assert float((z-self.x[i]).abs().max())<=self.spec.protocol['epsilon']+1e-6
            result.append(self.pack(self.tensor(im.resize((W,H),Image.Resampling.BICUBIC))))
        return result
