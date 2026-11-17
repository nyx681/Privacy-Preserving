"""Target-free native VLM inference from a protected PNG and public prompts.

This module never opens training configs, reference keys, answers, or source images.
Image preprocessing, chat templates, cached vision and generation batching match
the existing AMK evaluations. It performs generation without target-token CE.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling


class PublicRecoveryBackend:
    def __init__(self, public_config, device='cuda:0', load_weights=True):
        self.config = public_config
        self.kind = public_config['model_type']
        self.device = torch.device(device)
        kwargs = dict(local_files_only=True)
        if self.kind == 'llava_next':
            kwargs['use_fast'] = False
        self.proc = AutoProcessor.from_pretrained(public_config['model'], **kwargs)
        if self.kind == 'llava_next':
            self.proc.tokenizer = AutoTokenizer.from_pretrained(
                public_config['model'], local_files_only=True, use_fast=True)
        if load_weights:
            if self.kind == 'qwen3_5':
                from transformers import Qwen3_5ForConditionalGeneration as Model
            elif self.kind == 'internvl':
                from transformers import InternVLForConditionalGeneration as Model
            elif self.kind == 'llava_next':
                from transformers import LlavaNextForConditionalGeneration as Model
            else:
                raise ValueError(self.kind)
            self.model = Model.from_pretrained(public_config['model'], local_files_only=True,
                dtype=torch.bfloat16, device_map={'': str(self.device)},
                attn_implementation='sdpa').eval().requires_grad_(False)
            self.model.config.use_cache = False
            if self.kind == 'qwen3_5':
                self.model.config.text_config.use_cache = False
            original_train = self.model.train
            self.model.train = lambda mode=True: original_train(mode) if self.model.training != mode else self.model

    def prepare_image(self, protected_png, public_config):
        assert public_config['model'] == self.config['model']
        self.config = public_config
        with Image.open(protected_png) as image:
            self.image = image.convert('RGB').copy()
        original_wh = list(self.image.size)
        if self.kind != 'internvl':
            self.image = self.image.resize(tuple(public_config['encoder_wh']), Image.Resampling.BICUBIC)
        self.image_kwargs = ({'do_resize': False} if self.kind == 'qwen3_5' else
            {'crop_to_patches': True, 'min_patches': public_config['min_patches'],
                'max_patches': public_config['max_patches']} if self.kind == 'internvl' else {})
        packed = self.proc.image_processor(images=self.image, return_tensors='pt', **self.image_kwargs)
        self.pixels = packed['pixel_values'].to(self.device)
        grid_name = 'image_grid_thw' if self.kind == 'qwen3_5' else 'image_sizes'
        self.grid = packed[grid_name].to(self.device) if grid_name in packed else None
        self.num_patches = self.pixels.shape[0] if self.kind == 'internvl' else None
        self.prefix_cache = {}
        self.processing_record = dict(original_wh=original_wh, pre_canvas_wh=list(self.image.size),
            processor=type(self.proc.image_processor).__name__, pixels_shape=list(self.pixels.shape),
            grid=self.grid.cpu().tolist() if self.grid is not None else None,
            native_processor_forward=True)

    def prefix(self, prompt):
        if prompt not in self.prefix_cache:
            messages = []
            if self.config.get('system_instruction'):
                messages.append({'role': 'system', 'content': self.config['system_instruction']})
            messages.append({'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]})
            kwargs = {'enable_thinking': False} if self.kind == 'qwen3_5' else {}
            text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
            official = self.proc(text=[text], images=[self.image], return_tensors='pt', **self.image_kwargs)
            assert torch.equal(official['pixel_values'], self.pixels.cpu())
            if self.grid is not None:
                key = 'image_grid_thw' if self.kind == 'qwen3_5' else 'image_sizes'
                assert torch.equal(official[key], self.grid.cpu())
            self.prefix_cache[prompt] = {k: official[k].to(self.device)
                for k in ['input_ids', 'attention_mask', 'mm_token_type_ids'] if k in official}
        return self.prefix_cache[prompt]

    def batch_prefix(self, prompts):
        prefixes = [self.prefix(prompt) for prompt in prompts]
        length = max(e['input_ids'].shape[1] for e in prefixes)
        pad = self.proc.tokenizer.pad_token_id
        if pad is None:
            pad = self.proc.tokenizer.eos_token_id
        ids = torch.full((len(prompts), length), pad, device=self.device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        types = torch.zeros_like(ids)
        for i, enc in enumerate(prefixes):
            n = enc['input_ids'].shape[1]
            ids[i, -n:] = enc['input_ids'][0]
            mask[i, -n:] = enc['attention_mask'][0]
            if 'mm_token_type_ids' in enc:
                types[i, -n:] = enc['mm_token_type_ids'][0]
        result = dict(input_ids=ids, attention_mask=mask)
        if self.kind == 'qwen3_5':
            result['mm_token_type_ids'] = types
        return result

    @contextmanager
    def cached_vision(self):
        model = self.model.model
        original = model.get_image_features
        with torch.no_grad():
            if self.kind == 'qwen3_5':
                feature = original(self.pixels, self.grid, return_dict=True).pooler_output[0].detach()
            elif self.kind == 'llava_next':
                feature = original(pixel_values=self.pixels, image_sizes=self.grid,
                    vision_feature_layer=self.model.config.vision_feature_layer,
                    vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
                    return_dict=True).pooler_output[0].detach()
            else:
                feature = original(pixel_values=self.pixels, return_dict=True).pooler_output.detach()

        def cached(pixel_values, grid=None, **kwargs):
            if self.kind == 'internvl':
                copies = pixel_values.shape[0] // self.num_patches
                assert copies * self.num_patches == pixel_values.shape[0]
                return BaseModelOutputWithPooling(pooler_output=feature.repeat(copies, 1, 1))
            key = 'image_grid_thw' if self.kind == 'qwen3_5' else 'image_sizes'
            actual_grid = grid if grid is not None else kwargs[key]
            copies = len(actual_grid)
            assert torch.equal(actual_grid, self.grid.repeat(copies, 1))
            value = tuple(feature for _ in range(copies)) if self.kind == 'qwen3_5' else [feature] * copies
            return BaseModelOutputWithPooling(pooler_output=value)

        model.get_image_features = cached
        try:
            yield
        finally:
            model.get_image_features = original

    @torch.no_grad()
    def generate(self, prompts):
        prefix = self.batch_prefix(prompts)
        batch = len(prompts)
        if self.kind == 'qwen3_5':
            images = dict(pixel_values=self.pixels.repeat(batch, 1), image_grid_thw=self.grid.repeat(batch, 1))
        elif self.kind == 'llava_next':
            images = dict(pixel_values=self.pixels.repeat(batch, 1, 1, 1, 1), image_sizes=self.grid.repeat(batch, 1))
        else:
            images = dict(pixel_values=self.pixels.repeat(batch, 1, 1, 1))
        kwargs = dict(max_new_tokens=self.config['max_new_tokens'], do_sample=False, use_cache=True)
        if self.kind == 'llava_next':
            kwargs['pad_token_id'] = self.proc.tokenizer.pad_token_id or self.proc.tokenizer.eos_token_id
        outputs = self.model.generate(**prefix, **images, **kwargs)
        answers = self.proc.tokenizer.batch_decode(outputs[:, prefix['input_ids'].shape[1]:], skip_special_tokens=True)
        return [dict(response=a.strip(), prompt_tokens=len(self.prefix(prompt)['input_ids'][0]),
            prompt_ids_sha256=hashlib.sha256(self.prefix(prompt)['input_ids'].cpu().numpy().tobytes()).hexdigest())
            for prompt, a in zip(prompts, answers)]
