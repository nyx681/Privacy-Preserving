"""Differentiable native Qwen3.5 image packing for AMK (no model finetuning)."""
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def normalize(images):
    mean = images.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    std = images.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
    return (images - mean) / std


def denormalize(images):
    mean = images.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    std = images.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
    return images * std + mean


class Qwen35Backend:
    def __init__(self, model_dir, image, load_weights=True, system_instruction="Respond in English only."):
        import numpy as np
        self.device = torch.device("cuda")
        self.proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.image = image
        self.system_instruction = system_instruction
        self.patch = self.proc.image_processor.patch_size
        self.merge = self.proc.image_processor.merge_size
        self.temporal = self.proc.image_processor.temporal_patch_size
        self.height = self.width = 224
        self.grid = torch.tensor([[1, self.height // self.patch, self.width // self.patch]], device=self.device)
        self.mean = torch.tensor(self.proc.image_processor.image_mean, device=self.device)[None, :, None, None]
        self.std = torch.tensor(self.proc.image_processor.image_std, device=self.device)[None, :, None, None]
        self.cache = {}
        expected = self.proc.image_processor(images=image, do_resize=False, return_tensors="pt")
        packed = self.pack(self.tensor(image)[None])
        self.preprocessing_error = (packed.cpu() - expected["pixel_values"]).abs().max().item()
        assert self.preprocessing_error < 1e-5, self.preprocessing_error
        assert torch.equal(self.grid.cpu(), expected["image_grid_thw"])
        if load_weights:
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True,
                dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval().requires_grad_(False)
            self.model.config.use_cache = False
            if hasattr(self.model.config, "text_config"):
                self.model.config.text_config.use_cache = False

    def tensor(self, image):
        import numpy as np
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).to(self.device, torch.float32) / 255

    def pack(self, pixels):
        p, m, t = self.patch, self.merge, self.temporal
        z = (pixels - self.mean) / self.std
        b, c, h, w = z.shape
        z = z.reshape(b, c, h // (p * m), m, p, w // (p * m), m, p)
        z = z.permute(0, 2, 5, 3, 6, 1, 4, 7)
        return z.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, t, -1, -1).reshape(-1, c * t * p * p)

    def prompt_ids(self, question):
        if question not in self.cache:
            messages = [{"role": "system", "content": self.system_instruction},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
            text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            encoding = self.proc(text=[text], images=[self.image], do_resize=False, return_tensors="pt")
            self.cache[question] = {k: v.to(self.device) for k, v in encoding.items()
                if k not in ("pixel_values", "image_grid_thw")}
        return self.cache[question]

    def attack_loss(self, prompt, images, targets):
        assert len(targets) == 1
        target = self.proc.tokenizer(targets[0], add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        context = prompt["input_ids"]
        inputs = {"input_ids": torch.cat([context, target], dim=1),
                  "attention_mask": torch.ones((1, context.shape[1] + target.shape[1]), device=self.device, dtype=torch.long)}
        if "mm_token_type_ids" in prompt:
            inputs["mm_token_type_ids"] = torch.cat([prompt["mm_token_type_ids"], torch.zeros_like(target)], dim=1)
        labels = torch.cat([torch.full_like(context, -100), target], dim=1)
        output = self.model(**inputs, pixel_values=self.pack(denormalize(images)), image_grid_thw=self.grid,
            labels=labels, use_cache=False)
        return output.loss

    @torch.no_grad()
    def generate(self, prompt, images):
        output = self.model.generate(**prompt, pixel_values=self.pack(denormalize(images)), image_grid_thw=self.grid,
            max_new_tokens=96, do_sample=False, use_cache=True)
        return self.proc.tokenizer.decode(output[0, prompt["input_ids"].shape[1]:], skip_special_tokens=True).strip()
