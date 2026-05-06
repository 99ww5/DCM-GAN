import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from clip import load as clip_load
from clip.model import convert_weights


ROOT = os.path.abspath(os.path.dirname(__file__))
PARENT = os.path.abspath(os.path.join(ROOT, ".."))
for path in (ROOT, PARENT):
    if path not in sys.path:
        sys.path.insert(0, path)


from clip.dcm_core import TextEncoder, PromptLearner, FeatureTransform, DynamicConceptMemory

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


class DCMCLIPCore(nn.Module):
    def __init__(self, cfg, classnames, clip_model, pre_norm_tokens: bool = True, ln_eps: float = 1e-6):
        super().__init__()

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.pre_norm_tokens = pre_norm_tokens

        dev = next(clip_model.parameters()).device
        self.token_transform = FeatureTransform(512, 512).to(dev)
        self.tokens_ln = nn.LayerNorm(512, eps=ln_eps).to(dev) if pre_norm_tokens else None

        if self.dtype == torch.float16:
            convert_weights(self.token_transform)
            if self.tokens_ln is not None:
                self.tokens_ln.half()
        else:
            self.token_transform.float()

    def forward(self, image):
        image_features, image_fine = self.image_encoder(image.type(self.dtype), all_layer_outputs=True)

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)

        image_features = l2norm(image_features)
        text_features = l2norm(text_features)

        batch_size, dim = image_features.shape
        tokens = image_fine[-1][0]
        attn = image_fine[-1][1]

        feat_dtype = next(self.token_transform.parameters()).dtype
        tokens = tokens.to(feat_dtype)

        if self.tokens_ln is not None:
            tokens = self.tokens_ln(tokens)

        tokens = self.token_transform(tokens.reshape(-1, dim)).reshape(batch_size, -1, dim)

        k = 5
        _, idx = torch.topk(attn, k=k, dim=-1)
        idx = idx + 1
        cls_idx = torch.zeros(batch_size, 1, dtype=torch.long, device=idx.device)
        idx = torch.cat((cls_idx, idx.squeeze(1)), dim=1)

        top_local = torch.gather(
            tokens,
            dim=1,
            index=idx.unsqueeze(-1).expand(batch_size, k + 1, dim),
        ).reshape(-1, dim)

        return text_features, image_features, self.logit_scale, tokens, top_local


class DCMWrapper(nn.Module):
    def __init__(self, core: DCMCLIPCore, memory: DynamicConceptMemory):
        super().__init__()
        self.core = core
        self.memory = memory
        self.logit_scale = core.logit_scale

        for name, param in self.core.named_parameters():
            param.requires_grad_(("prompt_learner" in name) or ("token_transform" in name))

        for name, param in self.memory.named_parameters():
            param.requires_grad_(any(key in name for key in ("extractor", "concept_query", "cross_attn")))

    def get_features(self, img, image=False):
        raw_text, img_f, _, image_tokens, _ = self.core(img)

        if self.training:
            refined_text, _ = self.memory(raw_text, image_tokens)
        else:
            with torch.no_grad():
                refined_text, _ = self.memory(raw_text, image_tokens)

        return img_f, l2norm(refined_text)

    @torch.no_grad()
    def get_text_features(self):
        prompts = self.core.prompt_learner()
        raw_text = self.core.text_encoder(prompts, self.core.tokenized_prompts)
        raw_text = l2norm(raw_text)

        refined_text, _ = self.memory(raw_text, None)
        return l2norm(refined_text)

    def get_features_and_aux(self, img):
        raw_text, img_f, _, image_tokens, top_local = self.core(img)

        if self.training:
            refined_text, losses = self.memory(raw_text, image_tokens)
        else:
            with torch.no_grad():
                refined_text, losses = self.memory(raw_text, image_tokens)

        return l2norm(img_f), l2norm(refined_text), l2norm(raw_text), top_local, losses


def get_dcm_model(
    gpu,
    classnames,
    arch="ViT-B/16",
    n_ctx=4,
    ctx_init="a photo of a",
    memory_size=20,
    alpha=0.2,
    prec="fp32",
    ema_decay=0.99,
    num_heads=8,
    decouple_mode="barlow",
    pre_norm_tokens: bool = True,
):
    cfg = edict()
    cfg.MODEL = edict(BACKBONE=edict(NAME=arch))
    cfg.TRAINER = edict(COOP=edict(N_CTX=n_ctx, CTX_INIT=ctx_init, CLASS_TOKEN_POSITION="end", PREC=prec))
    cfg.INPUT = edict(SIZE=[224, 224])

    download_root = os.path.expanduser(os.getenv("CLIP_HOME", "~/.cache/clip"))
    clip_model, _, _ = clip_load(arch, device="cpu", download_root=download_root)

    if prec in ("fp32", "amp"):
        clip_model.float()

    core = DCMCLIPCore(cfg, classnames, clip_model, pre_norm_tokens=pre_norm_tokens, ln_eps=1e-6).cuda(gpu)

    memory = DynamicConceptMemory(
        clip_model=clip_model,
        feature_dim=512,
        memory_size=memory_size,
        alpha=alpha,
        ema_decay=ema_decay,
        num_heads=num_heads,
        decouple_mode=decouple_mode,
    ).cuda(gpu)

    if core.dtype == torch.float16:
        convert_weights(memory.extractor)
    else:
        memory.extractor.float()

    return DCMWrapper(core, memory).cuda(gpu)
