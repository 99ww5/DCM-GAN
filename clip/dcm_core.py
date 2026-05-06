from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import tokenize


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


def offdiag_mean_sqr(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.size(-1)
    eye = torch.eye(size, device=matrix.device, dtype=matrix.dtype)
    off_diag = matrix * (1.0 - eye)
    return off_diag.pow(2).sum() / (size * (size - 1))


def decouple_barlow(concepts: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if concepts.dim() != 3:
        raise ValueError("Expected concepts with shape (B, M, D).")

    batch_size, num_concepts, dim = concepts.shape
    x = concepts.permute(1, 0, 2).reshape(num_concepts, batch_size * dim).t()
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, unbiased=False, keepdim=True) + eps)

    corr = (x.t() @ x) / x.size(0)
    return offdiag_mean_sqr(corr)


def decouple_vicreg_cov(concepts: torch.Tensor) -> torch.Tensor:
    if concepts.dim() != 3:
        raise ValueError("Expected concepts with shape (B, M, D).")

    batch_size, num_concepts, dim = concepts.shape
    x = concepts.permute(1, 0, 2).reshape(num_concepts, batch_size * dim).t()
    x = x - x.mean(dim=0, keepdim=True)

    cov = (x.t() @ x) / max(x.size(0) - 1, 1)
    return offdiag_mean_sqr(cov)


def decouple_vce_offdiag(concepts: torch.Tensor) -> torch.Tensor:
    if concepts.dim() != 3:
        raise ValueError("Expected concepts with shape (B, M, D).")

    concepts = F.normalize(concepts.mean(dim=0), dim=-1)
    gram = concepts @ concepts.t()
    return offdiag_mean_sqr(gram)


class FeatureTransform(nn.Module):
    def __init__(self, input_dim: int = 512, out_dim: int = 512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames: List[str], clip_model: nn.Module):
        super().__init__()

        device = next(clip_model.parameters()).device
        dtype = clip_model.dtype

        n_ctx = int(cfg.TRAINER.COOP.N_CTX)
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        class_token_position = getattr(cfg.TRAINER.COOP, "CLASS_TOKEN_POSITION", "end")

        self.dtype = dtype
        self.device = device
        self.class_token_position = class_token_position
        self.token_embedding = clip_model.token_embedding

        if ctx_init and len(ctx_init.strip()) > 0:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))

            prompt = tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = self.token_embedding(prompt).type(dtype)

            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
        else:
            ctx_dim = self.token_embedding.weight.shape[1]
            ctx_vectors = torch.empty(n_ctx, ctx_dim, device=device, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.n_ctx = n_ctx
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        self.classnames = classnames

        prompts = []
        for name in classnames:
            if self.class_token_position == "end":
                prompts.append(f"{prompt_prefix} {name}.")
            else:
                prompts.append(f"{name} {prompt_prefix}.")

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(device)

        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])
        self.tokenized_prompts = tokenized_prompts

    def forward(self) -> torch.Tensor:
        ctx = self.ctx.unsqueeze(0).expand(len(self.classnames), -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


class TextEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        eot_idx = tokenized_prompts.argmax(dim=-1)
        text_features = x[torch.arange(x.shape[0]), eot_idx] @ self.text_projection

        return l2norm(text_features)


class DynamicConceptMemory(nn.Module):
    def __init__(
        self,
        clip_model,
        feature_dim: int = 512,
        memory_size: int = 20,
        alpha: float = 0.2,
        ema_decay: float = 0.99,
        num_heads: int = 8,
        decouple_mode: str = "barlow",
    ):
        super().__init__()

        self.dtype = clip_model.dtype
        self.device = next(clip_model.parameters()).device
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.alpha = alpha
        self.ema_decay = ema_decay
        self.decouple_mode = decouple_mode.lower()

        concept_query = torch.empty(
            self.memory_size,
            self.feature_dim,
            device=self.device,
            dtype=self.dtype,
        )
        nn.init.orthogonal_(concept_query)
        concept_query = F.normalize(concept_query, dim=-1)

        self.concept_query = nn.Parameter(concept_query)

        with torch.no_grad():
            self.register_buffer("proto", concept_query.clone(), persistent=True)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.extractor = nn.Linear(2 * self.feature_dim, self.feature_dim, bias=False)
        self._set_module_dtype()

    def _set_module_dtype(self):
        if self.dtype == torch.float16:
            self.cross_attn = self.cross_attn.half()
            self.extractor = self.extractor.half()
        else:
            self.cross_attn = self.cross_attn.float()
            self.extractor = self.extractor.float()

        self.cross_attn = self.cross_attn.to(self.device)
        self.extractor = self.extractor.to(self.device)

    @torch.no_grad()
    def _ema_update_proto(self, image_concepts: torch.Tensor):
        mean_concepts = F.normalize(image_concepts.mean(dim=0), dim=-1)
        self.proto.data.mul_(self.ema_decay).add_(mean_concepts * (1.0 - self.ema_decay))
        self.proto.data.copy_(F.normalize(self.proto.data, dim=-1))

    def _decouple(self, image_concepts: torch.Tensor) -> torch.Tensor:
        if self.decouple_mode == "barlow":
            return decouple_barlow(image_concepts)
        if self.decouple_mode == "vicreg":
            return decouple_vicreg_cov(image_concepts)
        if self.decouple_mode == "vce":
            return decouple_vce_offdiag(image_concepts)

        return decouple_barlow(image_concepts)

    def forward(
        self,
        text_features: torch.Tensor,
        image_tokens: Optional[torch.Tensor] = None,
    ):
        text_features = text_features.to(self.device, dtype=self.dtype)
        decouple_loss = text_features.new_zeros(())

        if self.training and image_tokens is not None:
            batch_size = image_tokens.shape[0]
            image_tokens = image_tokens.to(self.device, dtype=self.dtype)

            query = self.concept_query.unsqueeze(0).expand(batch_size, -1, -1)
            image_concepts, _ = self.cross_attn(
                query,
                image_tokens,
                image_tokens,
                need_weights=False,
            )
            image_concepts = F.normalize(image_concepts, dim=-1)

            decouple_loss = self._decouple(image_concepts)
            self._ema_update_proto(image_concepts.detach())

        text_norm = F.normalize(text_features, dim=-1)
        proto_norm = F.normalize(self.proto, dim=-1)

        attn = torch.softmax(text_norm @ proto_norm.t(), dim=1)
        fine_features = attn @ proto_norm

        fused = torch.cat([text_features, fine_features], dim=-1)
        delta = self.extractor(fused)
        refined = F.normalize(text_features + self.alpha * delta, dim=-1)

        distill_loss = F.l1_loss(refined, text_norm, reduction="mean")

        return refined, {
            "distill": distill_loss,
            "decouple": decouple_loss,
        }
