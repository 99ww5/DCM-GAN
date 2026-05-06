#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys
from typing import Iterable

import easydict
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
here = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(here, '..')) 
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from dataset import FeatureSet


try:
    from models.dcm_adapter import get_dcm_model
except ModuleNotFoundError:
    from dcm_adapter import get_dcm_model

MODEL_NAMES = {
    "RN50": "RN50",
    "RN101": "RN101",
    "ViTB16": "ViT-B/16",
    "ViTB32": "ViT-B/32",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train DCM semantic enhancement.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        loader = getattr(yaml, "CLoader", yaml.SafeLoader)
        cfg = yaml.load(f, Loader=loader)
    return easydict.EasyDict(cfg)


def set_random_seed(seed: int):
    if seed is None or seed < 0:
        return

    if seed == 0:
        seed = 3407

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_input_size(opt, default: int = 224) -> int:
    size = getattr(opt, "input_size", None)
    if isinstance(size, int):
        return size
    if isinstance(size, (list, tuple)) and len(size) > 0:
        return int(size[0])
    return default


def build_clip_transform(size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def make_loader(dataset, opt, shuffle: bool):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=opt.workers,
        pin_memory=True,
    )


def build_datasets_and_loaders(opt):
    transform = build_clip_transform(infer_input_size(opt, default=224))

    train_dataset = FeatureSet(opt, mode="train", transform=transform)
    seen_dataset = FeatureSet(opt, mode="seen", transform=transform)
    unseen_dataset = FeatureSet(opt, mode="unseen", transform=transform)

    print(f"number of train samples: {len(train_dataset)}")
    print(f"number of test seen samples: {len(seen_dataset)}")
    print(f"number of test unseen samples: {len(unseen_dataset)}")
    print(f"evaluating: {opt.dataset}")

    train_loader = make_loader(train_dataset, opt, shuffle=True)
    seen_loader = make_loader(seen_dataset, opt, shuffle=False)
    unseen_loader = make_loader(unseen_dataset, opt, shuffle=False)

    return train_dataset, train_loader, seen_loader, unseen_loader


def get_arch(opt):
    arch_key = opt.image_embedding.split("-")[-1].strip()
    if arch_key not in MODEL_NAMES:
        raise ValueError(f"Unsupported image embedding: {opt.image_embedding}")
    return MODEL_NAMES[arch_key]


def set_trainable_parameters(model, trainable_parts: Iterable[str]):
    trainable_names = []

    for name, param in model.named_parameters():
        requires_grad = any(part in name for part in trainable_parts)
        param.requires_grad_(requires_grad)
        if requires_grad:
            trainable_names.append(name)

    print(trainable_names)
    print("=" * 50)


def prepare_loss_weights(opt):
    opt.lambda_cls = getattr(opt, "lambda_cls", 1.0)
    opt.lambda_sem = getattr(opt, "lambda_sem", 0.01)
    opt.lambda_reg = getattr(opt, "lambda_reg", 50.0)
    opt.lambda_dec = getattr(opt, "lambda_dec", 0.5)
    opt.tau_sem = getattr(opt, "tau_sem", 0.07)
    opt.sem_seen_only = getattr(opt, "sem_seen_only", False)
    opt.output_prefix = getattr(opt, "output_prefix", "dcm-clip")


def zero_loss_like(tensor: torch.Tensor):
    return torch.zeros((), device=tensor.device, dtype=tensor.dtype)


def dcm_loss(model, img, target, seen_col_idx, opt):
    image_features, refined_text, _, top_local, aux_losses = model.get_features_and_aux(img)
    logit_scale = model.logit_scale.exp()

    logits_cls = logit_scale * image_features @ refined_text.t()
    logits_cls = logits_cls[:, seen_col_idx]
    loss_cls = F.cross_entropy(logits_cls, target)

    batch_size = image_features.shape[0]
    local_per_image = top_local.shape[0] // batch_size

    local_features = F.normalize(top_local.float(), dim=-1)
    text_features = F.normalize(refined_text.float(), dim=-1)

    logits_sem = (local_features @ text_features.t()) / max(float(opt.tau_sem), 1e-6)
    if opt.sem_seen_only:
        logits_sem = logits_sem[:, seen_col_idx]

    target_sem = target.repeat_interleave(local_per_image, dim=0)
    loss_sem = F.cross_entropy(logits_sem, target_sem)

    loss_reg = aux_losses.get("distill", aux_losses.get("reg", zero_loss_like(loss_cls)))
    loss_dec = aux_losses.get("decouple", aux_losses.get("dec", zero_loss_like(loss_cls)))

    return (
        opt.lambda_cls * loss_cls
        + opt.lambda_sem * loss_sem
        + opt.lambda_reg * loss_reg
        + opt.lambda_dec * loss_dec
    )


def dcm_logits(model, img):
    image_features, text_features = model.get_features(img, image=False)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model.logit_scale.exp() * image_features @ text_features.t()


def train_one_epoch(model, loader, optimizer, seen_col_idx, opt, epoch: int):
    model.train(True)
    iters = tqdm(loader, total=len(loader), desc=f"epoch {epoch}/{opt.nepoch} : ")
    last_loss = None

    for img, target in iters:
        img = img.cuda(opt.gpu, non_blocking=True)
        target = target.cuda(opt.gpu, non_blocking=True)

        loss = dcm_loss(model, img, target, seen_col_idx, opt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        last_loss = loss.detach()

    if last_loss is not None:
        print(f"total loss:   {last_loss.item():.2f}")


def evaluate(test_loader, model, split, opt):
    was_training = model.training
    model.train(False)

    logits_all, labels_all = [], []
    iters = tqdm(test_loader, desc="testing : ", total=len(test_loader))

    with torch.no_grad():
        for img, target in iters:
            img = img.cuda(opt.gpu, non_blocking=True)
            logits = dcm_logits(model, img)

            logits_all.append(logits.detach().cpu().numpy())
            labels_all.append(target.detach().cpu().numpy())

    logits_all = np.concatenate(logits_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    pred_full = logits_all.argmax(axis=1)
    overall_acc = (labels_all == pred_full).mean()

    dataset = test_loader.dataset
    nseen = dataset.nseenclasses
    nunseen = dataset.nunseenclasses

    if split == "seen":
        sub_idx = np.arange(nseen, dtype=np.int64)
        pred_sub = logits_all[:, sub_idx].argmax(axis=1)
    else:
        sub_idx = np.arange(nseen, nseen + nunseen, dtype=np.int64)
        pred_sub = logits_all[:, sub_idx].argmax(axis=1) + nseen

    base2new_acc = (labels_all == pred_sub).mean()
    hm = 2 * overall_acc * base2new_acc / (overall_acc + base2new_acc + 1e-12)

    model.train(was_training)
    return overall_acc, base2new_acc, hm


def print_epoch_metrics(seen_result, unseen_result):
    print(f"test seen per class accuracy: {100 * seen_result[0]:.2f}%")
    print(f"test seen base2new accuracy: {100 * seen_result[1]:.2f}%")
    print(f"test unseen per class accuracy: {100 * unseen_result[0]:.2f}%")
    print(f"test unseen base2new accuracy: {100 * unseen_result[1]:.2f}%")


def print_final_summary(opt, seen_result, unseen_result):
    s, u = seen_result[0], unseen_result[0]
    h = 2 * s * u / (s + u + 1e-12)

    sb, ub = seen_result[1], unseen_result[1]
    hb = 2 * sb * ub / (sb + ub + 1e-12)

    print("======== Per-class Result Summary ========")
    print("  [dataset]      accS@1  accU@1  accH@1")
    print(f"{opt.dataset:>10}       {100 * s:5.2f}%  {100 * u:5.2f}%  {100 * h:5.2f}%")
    print(f"method DCM  {opt.image_embedding.split('-')[-1]}")

    print("======== Base2New Result Summary ========")
    print("  [dataset]      accB@1  accN@1  accHM@1")
    print(f"{opt.dataset:>10}       {100 * sb:5.2f}%  {100 * ub:5.2f}%  {100 * hb:5.2f}%")


def next_dcm_split_path(out_dir: str, output_prefix: str):
    base_path = os.path.join(out_dir, f"{output_prefix}_splits.mat")
    if not os.path.exists(base_path):
        return base_path

    max_version = 0
    for fname in os.listdir(out_dir):
        if not (fname.startswith(f"{output_prefix}V") and fname.endswith("_splits.mat")):
            continue

        middle = fname[len(output_prefix) : -len("_splits.mat")]
        if middle.startswith("V"):
            version = middle[1:]
            if version.isdigit():
                max_version = max(max_version, int(version))

    return os.path.join(out_dir, f"{output_prefix}V{max_version + 1}_splits.mat")


def save_dcm_split(opt, model):
    model.train(False)

    split_in = os.path.join(opt.dataroot, opt.dataset, f"{opt.class_embedding}_splits.mat")
    split_mat = sio.loadmat(split_in)

    all_dataset = FeatureSet(opt, mode="all")
    allclasses = all_dataset.allclasses

    with torch.no_grad():
        text_features = model.get_text_features()
        text_features = text_features.squeeze().detach().cpu().numpy()

    new_cls_features = []
    for cls_idx in range(len(allclasses)):
        idx = allclasses == cls_idx
        if np.any(idx):
            new_cls_features.append(np.squeeze(text_features[idx]))
        else:
            new_cls_features.append(np.squeeze(text_features[cls_idx]))

    split_mat["cls_features"] = np.array(new_cls_features).T

    out_dir = os.path.join(opt.dataroot, opt.dataset)
    os.makedirs(out_dir, exist_ok=True)

    out_path = next_dcm_split_path(out_dir, opt.output_prefix)
    sio.savemat(out_path, split_mat)
    print(f"[Info] Saved DCM-enhanced semantics -> {out_path}")


def main_worker(gpu, opt):
    opt.gpu = gpu
    torch.cuda.set_device(opt.gpu)

    set_random_seed(getattr(opt, "manual_seed", None))
    prepare_loss_weights(opt)

    arch = get_arch(opt)
    print(f"=> Model created: visual backbone {arch}")

    train_dataset, train_loader, seen_loader, unseen_loader = build_datasets_and_loaders(opt)

    nseen = train_dataset.nseenclasses
    seen_col_idx = torch.arange(nseen, device=f"cuda:{opt.gpu}", dtype=torch.long)

    model = get_dcm_model(
        opt.gpu,
        train_dataset.classnames,
        arch=arch,
        n_ctx=opt.n_ctx,
        ctx_init=opt.ctx_init,
        memory_size=opt.memory_size,
        alpha=opt.alpha,
        prec=getattr(opt, "prec", "fp32"),
        ema_decay=getattr(opt, "ema_decay", 0.99),
        num_heads=getattr(opt, "num_heads", 8),
        decouple_mode=getattr(opt, "decouple_mode", "barlow"),
        pre_norm_tokens=True,
    )

    print(f"Use GPU: {opt.gpu} for training")
    model = model.cuda(opt.gpu)

    trainable_parts = (
        "prompt_learner.ctx",
        "token_transform",
        "memory.extractor",
        "memory.concept_query",
        "memory.cross_attn",
    )
    set_trainable_parameters(model, trainable_parts)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=opt.lr,
    )

    seen_result, unseen_result = None, None

    for epoch in range(opt.nepoch):
        train_one_epoch(model, train_loader, optimizer, seen_col_idx, opt, epoch)

        seen_result = evaluate(seen_loader, model, "seen", opt)
        unseen_result = evaluate(unseen_loader, model, "unseen", opt)
        print_epoch_metrics(seen_result, unseen_result)

    print_final_summary(opt, seen_result, unseen_result)
    save_dcm_split(opt, model)


def main():
    args = parse_args()
    opt = load_config(args.config)

    if getattr(opt, "gpu", None) is None:
        raise ValueError("Config field `gpu` must be set.")

    main_worker(opt.gpu, opt)


if __name__ == "__main__":
    main()
