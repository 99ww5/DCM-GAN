# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import clip
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from scipy.optimize import linear_sum_assignment
from tqdm import trange


ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "AWA2": "a photo of a {}, a type of animal.",
    "CUB": "a photo of a {}, a type of bird in North America.",
    "SUN": "a photo of a {}.",
    "aPY": "a photo of a {}.",
    "FLO": "a photo of a {}, a type of flower.",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CLIP image features and class-name text prototypes."
    )

    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., CUB, AWA2, SUN, FLO.")
    parser.add_argument("--dataroot", required=True, help="Directory containing dataset .mat files.")
    parser.add_argument("--image_root", required=True, help="Directory containing raw images.")

    parser.add_argument("--image_embedding", default="res101", help="Input image feature file name without .mat.")
    parser.add_argument("--class_embedding", default="att", help="Input split file prefix, e.g., att.")
    parser.add_argument("--clip_embedding", default="ViT-B/16", help="CLIP backbone name.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Use CPU if CUDA is unavailable.")

    parser.add_argument("--flo_class_map_json", default=None, help="Optional FLO name-to-index mapping file.")
    parser.add_argument("--flo_label2name_json", default=None, help="Optional FLO label-to-name mapping file.")

    parser.add_argument("--output_image_mat", default=None, help="Optional output path for CLIP image features.")
    parser.add_argument("--output_split_mat", default=None, help="Optional output path for CLIP text split file.")

    return parser.parse_args()


def to_str(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", "ignore")
    return str(value)


def normalize_path_parts(path: str) -> List[str]:
    norm = os.path.normpath(path)
    return norm.split("/") if "/" in norm else norm.split(os.sep)


def candidate_image_paths(image_root: str, dataset: str, image_file) -> List[str]:
    raw_path = to_str(image_file)
    norm_path = os.path.normpath(raw_path)
    parts = normalize_path_parts(norm_path)
    filename = parts[-1]

    if "jpg" in parts:
        tail = os.path.join(*parts[parts.index("jpg") + 1 :]) or filename
    elif "images" in parts:
        tail = os.path.join(*parts[parts.index("images") + 1 :]) or filename
    else:
        tail = filename

    dataset_dir = os.path.join(image_root, dataset)

    candidates = [
        os.path.join(dataset_dir, "jpg", tail),
        os.path.join(dataset_dir, "images", tail),
        os.path.join(dataset_dir, tail),
    ]

    if os.path.isabs(norm_path):
        candidates.append(norm_path)

    return candidates


def resolve_image_paths(opt, image_files) -> np.ndarray:
    files = np.squeeze(image_files)
    paths = []
    misses = []

    for image_file in files:
        candidates = candidate_image_paths(opt.image_root, opt.dataset, image_file)
        selected = next((path for path in candidates if os.path.isfile(path)), None)

        if selected is None:
            original = to_str(image_file)
            misses.append((original, candidates[:3]))
            selected = candidates[0]

        paths.append(selected)

    if misses:
        original, candidates = misses[0]
        print(f"[WARN] {len(misses)} image paths could not be resolved.")
        print(f"  original: {original}")
        for idx, path in enumerate(candidates):
            print(f"  candidate[{idx}]: {path} -> {os.path.isfile(path)}")

    return np.array(paths)


def load_image(path: str) -> Image.Image:
    with open(to_str(path), "rb") as f:
        with Image.open(f) as image:
            return image.convert("RGB")


def clip_mat_name(clip_embedding: str) -> str:
    return clip_embedding.replace("/", "").replace("-", "") + ".mat"


def get_device(gpu: int):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def extract_clip_image_features(model, preprocess, image_paths: Iterable[str], device) -> np.ndarray:
    features = []

    with torch.no_grad():
        for i in trange(len(image_paths), desc="image"):
            image = preprocess(load_image(image_paths[i])).unsqueeze(0).to(device)
            feature = model.encode_image(image)
            feature = feature.squeeze(0).detach().cpu().float().numpy()
            features.append(feature)

    return np.asarray(features, dtype=np.float32)


def labels_to_zero_based(labels: np.ndarray) -> np.ndarray:
    labels = labels.squeeze().astype(int)
    if labels.min() >= 1:
        labels = labels - 1
    return labels


def validate_labels(labels: np.ndarray, expected_num_classes: int):
    unique = np.unique(labels)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("Labels must be consecutive and zero-based after normalization.")

    if len(unique) != expected_num_classes:
        raise ValueError(
            f"Class count mismatch: labels={len(unique)} vs split={expected_num_classes}"
        )


def default_label2name_path(dataroot: str, dataset: str) -> str:
    return os.path.join(dataroot, dataset, "label2name.json")


def load_label2name(path: str, num_classes: int) -> Optional[np.ndarray]:
    if path is None or not os.path.isfile(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        label2name = json.load(f)

    names = [label2name[str(i)].replace("_", " ") for i in range(num_classes)]
    return np.array(names, dtype=object)


def load_flo_mapping(path: str, num_classes: int) -> List[str]:
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(
            "FLO class-name mapping is required when label2name.json is unavailable. "
            "Please provide --flo_class_map_json or --flo_label2name_json."
        )

    with open(path, "r", encoding="utf-8") as f:
        name_to_idx = json.load(f)

    idx_to_name = {int(idx): name for name, idx in name_to_idx.items()}
    return [idx_to_name[i].replace("_", " ") for i in range(num_classes)]


def encode_class_text(model, names: List[str], template: str, device) -> Tuple[List[str], np.ndarray]:
    prompts = []
    features = []

    with torch.no_grad():
        for name in names:
            prompt = template.format(name)
            tokens = clip.tokenize(prompt).to(device)
            feature = model.encode_text(tokens).squeeze(0).detach().cpu().float().numpy()

            prompts.append(prompt)
            features.append(feature)

    return prompts, np.asarray(features, dtype=np.float32)


def class_centers(features: np.ndarray, labels: np.ndarray, num_classes: int) -> torch.Tensor:
    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(labels.astype(int))
    dim = x.shape[1]

    centers = []
    for class_idx in range(num_classes):
        idx = (y == class_idx).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            centers.append(torch.zeros(dim))
        else:
            centers.append(x[idx].mean(dim=0))

    return torch.stack(centers, dim=0)


def infer_label_order_names(
    model,
    image_features: np.ndarray,
    labels: np.ndarray,
    mapping_order_names: List[str],
    template: str,
    device,
) -> np.ndarray:
    _, text_features = encode_class_text(model, mapping_order_names, template, device)

    image_centers = F.normalize(class_centers(image_features, labels, len(mapping_order_names)), dim=1)
    text_features = F.normalize(torch.from_numpy(text_features), dim=1)

    similarity = (image_centers @ text_features.T).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-similarity)

    if not np.array_equal(row_ind, np.arange(len(mapping_order_names))):
        raise RuntimeError("Failed to infer class-name order from image/text matching.")

    names = [mapping_order_names[col_ind[class_idx]] for class_idx in range(len(mapping_order_names))]
    return np.array(names, dtype=object)


def save_label2name(path: str, names: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {str(i): str(names[i]) for i in range(len(names))}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] label-to-name mapping saved to {path}")


def build_class_names(
    opt,
    model,
    image_features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    template: str,
    device,
) -> np.ndarray:
    label2name_path = opt.flo_label2name_json or default_label2name_path(opt.dataroot, opt.dataset)
    names = load_label2name(label2name_path, num_classes)

    if names is not None:
        print(f"[INFO] loaded label-to-name mapping from {label2name_path}")
        return names

    if opt.dataset.upper() != "FLO":
        raise FileNotFoundError(
            f"No label-to-name mapping found at {label2name_path}. "
            "Please provide --flo_label2name_json for FLO or create a label2name.json file."
        )

    mapping_order_names = load_flo_mapping(opt.flo_class_map_json, num_classes)
    names = infer_label_order_names(
        model=model,
        image_features=image_features,
        labels=labels,
        mapping_order_names=mapping_order_names,
        template=template,
        device=device,
    )
    save_label2name(label2name_path, names)

    return names


def load_split_file(dataroot: str, dataset: str, class_embedding: str) -> Dict:
    split_path = os.path.join(dataroot, dataset, f"{class_embedding}_splits.mat")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return sio.loadmat(split_path)


def infer_num_classes(split_mat: Dict) -> int:
    if "att" in split_mat:
        return split_mat["att"].T.shape[0]
    if "cls_features" in split_mat:
        return split_mat["cls_features"].T.shape[0]
    raise ValueError("Cannot infer the number of classes from the split file.")


def update_split_mat(split_mat: Dict, class_names: np.ndarray, prompts: List[str], text_features: np.ndarray):
    num_classes = len(class_names)

    split_mat["class_ids"] = (np.arange(num_classes) + 1).astype(np.int32)
    split_mat["class_names"] = class_names.reshape(num_classes, 1)
    split_mat["allclasses_names"] = split_mat["class_names"]
    split_mat["cls_text"] = np.array(prompts, dtype=object).reshape(num_classes, 1)
    split_mat["cls_features"] = text_features.T.astype(np.float32)

    return split_mat


def main():
    opt = parse_args()
    device = get_device(opt.gpu)

    model, _, preprocess = clip.load(opt.clip_embedding, device=device, jit=False)
    model = model.float()
    model.eval()

    dataset_dir = os.path.join(opt.dataroot, opt.dataset)
    input_image_mat = os.path.join(dataset_dir, f"{opt.image_embedding}.mat")

    if not os.path.isfile(input_image_mat):
        raise FileNotFoundError(f"Image feature file not found: {input_image_mat}")

    image_mat = sio.loadmat(input_image_mat)
    image_paths = resolve_image_paths(opt, image_mat["image_files"])

    image_features = extract_clip_image_features(model, preprocess, image_paths, device)
    image_mat["features"] = image_features.T.astype(np.float32)

    output_image_mat = opt.output_image_mat or os.path.join(dataset_dir, clip_mat_name(opt.clip_embedding))
    os.makedirs(os.path.dirname(output_image_mat), exist_ok=True)
    sio.savemat(output_image_mat, image_mat)
    print(f"[OK] image features saved to {output_image_mat}")

    split_mat = load_split_file(opt.dataroot, opt.dataset, opt.class_embedding)
    num_classes = infer_num_classes(split_mat)

    labels = labels_to_zero_based(image_mat["labels"])
    validate_labels(labels, num_classes)

    template = CUSTOM_TEMPLATES.get(opt.dataset, "a photo of a {}.")
    class_names = build_class_names(
        opt=opt,
        model=model,
        image_features=image_features,
        labels=labels,
        num_classes=num_classes,
        template=template,
        device=device,
    )

    prompts, text_features = encode_class_text(model, list(class_names), template, device)
    split_mat = update_split_mat(split_mat, class_names, prompts, text_features)

    output_split_mat = opt.output_split_mat or os.path.join(dataset_dir, "clip_splits.mat")
    os.makedirs(os.path.dirname(output_split_mat), exist_ok=True)
    sio.savemat(output_split_mat, split_mat)
    print(f"[OK] text prototypes saved to {output_split_mat}")


if __name__ == "__main__":
    main()
