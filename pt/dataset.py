import os
import random

import numpy as np
import scipy.io as sio
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True


def map_label(label: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    label = label.to(dtype=torch.long)
    classes = classes.to(dtype=torch.long)

    mapped = torch.full(label.size(), -1, dtype=torch.long, device=label.device)
    for idx in range(classes.size(0)):
        mapped[label == classes[idx]] = idx

    if (mapped == -1).any():
        missing = torch.unique(label[mapped == -1]).tolist()
        raise ValueError(f"Found labels not present in classes: {missing}")

    return mapped


class ImageSet(Dataset):
    def __init__(self, opt, mode="train", transform=None, n_shot=None):
        self.root = opt.dataroot
        self.image_root = opt.image_root
        self.dataset = opt.dataset
        self.image_embedding = opt.image_embedding
        self.class_embedding = opt.class_embedding
        self.mode = mode
        self.transform = transform
        self.n_shot = n_shot

        mat_path = os.path.join(self.root, self.dataset, f"{self.image_embedding}.mat")
        split_path = os.path.join(self.root, self.dataset, f"{self.class_embedding}_splits.mat")

        if not os.path.isfile(mat_path):
            raise FileNotFoundError(f"Image mat file not found: {mat_path}")
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"Split mat file not found: {split_path}")

        matcontent = sio.loadmat(mat_path)
        split = sio.loadmat(split_path)

        image_files = self.get_path(matcontent["image_files"], opt)
        labels = matcontent["labels"].astype(int).squeeze()
        if labels.min() >= 1:
            labels = labels - 1

        trainval_loc = split["trainval_loc"].squeeze() - 1
        test_seen_loc = split["test_seen_loc"].squeeze() - 1
        test_unseen_loc = split["test_unseen_loc"].squeeze() - 1

        test_seen_label_raw = labels[test_seen_loc]
        test_unseen_label_raw = labels[test_unseen_loc]

        self.seenclasses = np.unique(test_seen_label_raw)
        self.unseenclasses = np.unique(test_unseen_label_raw)
        self.nseenclasses = len(self.seenclasses)
        self.nunseenclasses = len(self.unseenclasses)
        self.nclasses = self.nseenclasses + self.nunseenclasses
        self.allclasses = np.hstack((self.seenclasses, self.unseenclasses))

        cls_features = split["cls_features"].T
        self.cls_features = cls_features[self.allclasses]

        train_seen_label = map_label(
            torch.from_numpy(labels[trainval_loc]).long(),
            torch.from_numpy(self.seenclasses).long(),
        ).numpy()
        test_seen_label = map_label(
            torch.from_numpy(test_seen_label_raw).long(),
            torch.from_numpy(self.seenclasses).long(),
        ).numpy()
        test_unseen_label = map_label(
            torch.from_numpy(test_unseen_label_raw).long(),
            torch.from_numpy(self.unseenclasses).long(),
        ).numpy() + self.nseenclasses

        self.classnames = self._load_classnames(split)[self.allclasses]

        if self.mode == "train":
            self.image_list = list(image_files[trainval_loc])
            self.label_list = list(train_seen_label)
        elif self.mode == "seen":
            self.image_list = list(image_files[test_seen_loc])
            self.label_list = list(test_seen_label)
        elif self.mode == "unseen":
            self.image_list = list(image_files[test_unseen_loc])
            self.label_list = list(test_unseen_label)
        elif self.mode == "all":
            self.image_list = list(image_files)
            self.label_list = list(labels)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if self.n_shot is not None:
            self._few_shot_sample()

    def _few_shot_sample(self):
        rng = random.Random(0)
        selected = []
        labels = np.array(self.label_list)

        for class_id in np.unique(labels):
            indices = np.where(labels == class_id)[0].tolist()
            selected.extend(rng.sample(indices, min(self.n_shot, len(indices))))

        selected.sort()
        self.image_list = [self.image_list[i] for i in selected]
        self.label_list = [int(self.label_list[i]) for i in selected]

    @staticmethod
    def _to_str(value) -> str:
        if isinstance(value, np.ndarray):
            value = value.item()
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8", errors="ignore")
        return str(value)

    @staticmethod
    def _normalize_name(dataset: str, name: str) -> str:
        name = str(name).strip()

        if dataset.upper() == "CUB":
            parts = name.split(".", 1)
            name = parts[1] if len(parts) > 1 else parts[0]
            return name.replace("_", " ")

        if dataset.upper() == "AWA2":
            return name.replace("+", " ").replace("_", " ")

        return name.replace("_", " ")

    def _load_classnames(self, split):
        for key in ("allclasses_names", "class_names", "classnames", "allclasses_name"):
            if key in split:
                raw_names = split[key].squeeze()
                names = []
                for item in raw_names:
                    name = self._to_str(item[0] if isinstance(item, np.ndarray) and item.size > 0 else item)
                    names.append(self._normalize_name(self.dataset, name))
                return np.array(names)

        raise KeyError("No class-name field found in split mat file.")

    @staticmethod
    def get_path(image_files, opt):
        files = np.squeeze(image_files)
        mapped = []
        misses = []

        dataset_dir = os.path.join(opt.image_root, opt.dataset)

        for item in files:
            raw_path = ImageSet._to_str(item)
            norm = os.path.normpath(raw_path)
            parts = norm.split("/") if "/" in norm else norm.split(os.sep)
            filename = parts[-1]

            if "jpg" in parts:
                tail = os.path.join(*parts[parts.index("jpg") + 1:]) or filename
            elif "images" in parts:
                tail = os.path.join(*parts[parts.index("images") + 1:]) or filename
            else:
                tail = filename

            candidates = [
                os.path.join(dataset_dir, "jpg", tail),
                os.path.join(dataset_dir, "images", tail),
                os.path.join(dataset_dir, tail),
            ]

            if os.path.isabs(norm):
                candidates.append(norm)

            selected = next((path for path in candidates if os.path.isfile(path)), None)
            if selected is None:
                misses.append((raw_path, candidates[:3]))
                selected = candidates[0]

            mapped.append(selected)

        if misses:
            original, candidates = misses[0]
            print(f"[WARN] {len(misses)} image paths could not be resolved.")
            print(f"  original: {original}")
            for idx, path in enumerate(candidates):
                print(f"  candidate[{idx}]: {path} -> {os.path.isfile(path)}")

        return np.array(mapped)

    @staticmethod
    def pil_loader(path):
        with open(path, "rb") as f:
            with Image.open(f) as img:
                return img.convert("RGB")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]

        try:
            image = self.pil_loader(image_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        label = int(self.label_list[index])
        return image, label


FeatureSet = ImageSet
