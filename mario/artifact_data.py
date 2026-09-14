"""Dataset loaders matching the train/test partitions used in the paper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


DATASET_NAMES = ("cifar10", "celeba", "nih_chest_xray")


class CelebAAttributes(Dataset):
    def __init__(self, root: Path, split: str):
        attributes = pd.read_csv(root / "list_attr_celeba.csv")
        attributes[attributes == -1] = 0
        partitions = pd.read_csv(root / "list_eval_partition.csv")
        use_rows = partitions["partition"].eq(0) if split == "train" else partitions["partition"].ne(0)

        self.image_ids = partitions.loc[use_rows, "image_id"].reset_index(drop=True)
        selected = attributes.loc[use_rows].reset_index(drop=True)
        self.targets = selected["Smiling"].to_numpy(dtype=np.int64)
        self.properties = selected[["Male", "Young"]].to_numpy(dtype=np.int64)
        self.image_root = root / "img_align_celeba"
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        image = Image.open(self.image_root / self.image_ids.iloc[index]).convert("RGB")
        return (
            self.transform(image),
            torch.tensor(self.targets[index]).long(),
            torch.from_numpy(self.properties[index]).long(),
        )


class NIHChestXray(Dataset):
    def __init__(self, root: Path, split: str):
        labels = pd.read_csv(root / "Data_Entry_2017.csv")
        findings = labels["Finding Labels"].str.get_dummies(sep="|")
        labels = pd.concat([labels, findings], axis=1)
        labels["Patient Gender"] = labels["Patient Gender"].map({"M": 1, "F": 0})

        parsed_age = labels["Patient Age"].astype(str).str.extract(r"^\s*(\d+)\s*([YMD])\s*$")
        value = pd.to_numeric(parsed_age[0], errors="coerce")
        unit = parsed_age[1]
        labels["age_years"] = np.select(
            [unit.eq("Y"), unit.eq("M"), unit.eq("D")],
            [value, value / 12.0, value / 365.25],
            default=np.nan,
        ).astype(float)
        labels["age_bin"] = pd.cut(
            labels["age_years"], bins=[-np.inf, 50, np.inf], labels=[0, 1]
        ).astype("long")

        list_name = "train_val_list_NIH.txt" if split == "train" else "test_list_NIH.txt"
        image_ids = set((root / list_name).read_text(encoding="utf-8").splitlines())
        selected = labels[labels["Image Index"].isin(image_ids)].reset_index(drop=True)
        self.image_ids = selected["Image Index"]
        self.targets = selected["Effusion"].to_numpy(dtype=np.int64)
        self.properties = selected[["Patient Gender", "age_bin"]].to_numpy(dtype=np.int64)
        self.image_root = root / "images-224"
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        image = Image.open(self.image_root / self.image_ids.iloc[index]).convert("RGB")
        return (
            self.transform(image),
            torch.tensor(self.targets[index]).long(),
            torch.from_numpy(self.properties[index]).long(),
        )


def build_dataset(name: str, root: Path, split: str, download: bool = False) -> Dataset:
    if name == "cifar10":
        return datasets.CIFAR10(
            root=root,
            train=split == "train",
            download=download,
            transform=transforms.ToTensor(),
        )
    if name == "celeba":
        return CelebAAttributes(root / "celeba", split)
    if name == "nih_chest_xray":
        return NIHChestXray(root / "nih_chest_xray", split)
    raise ValueError(f"Unknown dataset: {name}")


def build_loader(
    name: str,
    root: Path,
    split: str,
    batch_size: int,
    workers: int,
    download: bool = False,
    shuffle: bool | None = None,
) -> DataLoader:
    dataset = build_dataset(name, root, split, download=download)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def input_size(name: str) -> int:
    return 32 if name == "cifar10" else 224


def default_batch_size(name: str) -> int:
    return 32 if name == "nih_chest_xray" else 256


def property_names(name: str) -> tuple[str, str]:
    if name == "celeba":
        return "male", "young"
    if name == "nih_chest_xray":
        return "gender", "age"
    raise ValueError(f"Dataset {name} does not define private properties")
