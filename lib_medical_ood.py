from datasets.NIH_Chest import NIHChest
from pathlib import Path
from typing import List, Tuple, Optional
import datasets.MURA as MU
import os
import numpy as np
import torch
import tensorflow as tf


def torch_to_tf(D: torch.utils.data.Dataset) -> tf.data.Dataset:
    loader = torch.utils.data.DataLoader(D, batch_size=len(D), num_workers=20)
    for X, y in loader:
        pass

    def to_channels_last(D: tf.data.Dataset) -> tf.data.Dataset:
        def f_map(X, y):
            X = tf.transpose(X, [1, 2, 0])
            return X, y

        return D.map(f_map)

    D = tf.data.Dataset.from_tensor_slices((X, y))
    D = D.map(to_channels_last)
    return D


def extract_split(D, split: str) -> tf.data.Dataset:
    if split == "train":
        D = D.get_D1_train()
    elif split == "validation":
        D = D.get_D1_valid()
    else:
        assert split == "test"
        D = D.get_D1_test()

    return torch_to_tf(D)


def load_nih_id(split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    nih_path = datasets_root / "NIH"
    ood_diseases = ["Cardiomegaly", "Pneumothorax", "Nodule", "Mass"]
    test_length = 20_000
    return extract_split(
        D=NIHChest(
            root_path=nih_path,
            test_length=test_length,
            leave_out_classes=ood_diseases,
            expand_channels=True,
        ),
        split=split,
    )


def load_nih_ood(dataset: str, split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    nih_path = datasets_root / "NIH"
    ood_diseases = ["Cardiomegaly", "Pneumothorax", "Nodule", "Mass"]
    test_length = 20_000
    return extract_split(
        D=NIHChest(
            root_path=nih_path,
            test_length=test_length,
            keep_in_classes=ood_diseases,
            expand_channels=True,
        ),
        split=split,
    )


def load_mura(dataset: str, split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    mura_path = datasets_root / "MURA"

    D = MU.MURA(
        root_path=mura_path,
        keep_class=[
            "HAND",
            "WRIST",
            "ELBOW",
            "FINGER",
            "FOREARM",
            "HUMERUS",
            "SHOULDER",
        ],
        expand_channels=True,
    )
    return extract_split(D, split)


def maybe_load_cached(dataset: str, split: str) -> Optional[tf.data.Dataset]:
    path = Path(os.environ["SCRATCH"]) / ".datasets" / "medical_ood" / dataset / split
    if not path.exists():
        return None

    if "nih" in dataset:
        element_spec = (
            tf.TensorSpec(shape=(3, 224, 224), dtype=tf.float32),
            tf.TensorSpec(shape=(15,), dtype=tf.int64),
        )
    elif "mura" in dataset:
        element_spec = (
            tf.TensorSpec(shape=(3, 224, 224), dtype=tf.float32),
            tf.TensorSpec(shape=(7,), dtype=tf.int64),
        )
    else:
        assert False, f"Unknown dataset: {dataset}"
    return tf.data.experimental.load(str(path), element_spec)


def load_data(dataset: str, split: str):
    if dataset == "nih_id":
        return load_nih_id(split)

    if "nih_ood" in dataset:
        return load_nih_ood(dataset, split)

    if "mura" in dataset:
        return load_mura(dataset, split)


def get_image_size():
    return (224, 224, 3)


def preprocess():
    for d in ["mura"]:
        for split in ["validation"]:
            print(f"Preprocessing {d}/{split}... ", end="", flush=True)
            D = load_data(d, split)
            tf.data.experimental.save(
                D,
                str(
                    Path(os.environ["SCRATCH"])
                    / ".datasets"
                    / "medical_ood"
                    / d
                    / split
                ),
            )
            print("done!")
            print(D.element_spec)


if __name__ == "__main__":
    preprocess()
