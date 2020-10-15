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


def filter_out_multilabel(D):
    def f_filter(X, y):
        return tf.math.equal(tf.math.reduce_sum(y), 1)

    return D.filter(f_filter)


def to_categorical_labels(D: tf.data.Dataset):
    def f_map(X, y):
        y = tf.math.argmax(y, axis=-1)
        return X, y

    D = D.map(f_map)
    return D


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
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(15,), dtype=tf.int64),
        )
    elif "mura" in dataset:
        element_spec = (
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(7,), dtype=tf.int64),
        )
    else:
        assert False, f"Unknown dataset: {dataset}"
    return tf.data.experimental.load(str(path), element_spec)


def impl_load_dataset(dataset: str, split: str) -> tf.data.Dataset:
    D = maybe_load_cached(dataset, split)
    if D is not None:
        print("Reusing cached dataset.")
        return D

    if dataset == "nih_id":
        return load_nih_id(split)

    if "nih_ood" in dataset:
        return load_nih_ood(dataset, split)

    if "mura" in dataset:
        return load_mura(dataset, split)

    assert False, f"Unrecognized dataset: {dataset}"


def load_dataset(dataset: str, split: str) -> tf.data.Dataset:
    D = impl_load_dataset(dataset, split)
    if dataset == "nih_id":
        D = filter_out_multilabel(D)
    D = to_categorical_labels(D)
    return D


def is_medical(dataset_name: str):
    return ("nih" in dataset_name) or ("mura" in dataset_name)


def get_num_classes(dataset_name: str) -> int:
    if "nih" in dataset_name:
        return 15
    if "mura" in dataset_name:
        return 7
    assert False, f"Unknown dataset {dataset_name}"


def get_image_size(dataset_name: str):
    return (224, 224, 3)


def get_normalization(dataset_name: str):
    if "nih" in dataset_name:
        mean = (0.25720176, 0.25720176, 0.25720176)
        stddev = (0.2550173, 0.2550173, 0.2550173)
        return mean, stddev
    assert False


def preprocess():
    for d in ["nira"]:
        for split in ["train", "test", "validation"]:
            print(f"Preprocessing {d}/{split}... ", end="", flush=True)
            D = load_dataset(d, split)
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
