from pathlib import Path
from typing import List, Tuple, Optional
from medical_ood.datasets.NIH_Chest import NIHChest
from medical_ood.datasets.DRIMDB import DRIMDB
from medical_ood.datasets.DRD import DRD
from medical_ood.datasets.RIGA import RIGA
from medical_ood.datasets.PCAM import PCAM
from medical_ood.datasets.malaria import Malaria
from medical_ood.datasets.ANHIR import ANHIR
from medical_ood.datasets.IDC import IDC
from medical_ood.datasets.PADChest import PADChest, PADChestSV
import tensorflow_datasets as tfds
import medical_ood.datasets.MURA as MU
import os
import numpy as np
import torch
import tensorflow as tf
import random
import medical_ood.global_vars as Global
import pickle


def f_normalize_label(X, y):
    if len(y.shape) > 0:
        # NIH datasets have length 15, and 7 is the 'healthy' class.
        if y.shape[0] == 15:
            y = y[7]
        else:
            y = -1

    y = tf.cast(y, tf.int32)

    return X, y


def torch_to_tf(D: torch.utils.data.Dataset, max_size: int = 60_000) -> tf.data.Dataset:
    loader = torch.utils.data.DataLoader(
        D, batch_size=min(max_size, len(D)), num_workers=20
    )
    for X, y in loader:
        pass

    def to_channels_last(X, y):
        X = tf.transpose(X, [1, 2, 0])
        return X, y

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
        ),
        split=split,
    )


def load_nih_ood(split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    nih_path = datasets_root / "NIH"
    ood_diseases = ["Cardiomegaly", "Pneumothorax", "Nodule", "Mass"]
    test_length = 20_000
    return extract_split(
        D=NIHChest(
            root_path=nih_path,
            test_length=test_length,
            keep_in_classes=ood_diseases,
        ),
        split=split,
    )


def load_mura(split: str):
    assert split == "train"
    print(f"Loading mura with split {split}")
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
    )
    return extract_split(D, split)


def load_drimdb(split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    path = datasets_root / "DRIMDB"
    D = DRIMDB(root_path=path)
    return extract_split(D, split)


def load_drd(split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    path = datasets_root / "DRD"
    D = DRD(root_path=path)
    return extract_split(D, split)


def load_riga(split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    path = datasets_root / "RIGA"
    D = RIGA(root_path=path)
    return extract_split(D, split)


def load_pcam(split: str):
    def f_map(X, y):
        X = tf.cast(X, "float32") / 255.0
        y = tf.cast(y, "int32")
        X = tf.image.resize(X, (224, 224))
        return X, y

    D = tfds.load(
        "patch_camelyon",
        split=split,
        as_supervised=True,
        data_dir=os.path.join(os.getenv("SCRATCH", "~/"), ".datasets"),
        try_gcs=False,
    )
    D = D.map(f_map)
    return D


def load_malaria(split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    path = datasets_root / "Malaria"
    D = Malaria(root_path=path)
    return extract_split(D, split)


def load_anhir(split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    path = datasets_root / "ANHIR"
    D = ANHIR(root_path=path)
    return extract_split(D, split)


def load_idc(split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    path = datasets_root / "IDC"
    D = IDC(root_path=path)
    return extract_split(D, split)


def load_pc_for_nih(split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    path = datasets_root / "PC"
    classes = [
        "AP",
        "L",
        "AP_horizontal",
        "PED",
    ]
    return extract_split(PADChest(root_path=str(path), keep_class=classes), split)


def load_pc_uc2(split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    path = datasets_root / "PC"
    classes = [
        "AP",
        "PA",
        "AP_horizontal",
        "PED",
    ]
    return extract_split(PADChest(root_path=str(path), keep_class=classes), split)


def load_pc_uc3(split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    path = datasets_root / "PC"
    return extract_split(
        PADChestSV(
            root_path=str(path),
            binary=True,
            test_length=5000,
            keep_in_classes=["cardiomegaly", "pneumothorax", "nodule", "mass"],
        ),
        split,
    )


def load_pc_id(split: str):
    datasets_root = Path(os.environ["SCRATCH"]) / ".datasets"
    path = datasets_root / "PC"
    return extract_split(
        PADChestSV(
            root_path=str(path),
            binary=True,
            test_length=5000,
            leave_out_classes=["cardiomegaly", "pneumothorax", "nodule", "mass"],
        ),
        split,
    )


def load_uc1(dataset: str, split: str) -> tf.data.Dataset:
    assert split == "test"
    params = dataset.split("/")[1:]
    base_path = Path(os.environ["SCRATCH"]) / ".datasets" / "medical_ood" / "uc1"
    for p in params:
        base_path = base_path / p
        if p == "rgb":
            D_for_valid = DRD(str(Path(os.environ["SCRATCH"]) / ".datasets" / "DRD"))
        elif p == "gray":
            D_for_valid = NIHChest(
                str(Path(os.environ["SCRATCH"]) / ".datasets" / "NIH")
            )
        else:
            assert False, f"Unknown param {p}, from dataset {dataset}"

    combined_cache = base_path / "combined"
    if (combined_cache / "dataset").exists():
        with (combined_cache / "element_spec.pkl").open("r+b") as f:
            element_spec = pickle.load(f)

        return tf.data.experimental.load(str(combined_cache / "dataset"), element_spec)

    uc1s = [
        "UniformNoise",
        "NormalNoise",
        "NotMNIST",
        # "TinyImagenet",
        "MNIST",
        "FashionMNIST",
        "CIFAR100",
        "CIFAR10",
        "STL10",
    ]

    take_per_dset = 10_000 // len(uc1s) + 1

    def to_fake_label(X, y):
        y = -1
        return X, y

    print("Loading mura...")

    D_for_uc1 = []
    for d in uc1s:
        print(f"Loading {d}")
        dataset = Global.all_datasets[d]

        cache_path = base_path / d

        print(f"Generating dataset {d}")
        if "dataset_path" in dataset.__dict__:
            D = dataset(
                root_path=(
                    Path(os.environ["SCRATCH"])
                    / ".datasets"
                    / "medical_ood"
                    / "uc1"
                    / dataset.dataset_path
                ),
                download=True,
                extract=True,
            )
        else:
            D = dataset()
        ds = []
        rem = take_per_dset
        for split in [
            D.get_D2_test(D_for_valid),
            D.get_D2_valid(D_for_valid),
        ]:
            cur_d = torch_to_tf(split, max_size=rem)
            rem -= len(cur_d)
            ds.append(cur_d)
            if rem == 0:
                break

        D = ds[0]
        for x in ds[1:]:
            D = D.concatenate(x)

        D = D.take(take_per_dset)
        D = D.map(to_fake_label)

        D_for_uc1.append(D)

    D = D_for_uc1[0]
    for d in D_for_uc1[1:]:
        D = D.concatenate(d)
    D = D.shuffle(15_000)
    D = D.take(10_000)

    tf.data.experimental.save(D, str(combined_cache / "dataset"))
    with (combined_cache / "element_spec.pkl").open("w+b") as f:
        pickle.dump(D.element_spec, f)

    return D


def load_uc1_and_mura(split: str):
    assert split == "test"
    D = load_dataset("mura", "train").take(5_000)
    D = D.concatenate(load_uc1("uc1/gray", split).take(5_000).map(f_normalize_label))
    return D


def load_uc1_and_malaria(split: str):
    assert split == "test"
    D = load_dataset("malaria", "train").take(5_000)
    D = D.concatenate(load_uc1("uc1/rgb", split).take(5_000).map(f_normalize_label))
    return D


def maybe_load_cached(dataset: str, split: str) -> Optional[tf.data.Dataset]:
    path = Path(os.environ["SCRATCH"]) / ".datasets" / "medical_ood" / dataset / split
    if not path.exists():
        return None

    if (path / "element_spec.pkl").exists():
        with (path / "element_spec.pkl").open("r+b") as f:
            element_spec = pickle.load(f)
    elif "nih" in dataset:
        element_spec = (
            tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(15,), dtype=tf.int64),
        )
    elif "mura" in dataset:
        element_spec = (
            tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(7,), dtype=tf.int64),
        )
    else:
        assert False
    return tf.data.experimental.load(str(path / "dataset"), element_spec)


DATASETS = {
    "nih_id": load_nih_id,
    "nih_ood": load_nih_ood,
    "mura": load_mura,
    "drimdb": load_drimdb,
    "drd": load_drd,
    "riga": load_riga,
    "pcam": load_pcam,
    "malaria": load_malaria,
    # "anhir": load_anhir,
    "idc": load_idc,
    "uc1_and_mura": load_uc1_and_mura,
    "uc1_and_malaria": load_uc1_and_malaria,
    "uc1": load_uc1,
    "pc_for_nih": load_pc_for_nih,
    "pc_id": load_pc_id,
    "pc_uc2": load_pc_uc2,
    "pc_uc3": load_pc_uc3,
}


def impl_load_dataset(dataset: str, split: str) -> tf.data.Dataset:
    D = maybe_load_cached(dataset, split)
    if D is not None:
        return D

    raw_name = dataset.split("/")[0]
    assert raw_name in DATASETS, f"Unrecognized dataset: {dataset}"
    f = DATASETS[raw_name]
    if raw_name == "uc1":
        return load_uc1(dataset, split)
    return f(split)


def load_dataset(dataset: str, split: str) -> tf.data.Dataset:
    D = impl_load_dataset(dataset, split)
    D = D.map(f_normalize_label)

    if dataset in [
        "nih_ood",
        "pc_uc2",
        "drimdb",
        "riga",
        "pc_for_nih",
        "pc_uc3",
    ]:
        # Since this dataset is only ever used for test, put all samples
        # together.
        assert split == "test"
        for d in ["train", "validation"]:
            D = D.concatenate(impl_load_dataset(dataset, d).map(f_normalize_label))

        D = D.take(10_000)

    if dataset in ["pc_id", "drd"] and split == "test":
        D = D.concatenate(
            impl_load_dataset(dataset, "validation").map(f_normalize_label)
        )

    return D


def is_medical(dataset_name: str):
    raw_name = dataset_name.split("/")[0]
    return raw_name in DATASETS


def get_num_classes(dataset_name: str) -> int:
    assert is_medical(dataset_name)
    return 2


def get_image_size(dataset_name: str):
    D = load_dataset(dataset_name, "test")
    for X, y in D:
        break
    return tuple(X.shape.as_list())


def get_normalization(dataset_name: str):
    if "nih" in dataset_name:
        mean = (0.25720176,)
        stddev = (0.2550173,)
        return mean, stddev
    assert False


def preprocess(datasets=None, splits=["train", "test", "validation"]):
    if datasets is None:
        datasets = DATASETS.keys()
    for d in datasets:
        if "uc1" in d:
            continue

        if "mura" in d:
            splits = ["train"]

        for split in splits:
            if maybe_load_cached(d, split) is not None:
                continue

            cur_path = (
                Path(os.environ["SCRATCH"]) / ".datasets" / "medical_ood" / d / split
            )
            print(f"Preprocessing {d}/{split}... ", end="", flush=True)
            D = impl_load_dataset(d, split)

            if len(D) == 0:
                continue

            tf.data.experimental.save(
                D,
                str(cur_path / "dataset"),
            )
            print("done!")

            with (cur_path / "element_spec.pkl").open("w+b") as f:
                pickle.dump(D.element_spec, f)


def test():
    print("OOD datasets:")
    for d in [
        "drimdb",
        "nih_ood",
        "pc_for_nih",
        "pc_uc2",
        "pc_uc3",
        "riga",
        "uc1_and_malaria",
        "uc1_and_mura",
    ]:
        D = load_dataset(d, "test")
        n = 0
        for X, y in D:
            assert len(y.shape) == 0
            n += 1
        print(f"{d}/test: {n}")

    print("ID datasets:")
    for d in ["nih_id", "pc_id", "pcam", "drd"]:
        for s in ["train", "test"]:
            print(f"{d}: ", end="")
            D = load_dataset(d, s)
            n = 0
            for X, y in D:
                assert len(y.shape) == 0
                n += 1
            print(f"{s}({n}) ", end="")
        print()


if __name__ == "__main__":
    test()
