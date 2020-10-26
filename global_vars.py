"""
    This file lists all the global variables that are used throughout the project.
    The two major components of this file are the list of the datasets and the list of the models.
"""

"""
    This is where we keep a reference to all the dataset classes in the project.
"""
import medical_ood.datasets.MNIST as MNIST
import medical_ood.datasets.FashionMNIST as FMNIST
import medical_ood.datasets.notMNIST as NMNIST
import medical_ood.datasets.CIFAR as CIFAR
import medical_ood.datasets.noise as noise
import medical_ood.datasets.STL as STL
import medical_ood.datasets.TinyImagenet as TI
import medical_ood.datasets.NIH_Chest as NC
import medical_ood.datasets.MURA as MU
import medical_ood.datasets.PADChest as PC
import medical_ood.datasets.malaria as mal
import medical_ood.datasets.ANHIR as ANH
import medical_ood.datasets.DRD as DRD
import medical_ood.datasets.DRIMDB as DRM
import medical_ood.datasets.IDC as IDC
import medical_ood.datasets.PCAM as PCAM
import medical_ood.datasets.RIGA as RIGA

all_dataset_classes = [
    MNIST.MNIST,
    FMNIST.FashionMNIST,
    NMNIST.NotMNIST,
    CIFAR.CIFAR10,
    CIFAR.CIFAR100,
    STL.STL10,
    TI.TinyImagenet,
    noise.UniformNoise,
    noise.NormalNoise,
    STL.STL10d32,
    TI.TinyImagenetd32,
    NC.NIHChest,
    NC.NIHChestBinary,
    NC.NIHChestBinaryTest,
    NC.NIHChestBinaryTrainSplit,
    NC.NIHChestBinaryValSplit,
    NC.NIHChestBinaryTestSplit,
    MU.MURA,
    MU.MURAHAND,
    MU.MURAELBOW,
    MU.MURAFINGER,
    MU.MURAFOREARM,
    MU.MURAHUMERUS,
    MU.MURASHOULDER,
    MU.MURAWRIST,
    PC.PADChest,
    PC.PADChestAP,
    PC.PADChestPA,
    PC.PADChestL,
    PC.PADChestAPHorizontal,
    PC.PADChestPED,
    mal.Malaria,
    ANH.ANHIR,
    DRD.DRD,
    DRM.DRIMDB,
    IDC.IDC,
    PCAM.PCAM,
    PCAM.PCAMGray,
    RIGA.RIGA,
]

"""
    Not all the datasets can be used as a Dv, Dt (aka D2) for each dataset.
    The list below specifies which datasets can be used as the D2 for the other datasets.
    For instance, STL10 and CIFAR10 cannot face each other because they have 9 out 10 classes
    in common.
"""
d2_compatiblity = {
    # This can be used as d2 for            # this
    "MNIST": [
        "FashionMNIST",
        "CIFAR10",
        "CIFAR100",
        "STL10",
        "TinyImagenet",
        "STL10d32",
        "TinyImagenetd32",
        "NIHCC",
        "NIHChestBinaryTest",
        "NIHChestBinaryTrainSplit",
        "PADChest",
        "DRD",
        "PCAM",
    ],
    "NotMNIST": [
        "MNIST",
        "FashionMNIST",
        "CIFAR10",
        "CIFAR100",
        "STL10",
        "TinyImagenet",
        "STL10d32",
        "TinyImagenetd32",
        "NIHCC",
        "NIHChestBinaryTest",
        "NIHChestBinaryTrainSplit",
        "PADChest",
        "DRD",
        "PCAM",
    ],
    "FashionMNIST": [
        "MNIST",
        "CIFAR10",
        "CIFAR100",
        "STL10",
        "TinyImagenet",
        "STL10d32",
        "TinyImagenetd32",
        "NIHCC",
        "NIHChestBinaryTest",
        "NIHChestBinaryTrainSplit",
        "PADChest",
        "DRD",
        "PCAM",
    ],
    "CIFAR10": [
        "MNIST",
        "FashionMNIST",
        "CIFAR100",
        "TinyImagenet",
        "TinyImagenetd32",
        "NIHCC",
        "NIHChestBinaryTest",
        "NIHChestBinaryTrainSplit",
        "PADChest",
        "DRD",
        "PCAM",
    ],
    "CIFAR100": [
        "MNIST",
        "FashionMNIST",
        "CIFAR10",
        "STL10",
        "TinyImagenet",
        "STL10d32",
        "TinyImagenetd32",
        "NIHCC",
        "NIHChestBinaryTest",
        "NIHChestBinaryTrainSplit",
        "PADChest",
        "DRD",
        "PCAM",
    ],
    "STL10": [
        "MNIST",
        "FashionMNIST",
        "CIFAR100",
        "TinyImagenet",
        "TinyImagenetd32",
        "NIHCC",
        "NIHCC",
        "NIHChestBinaryTrainSplit",
        "PADChest",
        "DRD",
        "PCAM",
    ],
    "TinyImagenet": [
        "MNIST",
        "FashionMNIST",
        "CIFAR10",
        "CIFAR100",
        "STL10",
        "STL10d32",
        "NIHCC",
        "NIHChestBinaryTest",
        "NIHChestBinaryTrainSplit",
        "PADChest",
        "DRD",
        "PCAM",
    ],
    "NIHChestBinary": [
        "MNIST",
        "FashionMNIST",
        "CIFAR10",
        "CIFAR100",
        "STL10",
        "TinyImagenet",
        "STL10d32",
        "TinyImagenetd32",
        "DRD",
        "PCAM",
    ],
    "NIHCC": [
        "FashionMNIST",
        "CIFAR10",
        "CIFAR100",
        "STL10",
        "TinyImagenet",
        "STL10d32",
        "TinyImagenetd32",
        "NIHCC",
        "DRD",
        "PCAM",
    ],
    "NIHChestBinaryValSplit": [
        "FashionMNIST",
        "CIFAR10",
        "CIFAR100",
        "STL10",
        "TinyImagenet",
        "STL10d32",
        "TinyImagenetd32",
        "NIHChestBinaryTrainSplit",
        "DRD",
        "PCAM",
    ],
    "MURA": ["NIHCC", "PADChest"],
    "MURAHAND": ["NIHCC", "PADChest"],
    "MURAWRIST": ["NIHCC", "PADChest"],
    "MURAELBOW": ["NIHCC", "PADChest"],
    "MURAFINGER": ["NIHCC", "PADChest"],
    "MURAFOREARM": ["NIHCC", "PADChest"],
    "MURAHUMERUS": ["NIHCC", "PADChest"],
    "MURASHOULDER": ["NIHCC", "PADChest"],
    "PADChest": ["NIHCC", "PADChest"],
    "PADChestPA": ["NIHCC", "PADChest"],
    "PADChestAP": ["NIHCC", "PADChest"],
    "PADChestL": ["NIHCC", "PADChest"],
    "PADChestAPHorizontal": ["NIHCC", "PADChest"],
    "PADChestPED": ["NIHCC", "PADChest"],
    "Malaria": [
        "PCAM",
    ],
    "ANHIR": [
        "PCAM",
    ],
    "IDC": [
        "PCAM",
    ],
    "DRIMDB": [
        "DRD",
    ],
    "RIGA": [
        "DRD",
    ],
    # STL10 is not compatible with CIFAR10 because of the 9-overlapping classes.
    # Erring on the side of caution.
}

# We can augment the following training data with mirroring.
# We make sure there's no information leak in-between tasks.
mirror_augment = {
    "FashionMNIST",
    "CIFAR10",
    "CIFAR100",
    "STL10",
    "TinyImagenet",
    "STL10d32",
    "TinyImagenetd32",
}


##################################################################
# Do not change anything below, unless you know what you are doing.
"""
    all_datasets is automatically generated
    all_datasets = {
        'MNIST' : MNIST,
        ...
    }

"""
all_datasets = {}
for dscls in all_dataset_classes:
    all_datasets[dscls.__name__] = dscls


def get_ref_classifier(dataset):
    if dataset in dataset_reference_classifiers:
        return dataset_reference_classifiers[dataset]
    raise NotImplementedError()


def get_ref_autoencoder(dataset):
    if dataset in dataset_reference_autoencoders:
        return dataset_reference_autoencoders[dataset]
    raise NotImplementedError()


def get_ref_vae(dataset):
    if dataset in dataset_reference_vaes:
        return dataset_reference_vaes[dataset]
    raise NotImplementedError()


def get_ref_ali(dataset):
    if dataset in dataset_reference_ALI:
        return dataset_reference_ALI[dataset]
    raise NotImplementedError()


def get_ref_pixelcnn(dataset):
    if dataset in dataset_reference_pcnns:
        return dataset_reference_pcnns[dataset]
    raise NotImplementedError()


def get_method(name, args):
    elements = name.split("/")
    instance = all_methods[elements[0]](args)
    if len(elements) > 1:
        instance.default_model = int(elements[1])
    return instance
