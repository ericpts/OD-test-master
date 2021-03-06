import torch
import torchvision.transforms as transforms
from . import SubDataset, AbstractDomainInterface
from torchvision import datasets


class MNIST(AbstractDomainInterface):
    """
    MNIST: 60,000 train + 10,000 test.
    Ds: (50,000 train + 10,000 valid) + (10,000 test)
    Dv, Dt: 60,000 valid + 10,000 test.
    """

    dataset_path = "mnist"

    def __init__(
        self,
        root_path="./workspace/datasets/mnist",
        download=False,
        extract=False,
        doubledownsample=None,
    ):
        super(MNIST, self).__init__()

        im_transformer = transforms.Compose([transforms.ToTensor()])
        self.D1_train_ind = torch.arange(0, 50000).int()
        self.D1_valid_ind = torch.arange(50000, 60000).int()
        self.D1_test_ind = torch.arange(0, 10000).int()

        self.D2_valid_ind = torch.arange(0, 60000).int()
        self.D2_test_ind = torch.arange(0, 10000).int()

        self.ds_train = datasets.MNIST(
            root_path, train=True, transform=im_transformer, download=download
        )
        self.ds_test = datasets.MNIST(
            root_path, train=False, transform=im_transformer, download=download
        )

    def get_D1_train(self):
        return SubDataset(self.name, self.ds_train, self.D1_train_ind)

    def get_D1_valid(self):
        return SubDataset(self.name, self.ds_train, self.D1_valid_ind, label=0)

    def get_D1_test(self):
        return SubDataset(self.name, self.ds_test, self.D1_test_ind, label=0)

    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(
            self.name,
            self.ds_train,
            self.D2_valid_ind,
            label=1,
            transform=D1.conformity_transform(),
        )

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(
            self.name,
            self.ds_test,
            self.D2_test_ind,
            label=1,
            transform=D1.conformity_transform(),
        )

    def conformity_transform(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )


if __name__ == "__main__":
    dataset = MNIST()
    d1_train = dataset.get_D1_train()
    print(len(d1_train))

    import matplotlib.pyplot as plt

    for batch_ind in range(len(d1_train)):
        print(batch_ind)
        x, y = d1_train[batch_ind]
        plt.imshow(x.numpy().reshape(dataset.image_size))
