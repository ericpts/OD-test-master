import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from datasets import SubDataset, AbstractDomainInterface, ExpandRGBChannels
import os
import os.path as osp
import csv
import subprocess
from PIL import Image

def to_tensor(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

def group_normalize(crops):
    return torch.stack([transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])(crop) for crop in crops])

class MURABase(data.Dataset):
    def __init__(self, source_dir, split, index_file="train_image_paths.csv",
                  image_dir="train", imsize=224, transforms=None, to_rgb=False, download=False, extract=True):
        super(MURABase,self).__init__()
        self.source_dir = source_dir
        self.split = split
        self.index_file = index_file
        self.image_dir = image_dir

        self.imsize = imsize
        self.to_rgb = to_rgb
        if transforms is None:
            self.transforms = transforms.Compose([transforms.Resize((imsize, imsize)),
                                                  transforms.ToTensor()])
        else:
            self.transforms = transforms
        assert split in ["train", "val"]
        if extract:
            self.extract()
            cache_file = self.generate_index()
            self.img_list = cache_file['img_list']
            self.label_tensors = cache_file['label_tensors']
            self.split_inds = cache_file["split_inds"]

    def __len__(self):
        return len(self.split_inds)

    def __getitem__(self, item):
        index = self.split_inds[item]
        img_name = self.img_list[index]
        label = self.label_tensors[index]

        imp = osp.join(self.source_dir, self.image_dir, img_name)
        with open(imp, 'rb') as f:
            with Image.open(f) as img:
                if self.to_rgb:
                    img = self.transforms(img.convert('RGB'))
                else:
                    img = self.transforms(img.convert('L'))
        return img, label

    def extract(self):
        if os.path.exists(os.path.join(self.source_dir, self.image_dir)):
            return
        import tarfile
        with tarfile.open(os.path.join(self.source_dir, "images.tar.gz")) as tar:
            tar.extractall()
        return

    def generate_index(self):
        """
        Scan index file to create list of images and labels for each image
        :return:
        """
        img_list = []
        label_list = []
        print("Reading %s"%self.index_file)
        with open(osp.join(self.source_dir, self.index_file), 'r') as fp:
            csvf = csv.DictReader(fp, ['Image Path', ])
            for row in csvf:
                imp = osp.join(self.source_dir, row['Image Path'][10:])
                if osp.exists(imp):
                    #add subpath after 'train' or 'valid' and image name to img_list
                    img_list.append('/'.join(row['Image Path'][10:].split('/')[1:]))
                    label = [0, 1] if 'positive' in row['Image Path'] else [1, 0]
                    label_list.append(label)
        label_tensors = torch.LongTensor(label_list)
        return {'img_list': img_list, 'label_tensors': label_tensors, 'label_list': label_list,
                    'split_inds': torch.arange(len(img_list))
                    }


class MURA(AbstractDomainInterface):
    dataset_path = "MURA"

    def __init__(self, root_path="./workspace/datasets/MURA", downsample=None, expand_channels=False,
                 test_length=None, download=False, extract=True):
        self.name = "MURA"
        super(MURA, self).__init__()
        self.downsample = downsample
        self.expand_channels=expand_channels
        self.max_l = test_length
        cache_path = root_path
        source_path = root_path
        if downsample is not None:
            print("downsampling to", downsample)
            transform = transforms.Compose([transforms.Resize((downsample, downsample)),
                                            transforms.ToTensor()])
            self.image_size = (downsample, downsample)
        else:
            transform = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor()])
            self.image_size = (224, 224)

        self.ds_train = MURABase(source_path, "train", transforms=transform,index_file="train_image_paths.csv",
                                 image_dir="images_224", to_rgb=expand_channels, download=download, extract=extract)
        self.ds_valid = MURABase(source_path, "val", transforms=transform,index_file="valid_image_paths.csv",
                                 image_dir="images_224", to_rgb=expand_channels, download=download, extract=extract)

        if extract:
            self.D1_train_ind = self.get_filtered_inds(self.ds_train, shuffle=True)
            self.D1_valid_ind = self.get_filtered_inds(self.ds_valid, shuffle=True, max_l=self.max_l)
            self.D1_test_ind = self.get_filtered_inds(self.ds_valid, shuffle=True)

            self.D2_valid_ind = self.get_filtered_inds(self.ds_train, shuffle=True)
            self.D2_test_ind = self.get_filtered_inds(self.ds_valid)

    def get_filtered_inds(self, basedata: MURABase, shuffle=False, max_l=None):
        output_inds = torch.arange(0, len(basedata)).int()
        if shuffle:
            output_inds = output_inds[torch.randperm(len(output_inds))]
        if max_l is not None:
            if len(output_inds) >max_l:
                output_inds = output_inds[:max_l]
        return output_inds

    def get_D1_train(self):
        return SubDataset(self.name, self.ds_train, self.D1_train_ind)
    def get_D1_valid(self):
        return SubDataset(self.name, self.ds_valid, self.D1_valid_ind, label=0)
    def get_D1_test(self):
        return SubDataset(self.name, self.ds_valid, self.D1_test_ind, label=0)

    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        target_indices = self.D2_valid_ind
        return SubDataset(self.name, self.ds_train, target_indices, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        target_indices = self.D2_test_ind
        return SubDataset(self.name, self.ds_valid, target_indices, label=1, transform=D1.conformity_transform())

    def conformity_transform(self):
        target = 224
        if self.downsample is not None:
            target = self.downsample
        if self.expand_channels:
            return transforms.Compose([ExpandRGBChannels(),
                                        transforms.ToPILImage(),
                                       transforms.Resize((target, target)),
                                       transforms.ToTensor()
                                       ])
        else:
            return transforms.Compose([
                                       transforms.ToPILImage(),
                                       transforms.Grayscale(),
                                       transforms.Resize((target, target)),
                                       transforms.ToTensor()
                                       ])


if __name__ == "__main__":
    dataset = MURA()
    d1_train = dataset.get_D1_train()
    print(len(d1_train))
    loader = data.DataLoader(d1_train, batch_size=1, shuffle=True)
    import matplotlib.pyplot as plt
    for batch, batch_ind in zip(loader, range(10)):
        print(batch_ind)
        x, y = batch
        plt.imshow(x.numpy().reshape(dataset.image_size), cmap='gray')

