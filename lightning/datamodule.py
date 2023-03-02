from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningDataModule
from torchvision import transforms
import torch
import torch.nn as nn
import random
import json
from pathlib import Path
from torchvision.datasets.folder import pil_loader


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class LFDataModule(LightningDataModule):
    def __init__(
        self,
        traindataset,
        trainfilter,
        trainlambda_oh=None,
        trainlambda_bin=None,
        testdataset=None,
        validationdataset=None,
        validationfilter=None,
        vallambda_oh=None,
        posterior=None,
        lfmodelout=None,
        batch_size: int = 16,
        num_workers: int = 16,
        augment: bool = True,
        drop_last: bool = False,
        augment_style: str = "cifar",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.traindataset = traindataset
        self.trainlambda_oh = trainlambda_oh
        self.trainlambda_bin = trainlambda_bin
        self.trainfilter = trainfilter
        self.testdataset = testdataset
        self.validationfilter = validationfilter
        self.vallambda_oh = vallambda_oh
        self.posterior = posterior
        self.lfmodelout = lfmodelout
        self.validationdataset = validationdataset
        self.augment_style = augment_style
        self.augment = augment
        self.drop_last = drop_last

    def get_transforms(self):
        if self.augment:
            if self.augment_style.lower() == "cifar":
                # color augment
                transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            32, scale=(0.87, 1.0), ratio=(0.8, 1.2)
                        ),
                        transforms.RandomAdjustSharpness(1.2, p=0.1),
                        RandomApply(
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.03
                            ),
                            p=0.3,
                        ),
                        transforms.RandomAutocontrast(p=0.2),
                        RandomApply(
                            transforms.GaussianBlur((3, 3), (0.1, 0.5)), p=0.05
                        ),
                        transforms.Normalize(
                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                        ),  # normalize to -1,1
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.02),
                    ]
                )
            elif self.augment_style.lower() == "cifar64":
                # color augment
                transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            64, scale=(0.9, 1.0), ratio=(0.8, 1.2)
                        ),
                        transforms.RandomAdjustSharpness(1.2, p=0.1),
                        RandomApply(
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15
                            ),
                            p=0.3,
                        ),
                        transforms.RandomAutocontrast(p=0.2),
                        RandomApply(
                            transforms.GaussianBlur((3, 3), (0.1, 0.5)), p=0.05
                        ),
                        transforms.Normalize(
                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                        ),  # normalize to -1,1
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.02),
                    ]
                )
            elif self.augment_style.lower() == "food101":
                # color augment
                transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            128, scale=(0.7, 1.0), ratio=(0.8, 1.2)
                        ),
                        transforms.RandomAdjustSharpness(1.2, p=0.1),
                        RandomApply(
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02
                            ),
                            p=0.3,
                        ),
                        RandomApply(transforms.GaussianBlur((3, 3), (0.1, 0.5)), p=0.1),
                        transforms.RandomAutocontrast(p=0.2),
                        transforms.Normalize(
                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                        ),  # normalize to -1,1
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.02),
                    ]
                )
            elif self.augment_style.lower() == "LSUN":
                # color augment
                transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            128, scale=(0.7, 1.0), ratio=(0.8, 1.2)
                        ),
                        transforms.RandomAdjustSharpness(1.2, p=0.1),
                        RandomApply(
                            transforms.ColorJitter(
                                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
                            ),
                            p=0.3,
                        ),
                        RandomApply(transforms.GaussianBlur((3, 3), (0.1, 0.5)), p=0.1),
                        transforms.RandomAutocontrast(p=0.2),
                        transforms.Normalize(
                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                        ),  # normalize to -1,1
                        transforms.RandomHorizontalFlip(p=0.5)
                    ]
                )
            elif self.augment_style.lower() == "gtsrb64":
                # color augment
                transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            (64, 64), scale=(0.9, 1.0), ratio=(0.8, 1.2)
                        ),
                        transforms.RandomAdjustSharpness(1.2, p=0.1),
                        RandomApply(
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.05
                            ),
                            p=0.3,
                        ),
                        RandomApply(
                            transforms.GaussianBlur((3, 3), (0.1, 0.5)), p=0.05
                        ),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
            elif self.augment_style.lower() == "mnist":
                transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            32, scale=(0.87, 1.0), ratio=(0.8, 1.2)
                        ),
                        transforms.RandomAdjustSharpness(1.2, p=0.2),
                        RandomApply(transforms.GaussianBlur((3, 3), (0.1, 0.5)), p=0.1),
                        transforms.Normalize([0.5], [0.5]),  # normalize to -1,1
                    ]
                )
            elif self.augment_style.lower() == "fashionmnist":
                transform = transforms.Compose(
                    [
                        transforms.RandomAdjustSharpness(1.2, p=0.2),
                        transforms.Normalize([0.5], [0.5]),  # normalize to -1,1
                        transforms.RandomHorizontalFlip(p=0.5),
                    ]
                )
            elif self.augment_style.lower() == "mnisttest":
                transform = transforms.Compose(
                    [transforms.Normalize((0.5,), (0.5,))]
                )  # normalize to -1,1
            elif self.augment_style.lower() == "cifartest":
                transform = transforms.Compose(
                    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )  # normalize to -1,1
            else:
                raise NotImplementedError("augment_style unknown")
        else:
            if self.augment_style.lower() in ["cifar", "cifar64"]:
                transform = transforms.Compose(
                    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )  # normalize to -1,1
            elif self.augment_style.lower() == "gtsrb64":
                # normalize to -1,1
                transform = [
                    transforms.Resize((64, 64)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            elif self.augment_style.lower() == "mnist":
                transform = transforms.Compose(
                    [transforms.Normalize((0.5,), (0.5,))]
                )  # normalize to -1,1
            else:
                raise NotImplementedError("augment_style unknown")
        if self.augment_style.lower() in ["cifar", "cifar64"]:
            # color images
            testtransform = transforms.Compose(
                [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )  # normalize to -1,1
        elif self.augment_style.lower() in ["gtsrb64"]:
            testtransform = transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )  # normalize to -1,1
        else:
            # greyscale images
            testtransform = transforms.Compose(
                [transforms.Normalize((0.5,), (0.5,))]
            )  # normalize to -1,1
        return transform, testtransform

    def prepare_data(self, stage=None):
        # Use this method to do things that might write to disk or that need to be done only from a single GPU
        # in distributed settings. Like downloading the dataset for the first time.
        return None

    def setup(self, stage=None):
        # There are also data operations you might want to perform on every GPU, such as applying transforms
        # defined explicitly in your datamodule or assigned in init.
        transform, testtransform = self.get_transforms()
        self.data_train = LFDataset(
            self.traindataset,
            filteridxs=self.trainfilter,
            onehot_lfs=self.trainlambda_oh,
            lfmodelout=self.lfmodelout,
            posterior=self.posterior,
            binary_lfs=self.trainlambda_bin,
            transform=transform,
        )

        if self.validationdataset is not None:
            if self.validationfilter is not None:
                self.data_val = LFDataset(
                    self.validationdataset,
                    filteridxs=self.validationfilter,
                    onehot_lfs=self.vallambda_oh,
                    transform=testtransform,
                )
            else:
                self.data_val = LFDataset(
                    self.validationdataset, transform=testtransform
                )

        if self.testdataset is not None:
            self.data_test = LFDataset(self.testdataset, transform=testtransform)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        if self.data_val is not None:
            return DataLoader(
                self.data_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            return None

    def test_dataloader(self):
        if self.data_test is not None:
            return DataLoader(
                self.data_test, batch_size=self.batch_size, num_workers=self.num_workers
            )
        else:
            return None


class LFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        filteridxs=None,
        onehot_lfs=None,
        binary_lfs=None,
        posterior=None,
        lfmodelout=None,
        transform=None,
    ):
        self.dataset = dataset
        self.transform = transform
        self.onehot_lfs = onehot_lfs
        self.binary_lfs = binary_lfs
        self.filteridxs = filteridxs
        self.posterior = posterior
        self.lfmodelout = lfmodelout
        self.returnlfout = True if lfmodelout is not None else False
        self.returnfilter = True if self.filteridxs is not None else False
        self.returnohlfs = True if onehot_lfs is not None else False
        self.returnbinlfs = True if binary_lfs is not None else False
        self.returnposterior = True if posterior is not None else False
        self.returnbothlfs = self.returnohlfs and self.returnbinlfs

    def __getitem__(self, index):
        x, y = self.dataset[index]
        y = int(
            y
        )  # needed to make combination of tensordataset and image dataset compatible
        if self.transform:
            x = self.transform(x)

        if self.returnbothlfs:
            sample = {
                "imgs": x,
                "y": y,
                "oh_lfs": self.onehot_lfs[index],
                "binary_lfs": self.binary_lfs[index],
                "filteridxs": self.filteridxs[index],
            }
        elif self.returnohlfs:
            sample = {
                "imgs": x,
                "y": y,
                "oh_lfs": self.onehot_lfs[index],
                "filteridxs": self.filteridxs[index],
            }
        elif self.returnohlfs:
            sample = {
                "imgs": x,
                "y": y,
                "binary_lfs": self.binary_lfs[index],
                "filteridxs": self.filteridxs[index],
            }
        else:
            if self.returnfilter:
                sample = {"imgs": x, "y": y, "filteridxs": self.filteridxs[index]}
            else:
                sample = {"imgs": x, "y": y}

        if self.returnposterior:
            sample["posterior"] = self.posterior[index]

        if self.returnlfout:
            sample["lfmodelout"] = self.lfmodelout[index]

        return sample

    def __len__(self):
        return len(self.dataset)


class JSONImageDataset(torch.utils.data.Dataset):
    def __init__(self, jsonpath: str, img_root: str, transform=None, size=(64, 64)):
        with open(jsonpath, "r") as jsfile:
            data = json.loads(jsfile.read())
        data = [(k, int(v["label"]), v["data"]["image_path"]) for k, v in data.items()]
        data.sort(key=lambda x: int(x[0]))
        self.data = data
        self.img_root = Path(img_root)
        self.totensor = transforms.ToTensor()
        self.transform = transform
        self.size = size

    def __getitem__(self, index):
        _, y, path = self.data[index]
        x = pil_loader(self.img_root / path)
        if x.getbands()[0] == "L":
            x = x.convert("RGB")
        x = x.resize(self.size)

        x = self.totensor(x)

        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)
