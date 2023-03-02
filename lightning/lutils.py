import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from lightning.networks import (
    DCGenerator64,
    DCGeneratorThree,
    DCDiscriminatorThree,
    InfoDCDiscriminator,
    InfoDCDiscriminator64,
)

def create_L_ind(L, cardinality):
    """ Adapted from Snorkel v0.9
    Convert a label matrix with labels in 0...k to a one-hot format.
    Parameters
    ----------
    L
        An (n,m) label matrix with values in {0,1,...,C}, where 0 means abstains
    Returns
    cardinality
        Number of labels C
    -------
    torch.tensor
        An (n, m, C) tensor with values in {0,1}
    """
    n, m = L.shape

    L_ind = torch.zeros((n, m, cardinality), requires_grad=False)
    for class_y in range(1, cardinality + 1):
        # go through Y == 1 (negative), Y == 2 (positive)...
        # A[x::y] slices A starting at x at intervals of y
        # e.g., np.arange(9)[0::3] == np.array([0,3,6])
        L_ind[:, :, class_y - 1] = torch.where(L == class_y, 1, 0)
    return L_ind



def get_lf_stats(Lambdas, Ytrue):
    n, m = Lambdas.shape
    lf_accuracies = np.zeros(m)
    lf_coverage = np.zeros(m)
    for i, col in enumerate(Lambdas.T):
        pos = (Ytrue == col - 1).sum()
        nonabstain = np.count_nonzero(col)
        if nonabstain > 0:
            lf_accuracies[i] = pos / nonabstain
            lf_coverage[i] = nonabstain / n
    return lf_accuracies, lf_coverage


def load_awa_lfs(jsonpath):
    with open(jsonpath, "r") as jsfile:
        data = json.loads(jsfile.read())
    data = [(k, v["weak_labels"], int(v["label"])) for k, v in data.items()]
    data.sort(key=lambda x: int(x[0]))
    # original labels indexed 0..C-1
    # For LFs, we use 0 as abstain and do 1..C
    Lambdas = np.array([x[1] for x in data]) + 1
    if len(Lambdas.shape) > 2:
        Lambdas = Lambdas.squeeze()
    Ytrue = np.array([x[2] for x in data])
    return Lambdas, Ytrue


def load_fashion_mnist_lfs_idxs(jsonroot):
    # load train
    with open(os.path.join(jsonroot, "train.json"), "r") as jsfile:
        data = json.loads(jsfile.read())
    data = [
        (k, v["weak_labels"], int(v["label"]), int(v["data"])) for k, v in data.items()
    ]
    data.sort(key=lambda x: x[0])
    # original labels indexed 0..C-1
    # For LFs, we use 0 as abstain and do 1..C
    Lambdas = np.array([x[1] for x in data]) + 1
    if len(Lambdas.shape) > 2:
        Lambdas = Lambdas.squeeze()
    train_indices = np.array([x[3] for x in data])
    # load val indices
    with open(os.path.join(jsonroot, "valid.json"), "r") as jsfile:
        data = json.loads(jsfile.read())
    val_indices = [int(v["data"]) for k, v in data.items()]
    return Lambdas, train_indices, val_indices


def load_domainnet_lfs(jsonpath, threshold=0.9, applythreshold=False):
    with open(jsonpath, "r") as jsfile:
        data = json.loads(jsfile.read())
    data = [
        (k, v["weak_labels"], int(v["label"]), v["data"]["weak_probs"])
        for k, v in data.items()
    ]
    data.sort(key=lambda x: int(x[0]))
    # original labels indexed 0..C-1
    # For LFs, we use 0 as abstain and do 1..C
    Lambdas = np.array([x[1] for x in data]) + 1
    if len(Lambdas.shape) > 2:
        Lambdas = Lambdas.squeeze()
    Ytrue = np.array([x[2] for x in data])
    if applythreshold:
        for i, val in enumerate(data):
            ind = np.array(val[-1]).max(1) < threshold
            Lambdas[i, ind] = 0  # abstain since prob below threshold
    return Lambdas, Ytrue


def get_networks_base(architecture, latent_dim, latent_code_dim, img_shape):
    if architecture.lower() == "infodc":
        generator = DCGeneratorThree(
            latent_dim=latent_dim + latent_code_dim, img_shape=img_shape
        )
        discriminator = InfoDCDiscriminator(
            img_shape=img_shape, n_codes=latent_code_dim
        )
    elif architecture.lower() == "infodc64":
        generator = DCGenerator64(
            latent_dim=latent_dim + latent_code_dim, img_shape=img_shape
        )
        discriminator = InfoDCDiscriminator64(
            img_shape=img_shape, n_codes=latent_code_dim
        )
    elif architecture.lower() == "dc":
        generator = DCGeneratorThree(latent_dim=latent_dim, img_shape=img_shape)
        discriminator = DCDiscriminatorThree(img_shape=img_shape)
    else:
        raise NotImplementedError(
            "The following GAN architecture is not available: %s" % architecture
        )
    return generator, discriminator


def hinge_loss_dis(fake, real):
    assert (
        fake.dim() == 2 and fake.shape[1] == 1 and real.shape == fake.shape
    ), f"{fake.shape} {real.shape}"
    return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()


def hinge_loss_gen(fake):
    assert fake.dim() == 2 and fake.shape[1] == 1
    return -fake.mean()


def get_prob_labels_q(dataloadr, n_classes, model):
    yhat = torch.zeros(
        (len(dataloadr.dataset), n_classes), requires_grad=False, device="cpu"
    )
    counter = 0
    for imgs, _ in dataloadr:
        bsize = imgs.shape[0]
        with torch.no_grad():
            # Predict latent codes on training images
            pred_code_real = model.get_prob_labels_q(imgs)
        yhat[counter : counter + bsize] = pred_code_real.to("cpu")
        counter += bsize
    return yhat


def get_prob_labels_encoder(dataloadr, n_classes, model, lambda_onehot):
    yhat = torch.zeros(
        (len(dataloadr.dataset), n_classes), requires_grad=False, device="cpu"
    )
    counter = 0
    for imgs, _ in dataloadr:
        bsize = imgs.shape[0]
        with torch.no_grad():
            # Predict latent codes on training images
            y_estimate = model.get_prob_labels(
                imgs, lambda_onehot[counter : counter + bsize]
            )
        yhat[counter : counter + bsize] = y_estimate.to("cpu")
        counter += bsize
    return yhat


def get_posterior_equal_weight(n_classes, Lambdas, Lambda_fake=None):
    num_LFs = Lambdas.shape[1]
    param = torch.ones(1, 1, num_LFs)
    lambda_oh = create_L_ind(Lambdas, n_classes)
    aggregation = torch.matmul(param, lambda_oh).squeeze(1)
    full_posterior = torch.softmax(aggregation, dim=1)
    fake_posterior = None
    if Lambda_fake is not None:
        lambda_oh = create_L_ind(Lambda_fake, n_classes)
        aggregation = torch.matmul(param, lambda_oh).squeeze(1)
        fake_posterior = torch.softmax(aggregation, dim=1)
    return full_posterior, fake_posterior


def generate_fake_data(model, n_fake, n_classes, batch_size):
    yhat = torch.zeros((n_fake, n_classes), requires_grad=False, device="cpu")
    yhat_code = torch.zeros((n_fake, n_classes), requires_grad=False, device="cpu")
    counter = 0
    fake_images = []
    while counter < n_fake:
        bsize = batch_size if n_fake - counter > batch_size else int(n_fake - counter)

        with torch.no_grad():
            _, _, discrete_z_oh, _, gen_imgs = model.generate(bsize)
            # Predict latent codes on training images
            pred_yhat = model.get_prob_labels_q(gen_imgs)
            code_mapped = model.map_code_to_label(discrete_z_oh.float())

        fake_images.append(gen_imgs.cpu())
        yhat[counter : counter + bsize] = pred_yhat.to("cpu")
        yhat_code[counter : counter + bsize] = code_mapped.to("cpu")
        counter += bsize

    fake_images = torch.cat(fake_images)
    return fake_images, yhat, yhat_code


class LinearLrSchedule:
    def __init__(self, lr, n_epochs=150, minlr=1e-8):
        self.n_epochs = n_epochs
        self.minrate = minlr / lr

    def forward(self, step):
        rate = max(1.0 - step / self.n_epochs, self.minrate)
        return rate


def get_linear_lr_schedule(lr, n_epochs=150, minlr=1e-8):
    m = LinearLrSchedule(lr, n_epochs, minlr)
    return m.forward


def get_datasets(args, basetransforms=True, altpath=None):
    if args.dataset.lower() == "lsun":
        n_classes = 10
        augment_style = None
        img_shape = (3, 256, 256)
        traindataset = datasets.LSUN(args.data_path, classes='train')
        testdataset = None
    elif args.dataset == "MNIST":
        n_classes = 10
        if basetransforms:
            # get base normalization to -1, 1 used during LF creation
            pretransforms = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            pretransforms = transforms.Compose(
                [transforms.Resize(32), transforms.ToTensor()]
            )
        augment_style = "mnist"
        img_shape = (1, 32, 32)
        traindataset = datasets.MNIST(
            args.data_path, train=True, download=True, transform=pretransforms
        )
        testdataset = datasets.MNIST(
            args.data_path, train=False, download=True, transform=pretransforms
        )
    elif args.dataset == "FashionMNIST":
        n_classes = 10
        img_shape = (1, 32, 32)
        augment_style = "mnist"
        if basetransforms:
            # get base normalization to -1, 1 used during LF creation
            pretransforms = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            pretransforms = transforms.Compose(
                [transforms.Resize(32), transforms.ToTensor()]
            )
        traindataset = datasets.FashionMNIST(
            args.data_path, train=True, download=True, transform=pretransforms
        )
        testdataset = datasets.FashionMNIST(
            args.data_path, train=False, download=True, transform=pretransforms
        )
    elif (
        args.dataset == "CIFAR10"
        or args.dataset == "CIFAR"
        or args.dataset == "CIFAR10_small"
    ):
        if basetransforms:
            # get base normalization to -1, 1 used during LF creation
            pretransforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            pretransforms = transforms.ToTensor()
        n_classes = 10
        img_shape = (3, 32, 32)
        augment_style = "cifar"
        traindataset = datasets.CIFAR10(
            args.data_path, train=True, download=True, transform=pretransforms
        )
        testdataset = datasets.CIFAR10(
            args.data_path, train=False, download=True, transform=pretransforms
        )
    elif args.dataset.lower() == "gtsrb":

        if basetransforms:
            # get base normalization to -1, 1 used during LF creation
            pretransforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            pretransforms = transforms.Compose(
                [transforms.Resize((64, 64)), transforms.ToTensor()]
            )
        n_classes = 43
        img_shape = (3, 64, 64)
        augment_style = "gtsrb64"
        traindataset = datasets.GTSRB(
            args.data_path, split="train", download=True, transform=pretransforms
        )
        testdataset = datasets.GTSRB(
            args.data_path, split="test", download=True, transform=pretransforms
        )
    else:
        raise NotImplementedError("Dataset %s not available" % args.dataset)
    return traindataset, testdataset, img_shape, n_classes, augment_style


def get_datasets_stylegan(args, basetransforms=True, altpath=None):
    """
    Loads datasets without transformations
    """
    if args.dataset.lower() == "lsun":
        n_classes = 10
        augment_style = None
        img_shape = (3, 512, 512)
        traindataset = datasets.LSUN(args.data_path, classes='train')
        testdataset = None
    elif args.dataset == "MNIST":
        n_classes = 10
        augment_style = None
        img_shape = (1, 32, 32)
        traindataset = datasets.MNIST(args.data_path, train=True, download=True)
        testdataset = datasets.MNIST(args.data_path, train=False, download=True)
    elif args.dataset == "FashionMNIST":
        n_classes = 10
        img_shape = (1, 32, 32)
        augment_style = "mnist"
        traindataset = datasets.FashionMNIST(args.data_path, train=True, download=True)
        testdataset = datasets.FashionMNIST(args.data_path, train=False, download=True)
    elif (
        args.dataset == "CIFAR10"
        or args.dataset == "CIFAR"
        or args.dataset == "CIFAR10_small"
    ):

        n_classes = 10
        img_shape = (3, 32, 32)
        augment_style = "cifar"
        traindataset = datasets.CIFAR10(args.data_path, train=True, download=True)
        testdataset = datasets.CIFAR10(args.data_path, train=False, download=True)
    elif args.dataset.lower() == "gtsrb":
        n_classes = 43
        img_shape = (3, 64, 64)
        augment_style = "gtsrb64"
        traindataset = datasets.GTSRB(args.data_path, split="train", download=True)
        testdataset = datasets.GTSRB(args.data_path, split="test", download=True)
    else:
        raise NotImplementedError("Dataset %s not available" % args.dataset)
    return traindataset, testdataset, img_shape, n_classes, augment_style
