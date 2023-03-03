import os
from datetime import datetime
from pytorch_lightning import loggers as pl_loggers
from lightning.module import LightningInfoGAN, GANLabelModel
from lightning.datamodule import LFDataModule, JSONImageDataset
from lightning.lutils import (
    get_datasets,
    load_domainnet_lfs,
    load_awa_lfs,
    create_L_ind
)
from argparse import ArgumentParser, Namespace
from pytorch_lightning.trainer import Trainer
import numpy as np
import torch
import warnings


def main(args: Namespace) -> None:
    storedir = args.storedir
    os.makedirs(storedir, exist_ok=True)
    os.makedirs(os.path.join(storedir, "wsganlogs"), exist_ok=True)
    os.makedirs(os.path.join(storedir, "lfmodel"), exist_ok=True)
    max_n_fake = int(max(args.fake_data_sizes))
    # name for subfolder to save to
    datasetsavename = args.dataset
    # ------------------------
    # Determine dataset, load data and LF outputs
    # ------------------------
    train_idxs = None
    drop_last = args.droplast
    architecture = args.architecture
    latent_dim = args.latent_dim
    if args.dataset.lower() in ["awa2", "awa", "domainnet"]:
        if args.imgsize == 32:
            img_shape = (3, 32, 32)
            augment_style = "cifar"
        elif args.imgsize == 64:
            img_shape = (3, 64, 64)
            augment_style = "cifar64"
            architecture += "64"
        else:
            raise NotImplementedError("img size selected not implemented. Check StyleWSGAN repository instead.")

        if args.dataset.lower() in "awa2":
            img_root = args.data_path  # path to images
            dset_lfs = args.lffname # labeling function file, saved as json
            if not dset_lfs.endswith(".json"):
                dset_lfs = os.path.join(dset_lfs, "train.json")
            datasetsavename = "awa2"
            n_classes = 10
            Lambdas, Ytrue = load_awa_lfs(dset_lfs)
        elif args.dataset.lower() == "domainnet":
            img_root = args.data_path  # path to images
            dset_lfs = args.lffname # labeling function file, saved as json
            if not dset_lfs.endswith(".json"):
                dset_lfs = os.path.join(dset_lfs, "train.json")
            datasetsavename = "domainnet"
            n_classes = 10
            drop_last = True  # number of train images leads to last batch of size 1
            # load train LFs
            Lambdas, Ytrue = load_domainnet_lfs(dset_lfs, applythreshold=False)
        else:
            raise NotImplementedError("dataset not found")

        num_LFs = Lambdas.shape[1]

        # set up datasets
        trainset_sub = JSONImageDataset(
            jsonpath=dset_lfs,
            img_root=img_root,
            transform=None,
            size=img_shape[1:],
        )
    else:
        # ------------------------
        # pytorch vision dataset
        # ------------------------

        traindataset, testdataset, img_shape, n_classes, augment_style = get_datasets(
            args, basetransforms=False
        )

        if args.dataset.lower() == "gtsrb":
            architecture += "64"

        # ------------------------
        # Load fixed LFs
        # ------------------------
        print("loading fixed LFs from %s" % args.lffname)
        # load precomputed LFs and training indices
        # Lambedas is the LF output matrix
        try:
            (
                train_idxs,
                val_idxs,
                Lambdas,
                LF_accuracies,
                LF_propensity,
                LF_labels,
                ValLambdas,
            ) = torch.load(args.lffname, map_location=lambda storage, loc: storage)
        except ValueError:
            (
                train_idxs,
                val_idxs,
                Lambdas,
                LF_accuracies,
                LF_propensity,
                LF_labels,
            ) = torch.load(args.lffname, map_location=lambda storage, loc: storage)

        # choose subset of LFs.
        # Loaded LFs were already randomized, so choose by increasing index
        max_lfs = args.numlfs
        tmp_num_lfs = Lambdas.shape[1]
        if tmp_num_lfs > max_lfs:
            lfidxs = np.arange(max_lfs)
            print("Chosen LF indexes:\n", lfidxs)
            Lambdas = Lambdas[:, lfidxs]
            if isinstance(LF_accuracies, list):
                LF_accuracies = np.array(LF_accuracies)
            LF_accuracies = LF_accuracies[lfidxs]
        if tmp_num_lfs < max_lfs:
            warnings.warn(
                "WARNING: max number of LFs chosen greater than available number of LFs"
            )

        num_LFs = Lambdas.shape[1]
        datasetsavename = datasetsavename + "_%dlfs" % num_LFs
        trainset_sub = torch.utils.data.Subset(traindataset, train_idxs)

    # create a torch tensor of the LF outputs
    if Lambdas is not None:
        lambda_tensor = torch.tensor(Lambdas, requires_grad=False).float()
        # set up the indicator vector of non-abstains (i.e. at least one LF vote available for sample)
        full_filter_idx = lambda_tensor.sum(1) != 0
        print("Num samples with non-abstains:", full_filter_idx.sum())
        # create onehot representation of LFs
        lambda_oh = create_L_ind(lambda_tensor, n_classes)
    else:
        full_filter_idx = None
        lambda_oh = None

    # ------------------------
    # Determine where to save random fake images (if any) at the end of training 
    # ------------------------
    fake_data_store = None
    if max_n_fake > 0:
        curr_dt = datetime.now()
        timestamp = int(round(curr_dt.timestamp()))
        if args.save_suffix:
            fpath = "wsganlogs/fakedata/%s/%s/%d" % (
                datasetsavename,
                args.save_suffix,
                timestamp,
            )
        else:
            fpath = "wsganlogs/fakedata/%s/%d" % (datasetsavename, timestamp)

        fpath = os.path.join(storedir, fpath)
        os.makedirs(fpath, exist_ok=True)
        fake_data_store = os.path.join(fpath, "fake_data.pt")
        
    # ------------------------
    # INIT DATA MODULE
    # ------------------------
    dm = LFDataModule(
        trainset_sub,
        full_filter_idx,
        trainlambda_oh=lambda_oh,
        trainlambda_bin=None,
        testdataset=None,
        lfmodelout=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        augment_style=augment_style,
        drop_last=drop_last,
    )

    # ------------------------
    # INIT LIGHTNING MODULE
    # ------------------------
    if args.whichmodule == "LightningInfoGAN" or args.whichmodule == "InfoGAN":
        model = LightningInfoGAN(
            num_LFs,
            args.batch_size,
            img_shape=img_shape,
            architecture=architecture,
            dlr=args.dlr,
            glr=args.glr,
            ilr=args.ilr,
            b1=args.b1,
            b2=args.b2,
            latent_dim=latent_dim,
            latent_code_dim=n_classes,
            class_balance=args.class_balance,
            num_fake=max_n_fake,
            epoch_generate=args.max_epochs - 1,
            fake_data_store=fake_data_store,
        )
    elif args.whichmodule == "GANLabelModel" or args.whichmodule == "WSGAN":
        # The WSGAN model using a simple DCGAN architecture
        model = GANLabelModel(
            num_LFs,
            args.batch_size,
            img_shape=img_shape,
            architecture=architecture,
            dlr=args.dlr,
            glr=args.glr,
            lmlr=args.lmlr,
            ilr=args.ilr,
            b1=args.b1,
            b2=args.b2,
            latent_dim=latent_dim,
            latent_code_dim=n_classes,
            encodertype=args.ganenctype,
            class_balance=args.class_balance,
            n_classes=n_classes,
            decaylossterm=args.decaylossterm,
            num_fake=max_n_fake,
            epoch_generate=args.max_epochs - 1,
            fake_data_store=fake_data_store,
            decaylossparam=args.decaylossparam,
        )
    else:
        raise NotImplementedError()


    # ------------------------
    # INIT TRAINER
    # ------------------------
    # set up logging
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.tensorboard.html
    if args.save_suffix:
        foldername = "wsganlogs/%s/%s/" % (
            datasetsavename,
            args.save_suffix,
        )
    else:
        foldername = "wsganlogs/%s/" % datasetsavename

    logdir = os.path.join(storedir, foldername)
    os.makedirs(logdir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(
        logdir, name=args.whichmodule
    )
    # set up trainer
    trainer = Trainer(
        gradient_clip_val=0.5,
        accelerator='gpu', 
        devices=args.gpus, # gpus IDs to use. NOTE: code was only tested on a single GPU.
        logger=tb_logger,
        max_epochs=args.max_epochs,
    )

    # ------------------------
    # START GAN Model training
    # ------------------------
    trainer.fit(model, dm)
    

if __name__ == "__main__":
    # python lightning/main.py --data_path /home/scratch/benediktb/data
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST", help="dataset to load")
    parser.add_argument(
        "--data_path", type=str, default=os.getcwd(), help="dataset to load"
    )
    parser.add_argument(
        "--lffname",
        type=str,
        default=None,
        required=False,
        help="path to precomputed LFs and training indexes",
    )
    parser.add_argument(
        "--whichmodule",
        type=str,
        default="GANLabelModel",
        help="Which WSGAN model to run: InfoGAN or GANLabelModel",
    )

    parser.add_argument(
        "--imgsize",
        type=int,
        default=32,
        help="Size of images. 32x32 or 64x64. Will only be changed for AwA2 and Domainnet",
    )

    parser.add_argument(
        "--numlfs", type=int, default=40, help="number of labeling functions (LFs) to load from LF dataset"
    )

    parser.add_argument(
        "--droplast", default=False, action="store_true", help="Drop last uneven batch"
    )


    # ------------------------
    # Add DataLoader args
    parser = LFDataModule.add_argparse_args(parser)

    # Add model specific args
    parser = GANLabelModel.add_argparse_args(parser)

    ##########################
    # GAN trainer args
    ##########################
    parser.add_argument("--gpus", nargs="+", type=int, help="GPU ids", required=True)
    parser.add_argument("--max_epochs", type=int, help="Number of training epochs", default=150)

    ##########################
    # GAN labelmodel arguments
    ##########################
    parser.add_argument(
        "--ganenctype",
        type=str,
        default="encoderX",
        help="Type of encoder of GAN label model, one of: encoder, encoderX, encoderL, vector. Recommended: encoderX",
    )


    #########################
    # path parameters
    #########################
    parser.add_argument(
        "--storedir",
        type=str,
        required=True,
        help="path to save logs, fake images , and checkpoints to",
    )

    parser.add_argument(
        "--save_suffix",
        type=str,
        default="",
        help="Suffix to append to logging folder for results",
    )

    ################################
    # fake data parameters
    ################################
    parser.add_argument(
        "-fk",
        "--fake_data_sizes",
        type=int,
        nargs="+",
        default=[0, 10000, 20000, 30000],
        help="Fake datapoints to generate. Use like: -fk 0 1000 2000",
    )


    # Parse all arguments
    args = parser.parse_args()
    main(args)
