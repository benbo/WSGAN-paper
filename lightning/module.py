# This code is in part based on code released by The PyTorch Lightning team,
# https://github.com/PyTorchLightning/pytorch-lightning,
# released under the Apache License, Version 2.0.

from argparse import ArgumentParser
import torch
import torch.nn.functional as F

from pytorch_lightning.core import LightningModule
import torchvision
from lightning.networks import EncoderLabelmodel
import itertools
from typing import Optional, List
import torchmetrics
from lightning.metrics import ARI, InceptionScores, inception_transforms
from lightning.lutils import get_linear_lr_schedule, get_networks_base

def soft_cross_entropy(pred, soft_targets):
    '''
    Cross entropy with soft targets (probabilities), while pred is assumed to be logits.
    '''
    predlogsoftmax = F.log_softmax(pred, dim=1)
    return torch.mean(torch.sum(- soft_targets * predlogsoftmax, 1))

class LightningGAN(LightningModule):
    def __init__(
        self,
        num_LFs,
        batch_size,
        img_shape: tuple = (1, 28, 28),
        architecture: str = "dc",
        dlr: float = 0.0004,
        glr: float = 0.0001,
        b1: float = 0.5,
        b2: float = 0.999,
        latent_dim: int = 100,
        smooth_labels: bool = True,
        flip_labels: float = 0.04,
        dsgd: bool = False,
        lrscheduler: bool = False,
        log_inception_score: bool = True,
    ):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.log_inception_score = log_inception_score
        self.architecture = architecture
        self.num_LFs = num_LFs
        self.batch_size = batch_size
        self.smooth_labels = smooth_labels
        self.flip_labels = flip_labels
        self.lrscheduler = lrscheduler
        self.dsgd = dsgd  # if True use sgd for discriminator, ADAM for generator

        self.dlr = dlr
        self.glr = glr
        self.b1 = b1
        self.b2 = b2

        # networks
        self._init_networks()

        # set up metrics for logging
        self.inception_scores = InceptionScores()
        self.max_isb_to_add = 50000 / batch_size
        self.isb_added = 0

        # create fixed z to store images after each epoch
        num_val = 20
        self.validation_z = torch.randn(num_val, latent_dim, device=self.device)

    def _init_networks(self):
        
        self.generator, self.discriminator = get_networks_base(
            self.architecture, self.latent_dim, 0, self.img_shape
        )

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("pl.GAN")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser

        parser.add_argument(
            "--dlr",
            type=float,
            default=0.0004,
            help="adam learning rate for discriminator",
        )
        parser.add_argument(
            "--glr", type=float, default=0.0001, help="adam learning rate for generator"
        )
        parser.add_argument(
            "--b1",
            type=float,
            default=0.5,
            help="adam: decay of first order momentum of gradient",
        )
        parser.add_argument(
            "--b2",
            type=float,
            default=0.999,
            help="adam: decay of second order momentum of gradient",
        )
        parser.add_argument(
            "--latent_dim",
            type=int,
            default=100,
            help="dimensionality of the latent space",
        )
        parser.add_argument(
            "--architecture",
            type=str,
            default="dc",
            help="GAN architecture",
        )
        parser.add_argument(
            "--smooth_labels", type=bool, default=True, help="Use label smoothing."
        )
        parser.add_argument(
            "--flip_labels",
            type=float,
            default=0.04,
            help="Flip small amount of labels according to this float.",
        )
        parser.add_argument(
            "--dsgd", type=bool, default=False, help="Use SGD for discriminator."
        )
        parser.add_argument(
            "--lrscheduler", type=bool, default=False, help="Use lr scheduler."
        )
        return parser_out

    def configure_optimizers(self):
        dlr = self.dlr
        glr = self.glr

        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=glr, betas=(b1, b2))

        if self.dsgd:
            opt_d = torch.optim.SGD(
                self.discriminator.parameters(), lr=dlr, momentum=0.9, weight_decay=1e-5
            )
        else:
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(), lr=dlr, betas=(b1, b2)
            )

        if self.lrscheduler:
            # TODO add epochs as hyper parameter
            g_schedule = get_linear_lr_schedule(glr, n_epochs=200, minlr=1e-7)
            g_scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt_g, g_schedule, verbose=True
            )

            d_schedule = get_linear_lr_schedule(dlr, n_epochs=200, minlr=1e-7)
            d_scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt_d, d_schedule, verbose=True
            )

            return [opt_d, opt_g], [g_scheduler, d_scheduler]
        else:
            return [opt_d, opt_g], []

    def generate(self, batch_size):
        # sample noise
        z = torch.randn(batch_size, self.latent_dim).to(self.device)

        # Generate a batch of images
        gen_imgs = self(z)
        return z, None, None, None, gen_imgs

    def forward(self, z, y=None):
        if y is None:
            return self.generator(z)
        else:
            return self.generator(z, y)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def base_gan_step(self, z, batch, optimizer_idx, disc_var=None):
        imgs = batch["imgs"]

        # train discriminator
        if optimizer_idx == 0:
            # Measure discriminator's ability to classify real from generated samples

            # alternate between real and face images with each batch
            # how well can it label as real?

            # label smoothing
            if self.smooth_labels:
                # bweteen 0.8 and 1.0 for real
                valid = 1.0 - 0.2 * torch.rand(imgs.size(0), 1)
                valid = valid.type_as(imgs)

                # between 0.0 and 0.2 for fake
                fake = 0.2 * torch.rand(imgs.size(0), 1)
                fake = fake.type_as(imgs)
            else:
                valid = torch.ones(imgs.size(0), 1)
                valid = valid.type_as(imgs)

                fake = torch.zeros(imgs.size(0), 1)
                fake = fake.type_as(imgs)

            if self.flip_labels > 0.0:
                flipidxs = torch.bernoulli(
                    torch.Tensor(imgs.shape[0]).fill_(self.flip_labels)
                ).bool()
                if torch.any(flipidxs):
                    valcopy = torch.clone(valid)
                    valid[flipidxs] = fake[flipidxs]
                    fake[flipidxs] = valcopy[flipidxs]

            # codes and validity are both logits, not normalized
            validity = self.discriminator(imgs)
            with torch.no_grad():
                fakeimgs = self(z) if disc_var is None else self(z, disc_var)

            real_loss = self.adversarial_loss(validity, valid)
            fake_loss = self.adversarial_loss(self.discriminator(fakeimgs), fake)

            # discriminator loss
            d_loss = (real_loss + fake_loss) / 2.0

            tqdm_dict = {
                "d_loss": d_loss,
                "real_loss": real_loss,
                "fake_loss": fake_loss,
            }
            self.log_dict(tqdm_dict)

            if self.log_inception_score:
                if self.current_epoch == 0:
                    if self.isb_added < self.max_isb_to_add:
                        self.isb_added += 1
                        self.inception_scores.update(inception_transforms(imgs), True)

            return d_loss

        # train generator
        if optimizer_idx == 1:
            # try to fool discriminator
            # label smoothing
            if self.smooth_labels:
                # bweteen 0.8 and 1.0
                valid = 1.0 - 0.2 * torch.rand(imgs.size(0), 1)
                valid = valid.type_as(imgs)
            else:
                valid = torch.ones(imgs.size(0), 1)
                valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            fakeimgs = self(z) if disc_var is None else self(z, disc_var)
            g_loss = self.adversarial_loss(self.discriminator(fakeimgs), valid)

            tqdm_dict = {"g_loss": g_loss}
            self.log_dict(tqdm_dict)
            return g_loss

        return None

    def training_step(self, batch, batch_idx, optimizer_idx):
        # sample noise
        imgs = batch["imgs"]
        z = torch.randn(imgs.shape[0], self.latent_dim).type_as(imgs)
        return self.base_gan_step(z, batch, optimizer_idx)

    def base_epoch_end(self):
        z = self.validation_z.type_as(self.generator.dummy.weight)
        # log sampled images
        with torch.no_grad():
            sample_imgs = self(z)
            if sample_imgs.shape[-1] < 64:
                sample_imgs = F.interpolate(
                    sample_imgs, size=64
                )  # resize to better visualize in tensorboard
        grid = torchvision.utils.make_grid(sample_imgs, normalize=True, nrow=10)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

        # inception score
        if self.log_inception_score:
            if self.current_epoch >= 0 and self.current_epoch % 10 == 0:
                # incpetion score should be computed. True images already added.
                # Add fake images now
                num_fake_to_add = sum(
                    [x.shape[0] for x in self.inception_scores.real_features]
                )
                count = 0

                with torch.no_grad():
                    while count < num_fake_to_add:
                        _, _, _, _, gen_imgs = self.generate(self.batch_size)
                        self.inception_scores.update(
                            inception_transforms(gen_imgs), False
                        )
                        count += gen_imgs.shape[0]
                scores = self.inception_scores.compute()
                for key, val in scores.items():
                    self.logger.experiment.add_scalar(
                        key, val, global_step=self.current_epoch
                    )
                # remove fake image features
                self.inception_scores.fake_features.clear()

    def on_train_epoch_end(self):
        self.base_epoch_end()


class LightningInfoGAN(LightningGAN):
    def __init__(
        self,
        num_LFs,
        batch_size,
        img_shape: tuple = (1, 28, 28),
        architecture: str = "infodc",
        dlr: float = 0.0004,
        glr: float = 0.0001,
        ilr: float = 0.00005,
        b1: float = 0.5,
        b2: float = 0.999,
        latent_dim: int = 100,
        latent_code_dim: int = 10,
        smooth_labels: bool = True,
        flip_labels: float = 0.04,
        dsgd: bool = False,
        class_balance: Optional[List] = None,
        lmda: float = 0.1,
        lrscheduler: bool = False,
        n_classes: Optional[int] = None,
        epoch_generate=-1,
        num_fake=2000,
        fake_data_store=None,
        log_inception_score: bool = True,
    ):
        self.ilr = ilr
        self.epoch_generate = epoch_generate
        self.num_fake = num_fake
        self.fake_data_store = fake_data_store
        self.latent_code_dim = latent_code_dim
        self.lmda = lmda
        if n_classes is None:
            n_classes = latent_code_dim
        self.n_classes = n_classes
        super().__init__(
            num_LFs,
            batch_size,
            img_shape,
            architecture,
            dlr,
            glr,
            b1,
            b2,
            latent_dim,
            smooth_labels,
            flip_labels,
            dsgd,
            lrscheduler,
            log_inception_score,
        )
        
        if class_balance is None:
            self.class_balance = (
                torch.ones(self.n_classes, device=self.device) / self.n_classes
            )
        else:
            self.class_balance = torch.tensor(
                class_balance, dtype=torch.float, device=self.device
            )
        self.class_balance = self.class_balance.repeat((batch_size, 1))

        discrete_z = torch.cat(
            (
                torch.arange(self.latent_code_dim, device=self.device),
                torch.arange(self.latent_code_dim, device=self.device),
            )
        )

        self.posterior_ari = ARI()

        # create fixed z to store images after each epoch
        num_val = latent_code_dim * 2
        num_val = 40 if num_val > 40 else num_val
        discrete_z = discrete_z[:num_val]
        noise = torch.randn(num_val, self.latent_dim, device=self.device)
        discrete_z_oh = F.one_hot(discrete_z, num_classes=self.latent_code_dim)

        self.validation_z = torch.cat((noise, discrete_z_oh.float()), 1)
        self.val_noise = noise
        self.val_discrete_z = discrete_z


    def _init_networks(self):
        # networks
        if self.architecture not in {"infodc", "infodc64"}:
            raise NotImplementedError(
                "%s networks not available for InfoGAN" % self.architecture
            )

        self.generator, self.discriminator = get_networks_base(
            self.architecture, self.latent_dim, self.latent_code_dim, self.img_shape
        )

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("pl.GAN")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser
        parser.add_argument(
            "--ilr", type=float, default=0.0001, help="adam learning rate for info term"
        )
        parser.add_argument(
            "--dlr",
            type=float,
            default=0.0004,
            help="adam learning rate for discriminator",
        )
        parser.add_argument(
            "--glr", type=float, default=0.0001, help="adam learning rate for generator"
        )
        parser.add_argument(
            "--b1",
            type=float,
            default=0.5,
            help="adam: decay of first order momentum of gradient",
        )
        parser.add_argument(
            "--b2",
            type=float,
            default=0.999,
            help="adam: decay of second order momentum of gradient",
        )
        parser.add_argument(
            "--latent_dim",
            type=int,
            default=100,
            help="dimensionality of the latent space",
        )
        parser.add_argument(
            "--architecture",
            type=str,
            default="infodc",
            help="GAN architecture, e.g. infodc",
        )
        parser.add_argument(
            "--smooth_labels", type=bool, default=True, help="Use label smoothing."
        )
        parser.add_argument(
            "--flip_labels",
            type=float,
            default=0.04,
            help="Flip small amount of labels according to this float.",
        )
        parser.add_argument(
            "--dsgd", type=bool, default=False, help="Use SGD for discriminator."
        )
        parser.add_argument(
            "--lrscheduler", type=bool, default=False, help="Use lr scheduler."
        )
        parser.add_argument(
            "--class_balance",
            nargs="+",
            default=None,
            type=float,
            help="floats of class  balance",
            required=False,
        )
        parser.add_argument(
            "--latent_code_dim",
            type=int,
            default=10,
            help="dimensionality of the discrete code space",
        )
        parser.add_argument(
            "--lmda", type=float, default=0.1, help="lambda for discrete code info loss"
        )
        parser.add_argument(
            "--num_fake",
            type=int,
            default=2000,
            help="number of fake images to generate at the end",
        )
        parser.add_argument(
            "--epoch_generate",
            default=-1,
            type=int,
            help="Epoch at which to generate and save fake data",
            required=False,
        )
        return parser_out

    def configure_optimizers(self):
        # different learning rate paramaters for label model and other models
        dlr = self.dlr
        glr = self.glr
        ilr = self.ilr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=glr, betas=(b1, b2))
        if self.dsgd:
            opt_d = torch.optim.SGD(
                self.discriminator.parameters(), lr=dlr, momentum=0.9, weight_decay=1e-5
            )
        else:
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(), lr=dlr, betas=(b1, b2)
            )

        opt_info = torch.optim.Adam(
            itertools.chain(
                self.discriminator.parameters(), self.generator.parameters()
            ),
            lr=ilr,
            betas=(b1, b2),
        )
        if self.lrscheduler:
            # TODO pass epochs as argument if lr schedulers are used
            g_schedule = get_linear_lr_schedule(glr, n_epochs=200, minlr=1e-7)
            g_scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt_g, g_schedule, verbose=True
            )

            d_schedule = get_linear_lr_schedule(dlr, n_epochs=200, minlr=1e-7)
            d_scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt_d, d_schedule, verbose=True
            )

            i_schedule = get_linear_lr_schedule(ilr, n_epochs=200, minlr=1e-7)
            i_scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt_info, i_schedule, verbose=True
            )

            return [opt_d, opt_g, opt_info], [g_scheduler, d_scheduler, i_scheduler]
        else:
            return [opt_d, opt_g, opt_info], []

    def get_prob_labels_q(self, imgs):
        pred_code = self.discriminator.predict_code(imgs.to(self.device))
        return pred_code

    @staticmethod
    def code_loss(y_hat, y):
        return F.cross_entropy(y_hat, y)

    def generate(self, batch_size):
        # sample noise
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)

        discrete_z = (
            torch.multinomial(self.class_balance[:batch_size, :], 1)
            .squeeze()
            .to(self.device)
        )
        discrete_z_oh = F.one_hot(discrete_z, num_classes=self.latent_code_dim)

        z = torch.cat((noise, discrete_z_oh.float()), 1).to(self.device)

        # Generate a batch of images
        gen_imgs = self(z)
        return z, discrete_z, discrete_z_oh, None, gen_imgs

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs = batch["imgs"]
        # sample noise
        noise = torch.randn(imgs.shape[0], self.latent_dim, device=self.device)

        discrete_z = (
            torch.multinomial(self.class_balance[: imgs.shape[0], :], 1)
            .squeeze()
            .to(self.device)
        )
        discrete_z_oh = F.one_hot(discrete_z, num_classes=self.latent_code_dim)
        z = torch.cat((noise, discrete_z_oh.float()), 1).type_as(imgs)

        if optimizer_idx == 2:
            # info loss
            discrete_z = discrete_z.to(imgs.device)
            # try to fool discriminator
            # label smoothing
            if self.smooth_labels:
                # bweteen 0.8 and 1.0
                valid = -0.2 * torch.rand(imgs.size(0), 1) + 1.0
                valid = valid.type_as(imgs)
            else:
                valid = torch.ones(imgs.size(0), 1)
                valid = valid.type_as(imgs)

            fakeimgs = self(z)
            pred_code = self.discriminator.predict_code(fakeimgs)
            info_loss = self.code_loss(pred_code, discrete_z)

            # log performance of label estimate
            with torch.no_grad():
                pred_code_real = self.discriminator.predict_code(imgs)
                _, pred_code_crisp = torch.max(pred_code_real, 1)
                self.posterior_ari.update(pred_code_crisp, batch["y"])
                self.log(
                    "train_posterior_ari",
                    self.posterior_ari,
                    on_step=False,
                    on_epoch=True,
                )

            tqdm_dict = {"info_loss": info_loss}
            self.log_dict(tqdm_dict)
            return self.lmda * info_loss
        else:
            z = torch.cat((noise, discrete_z_oh.float()), 1).type_as(imgs)
            return self.base_gan_step(z, batch, optimizer_idx)

    def on_train_epoch_end(self):
        if self.current_epoch == self.epoch_generate:
            if self.num_fake > 0:
                counter = 0
                fake_images = []
                while counter < self.num_fake:
                    # generate fake data
                    bsize = (
                        self.batch_size
                        if self.num_fake - counter > self.batch_size
                        else int(self.num_fake - counter)
                    )
                    with torch.no_grad():
                        _, _, _, _, gen_imgs = self.generate(bsize)
                    fake_images.append(gen_imgs.cpu())
                    counter += bsize

                fake_images = torch.cat(fake_images)
                print("saving fake data: %s" % self.fake_data_store)
                torch.save([fake_images, None, None], self.fake_data_store)

        self.base_epoch_end()

    def base_epoch_end(self):
        z = self.validation_z.type_as(self.generator.dummy.weight)
        # log sampled images
        with torch.no_grad():
            sample_imgs = self(z)
            if sample_imgs.shape[-1] < 64:
                sample_imgs = F.interpolate(
                    sample_imgs, size=64
                )  # resize to better visualize in tensorboard
        grid = torchvision.utils.make_grid(sample_imgs, normalize=True, nrow=10)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

        # inception score
        if self.log_inception_score:
            if self.current_epoch >= 0 and self.current_epoch % 10 == 0:
                # incpetion score should be computed. True images already added.
                # Add fake images now
                num_fake_to_add = sum(
                    [x.shape[0] for x in self.inception_scores.real_features]
                )
                count = 0
                with torch.no_grad():
                    while count < num_fake_to_add:
                        _, _, _, _, gen_imgs = self.generate(self.batch_size)
                        self.inception_scores.update(
                            inception_transforms(gen_imgs), False
                        )
                        count += gen_imgs.shape[0]
                scores = self.inception_scores.compute()
                for key, val in scores.items():
                    self.logger.experiment.add_scalar(
                        key, val, global_step=self.current_epoch
                    )
                # remove fake image features
                self.inception_scores.fake_features.clear()


class GANLabelModel(LightningGAN):
    def __init__(
        self,
        num_LFs,
        batch_size,
        img_shape: tuple = (1, 28, 28),
        architecture: str = "infodc",
        dlr: float = 0.0004,
        glr: float = 0.0001,
        ilr: float = 0.0001,
        lmlr: float = 0.00008,
        b1: float = 0.5,
        b2: float = 0.999,
        latent_dim: int = 100,
        latent_code_dim: int = 10,
        smooth_labels: bool = True,
        flip_labels: float = 0.04,
        dsgd: bool = False,
        class_balance: Optional[List] = None,
        lmda: float = 0.1,
        lrscheduler: bool = False,
        n_classes: Optional[int] = None,
        encodertype: str = "encoderX",
        freeze_features=False,
        decaylossterm=1.0,
        epoch_generate=-1,
        num_fake=2000,
        fake_data_store=None,
        log_inception_score: bool = True,
        decaylossparam: float = 1.5,
        burnin=False,
    ):
        self.burnin = burnin
        self.decaylossparam = decaylossparam
        self.epoch_generate = epoch_generate
        self.num_fake = num_fake
        self.fake_data_store = fake_data_store
        self.ilr = ilr
        self.lmlr = lmlr
        self.latent_code_dim = latent_code_dim
        self.lmda = lmda
        if n_classes is None:
            n_classes = latent_code_dim
        self.n_classes = n_classes
        super().__init__(
            num_LFs,
            batch_size,
            img_shape,
            architecture,
            dlr,
            glr,
            b1,
            b2,
            latent_dim,
            smooth_labels,
            flip_labels,
            dsgd,
            lrscheduler,
            log_inception_score,
        )

        self.freeze_features = freeze_features
        if img_shape[1] == 28:
            imgfeaturesize = 512 * 9
        elif img_shape[1] == 32:
            imgfeaturesize = 512 * 16
        elif img_shape[1] == 64:
            imgfeaturesize = 512 * 16
        else:
            raise NotImplementedError(
                "Large image size not handled by this implementation, check the StyleWSGAN repository"
            )

        self.encoderlabelmodel = EncoderLabelmodel(
            num_LFs,
            imgfeaturesize,
            self.n_classes,
            [256, 128, 64],
            modeltype=encodertype,
        )

        self.fone = torch.nn.Linear(self.n_classes, self.n_classes)
        self.ftwo = torch.nn.Linear(self.n_classes, self.n_classes)
        # MSE loss for decay term
        self.decaylossterm = decaylossterm
        self.mse_loss = torch.nn.MSELoss()

        # Use class balance to sample latent code?
        # Not used currently, needs to be adaptive during epochs to make sense.
        if class_balance is None:
            self.class_balance = (
                torch.ones(self.n_classes, device=self.device) / self.n_classes
            )
        else:
            self.class_balance = torch.tensor(
                class_balance, dtype=torch.float, device=self.device
            )
        self.class_balance = self.class_balance.repeat((batch_size, 1))

        discrete_z = torch.cat(
            (
                torch.arange(self.latent_code_dim, device=self.device),
                torch.arange(self.latent_code_dim, device=self.device),
            )
        )

        self.posterior_ari = ARI()
        if self.n_classes > 2:
            task = "multiclass"
            self.posterior_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=self.n_classes, average='micro')
        else:
            task = "binary"
            self.posterior_accuracy = torchmetrics.classification.BinaryAccuracy()
        # https://torchmetrics.readthedocs.io/en/latest/references/modules.html#averageprecision
        self.posterior_ap_weighted = torchmetrics.AveragePrecision(task=task,
            num_classes=self.n_classes, average="weighted"
        )
        self.posterior_fone = torchmetrics.F1Score(task=task,
            num_classes=self.n_classes, average="weighted"
        )
        self.posterior_rec = torchmetrics.Recall(task=task,
            num_classes=self.n_classes, average="weighted"
        )
        self.posterior_prec = torchmetrics.Precision(task=task,
            num_classes=self.n_classes, average="weighted"
        )

        # create fixed z to store images after each epoch
        num_val = latent_code_dim * 2
        num_val = 40 if num_val > 40 else num_val
        discrete_z = discrete_z[:num_val]
        noise = torch.randn(num_val, self.latent_dim, device=self.device)
        discrete_z_oh = F.one_hot(discrete_z, num_classes=self.latent_code_dim)
        self.validation_z = torch.cat((noise, discrete_z_oh.float()), 1)
        self.val_noise = noise
        self.val_discrete_z = discrete_z.to(self.device)

    def _init_networks(self):
        # networks
        if self.architecture not in {"infodc", "infodc64"}:
            raise NotImplementedError(
                "%s networks not available for GANLabelModel" % self.architecture
            )
        self.generator, self.discriminator = get_networks_base(
            self.architecture, self.latent_dim, self.latent_code_dim, self.img_shape
        )

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("pl.GAN")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser
        parser.add_argument(
            "--ilr", type=float, default=0.0001, help="adam learning rate for info term"
        )
        parser.add_argument(
            "--dlr",
            type=float,
            default=0.0004,
            help="adam learning rate for discriminator",
        )
        parser.add_argument(
            "--glr", type=float, default=0.0001, help="adam learning rate for generator"
        )
        parser.add_argument(
            "--lmlr",
            type=float,
            default=0.00008,
            help="adam learning rate for labelmodel",
        )
        parser.add_argument(
            "--b1",
            type=float,
            default=0.5,
            help="adam: decay of first order momentum of gradient",
        )
        parser.add_argument(
            "--b2",
            type=float,
            default=0.999,
            help="adam: decay of second order momentum of gradient",
        )
        parser.add_argument(
            "--latent_dim",
            type=int,
            default=100,
            help="dimensionality of the latent space",
        )
        parser.add_argument(
            "--architecture",
            type=str,
            default="infodc",
            help="GAN architecture",
        )
        parser.add_argument(
            "--not_smooth_labels",
            default=True,
            action="store_false",
            help="Do not use label smoothing.",
        )
        parser.add_argument(
            "--flip_labels",
            type=float,
            default=0.04,
            help="Flip small amount of labels according to this float.",
        )
        parser.add_argument(
            "--dsgd",
            default=False,
            action="store_true",
            help="Use SGD for discriminator.",
        )
        parser.add_argument(
            "--lrscheduler",
            default=False,
            action="store_true",
            help="Use lr scheduler.",
        )
        parser.add_argument(
            "--class_balance",
            nargs="+",
            default=None,
            type=float,
            help="floats of class  balance",
            required=False,
        )
        parser.add_argument(
            "--latent_code_dim",
            type=int,
            default=10,
            help="dimensionality of the discrete code space",
        )
        parser.add_argument(
            "--lmda", type=float, default=0.1, help="lambda for discrete code info loss"
        )
        parser.add_argument(
            "--decaylossterm",
            default=1.0,
            type=float,
            help="Weight for decay loss term. Keeps initial label model weights equal, slowly decays this term per epoch",
            required=False,
        )
        parser.add_argument(
            "--num_fake",
            type=int,
            default=2000,
            help="number of fake images to generate at the end",
        )
        parser.add_argument(
            "--epoch_generate",
            default=-1,
            type=int,
            help="Epoch at which to generate and save fake data",
            required=False,
        )
        parser.add_argument(
            "--decaylossparam",
            default=1.5,
            type=float,
            help="Parameter for decay loss term. Higher parameter -> term disappears faster",
            required=False,
        )

        return parser_out

    def configure_optimizers(self):
        # different learning rate paramaters for label model and other models
        dlr = self.dlr
        glr = self.glr
        ilr = self.ilr
        lmlr = self.lmlr

        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=glr, betas=(b1, b2))
        if self.dsgd:
            opt_d = torch.optim.SGD(
                self.discriminator.parameters(), lr=dlr, momentum=0.9, weight_decay=1e-5
            )
        else:
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(), lr=dlr, betas=(b1, b2)
            )

        opt_info = torch.optim.Adam(
            itertools.chain(
                self.discriminator.parameters(), self.generator.parameters()
            ),
            lr=ilr,
            betas=(b1, b2),
        )

        opt_lf = torch.optim.Adam(
            itertools.chain(
                self.discriminator.parameters(),
                self.encoderlabelmodel.parameters(),
                self.fone.parameters(),
                self.ftwo.parameters(),
            ),
            lr=lmlr,
            betas=(b1, b2),
        )

        if self.lrscheduler:
            # TODO add epochs as hyper parameter
            g_schedule = get_linear_lr_schedule(glr, n_epochs=200, minlr=glr * 0.01)
            g_scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt_g, g_schedule, verbose=True
            )

            d_schedule = get_linear_lr_schedule(dlr, n_epochs=200, minlr=dlr * 0.01)
            d_scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt_d, d_schedule, verbose=True
            )

            i_schedule = get_linear_lr_schedule(ilr, n_epochs=200, minlr=ilr * 0.01)
            i_scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt_info, i_schedule, verbose=True
            )

            lf_schedule = get_linear_lr_schedule(lmlr, n_epochs=200, minlr=lmlr * 0.01)
            lf_scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt_lf, lf_schedule, verbose=True
            )

            return (
                [opt_d, opt_g, opt_info, opt_lf],
                [g_scheduler, d_scheduler, i_scheduler, lf_scheduler],
            )
        else:

            return [opt_d, opt_g, opt_info, opt_lf], []
        

    def map_code_to_label(self, code):
        with torch.no_grad():
            yestimate = F.softmax(self.fone(code.to(self.device)), dim=1)
        return yestimate

    def get_prob_labels_q(self, imgs):
        with torch.no_grad():
            pred_code = self.discriminator.predict_code(imgs.to(self.device))
            yestimate = F.softmax(self.fone(F.softmax(pred_code, dim=1)), dim=1)
        return yestimate

    def get_prob_labels(self, imgs, lambda_onehot):
        with torch.no_grad():
            features = self.discriminator.get_features(imgs.to(self.device))
            yestimate = self.encoderlabelmodel(features, lambda_onehot.to(self.device))
        return yestimate

    @staticmethod
    def code_loss(y_hat, y):
        return F.cross_entropy(y_hat, y)

    def generate(self, batch_size):
        # sample noise
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)

        discrete_z = (
            torch.multinomial(self.class_balance[:batch_size, :], 1)
            .squeeze()
            .to(self.device)
        )
        discrete_z_oh = F.one_hot(discrete_z, num_classes=self.latent_code_dim)

        z = torch.cat((noise, discrete_z_oh.float()), 1)

        # Generate a batch of images
        gen_imgs = self(z)
        return z, discrete_z, discrete_z_oh, None, gen_imgs

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs = batch["imgs"]
        # sample noise
        noise = torch.randn(imgs.shape[0], self.latent_dim, device=self.device)

        discrete_z = (
            torch.multinomial(self.class_balance[: imgs.shape[0], :], 1)
            .squeeze()
            .to(self.device)
        )
        discrete_z_oh = F.one_hot(discrete_z, num_classes=self.latent_code_dim)

        if self.burnin:
            if optimizer_idx >= 2:
                # only do basic gan steps uring burnin period
                if self.current_epoch <= 10:
                    # skip this step
                    return None

        if optimizer_idx == 2:
            # info loss
            discrete_z = discrete_z.to(imgs.device)
            # try to fool discriminator
            z = torch.cat((noise, discrete_z_oh.float()), 1).type_as(imgs)
            fakeimgs = self(z)
            pred_code = self.discriminator.predict_code(fakeimgs)
            info_loss = self.code_loss(pred_code, discrete_z)

            self.log_dict({"info_loss": info_loss})
            return self.lmda * info_loss

        elif optimizer_idx == 3:
            # LF alignment loss
            filteridx = batch["filteridxs"]
            filtersum = filteridx.sum()
            if filtersum > 0:
                lambdas = batch["oh_lfs"][filteridx]
                imgs_filtered = imgs[filteridx]

                y_sub = batch["y"][filteridx]  # real label, for logging only
                if imgs_filtered.dim() == 3:
                    lambdas = lambdas.unsqueeze(0)
                    imgs_filtered = imgs_filtered.unsqueeze(0)
                    y_sub = y_sub.unsqueeze(0)

                pred_code, features = self.discriminator.get_code_features(
                    imgs_filtered, freeze=self.freeze_features
                )

                latent_GAN_code = F.softmax(pred_code, dim=1)  # latent code

                # only use points where at least one LF votes

                yestimate, accs = self.encoderlabelmodel(
                    features.detach(), lambdas, get_accuracies=True
                )

                lfloss = soft_cross_entropy(
                    self.fone(latent_GAN_code), yestimate.detach()
                )
                lfloss += soft_cross_entropy(
                    self.ftwo(yestimate), latent_GAN_code.detach()
                )
                tqdm_dict = {"lfloss": lfloss}
                if self.decaylossterm>0.0:
                    # force label model accuracy weights to stay close to 0.5 at early epochs
                    # This means we keep a label model with equal weighted accuracies at the beginning
                    # which and enables the discrete GAN variables to catch up while in its early stages
                    decayloss = (
                        self.n_classes
                        / (self.current_epoch * self.decaylossparam + 1.0)
                        * self.mse_loss(accs, torch.ones_like(accs) * 0.5)
                        / self.num_LFs
                    )
                    tqdm_dict["decayloss"] = self.decaylossterm*decayloss
                    acc_d = accs.detach()
                    tqdm_dict["mean_acc_param"] = acc_d.mean()
                    tqdm_dict["stddev_acc_param"] = acc_d.std()
                    lfloss += self.decaylossterm*decayloss

                self.log_dict(tqdm_dict)
                # log performance of label estimate
                with torch.no_grad():
                    # log GAN code ARI on non-abstains
                    _, pred_code_crisp = torch.max(pred_code, 1)
                    self.posterior_ari.update(pred_code_crisp, y_sub)
                    self.log(
                        "train_posterior_ari_gan_code",
                        self.posterior_ari,
                        on_step=False,
                        on_epoch=True,
                    )

                    # log label model estimate performance on non-abstains
                    _, pred_label_crisp = torch.max(yestimate, 1)
                    self.posterior_accuracy(pred_label_crisp, y_sub)
                    self.log(
                        "train_posterior_acc",
                        self.posterior_accuracy,
                        on_step=False,
                        on_epoch=True,
                    )

                    self.posterior_ap_weighted(yestimate, y_sub)
                    self.log(
                        "train_posterior_averageprecision_weighted",
                        self.posterior_ap_weighted,
                        on_step=False,
                        on_epoch=True,
                    )

                    self.posterior_fone(pred_label_crisp, y_sub)
                    self.log(
                        "train_posterior_f1",
                        self.posterior_fone,
                        on_step=False,
                        on_epoch=True,
                    )

                    self.posterior_rec(pred_label_crisp, y_sub)
                    self.log(
                        "train_posterior_recall",
                        self.posterior_rec,
                        on_step=False,
                        on_epoch=True,
                    )

                    self.posterior_prec(pred_label_crisp, y_sub)
                    self.log(
                        "train_posterior_precision",
                        self.posterior_prec,
                        on_step=False,
                        on_epoch=True,
                    )
                return lfloss
            else:
                return None

        else:
            z = torch.cat((noise, discrete_z_oh.float()), 1).type_as(imgs)
            return self.base_gan_step(z, batch, optimizer_idx)

    def on_train_epoch_end(self):
        if self.current_epoch == self.epoch_generate:
            if self.num_fake > 0:
                yhat = torch.zeros(
                    (self.num_fake, self.n_classes), requires_grad=False, device="cpu"
                )
                yhat_code = torch.zeros(
                    (self.num_fake, self.n_classes), requires_grad=False, device="cpu"
                )
                counter = 0
                fake_images = []
                while counter < self.num_fake:
                    # generate fake data and pseudlabels
                    bsize = (
                        self.batch_size
                        if self.num_fake - counter > self.batch_size
                        else int(self.num_fake - counter)
                    )
                    with torch.no_grad():
                        _, discrete_var, discrete_z_oh, _, gen_imgs = self.generate(
                            bsize
                        )
                        # Predict latent codes on training images
                        pred_yhat = self.get_prob_labels_q(gen_imgs)
                        code_mapped = self.map_code_to_label(discrete_z_oh.float())

                    fake_images.append(gen_imgs.cpu())
                    yhat[counter : counter + bsize] = pred_yhat.to("cpu")
                    yhat_code[counter : counter + bsize] = code_mapped.to("cpu")
                    counter += bsize

                fake_images = torch.cat(fake_images)
                print("saving fake data: %s" % self.fake_data_store)
                torch.save([fake_images, yhat, yhat_code], self.fake_data_store)

        self.base_epoch_end()

    def base_epoch_end(self):
        # log sampled images
        with torch.no_grad():
            z = self.validation_z.type_as(self.generator.dummy.weight)
            noise = self.val_noise.to(self.device)
            dz = self.val_discrete_z.to(self.device)
            sample_imgs = self(z)
            if sample_imgs.shape[-1] < 64:
                sample_imgs = F.interpolate(
                    sample_imgs, size=64
                )  # resize to better visualize in tensorboard
        grid = torchvision.utils.make_grid(sample_imgs, normalize=True, nrow=10)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

        # inception score
        if self.log_inception_score:
            if self.current_epoch >= 0 and self.current_epoch % 10 == 0:
                # incpetion score should be computed. True images already added.
                # Add fake images now
                num_fake_to_add = sum(
                    [x.shape[0] for x in self.inception_scores.real_features]
                )
                count = 0

                with torch.no_grad():
                    while count < num_fake_to_add:
                        _, _, _, _, gen_imgs = self.generate(self.batch_size)
                        self.inception_scores.update(
                            inception_transforms(gen_imgs), False
                        )
                        count += gen_imgs.shape[0]
                scores = self.inception_scores.compute()
                for key, val in scores.items():
                    self.logger.experiment.add_scalar(
                        key, val, global_step=self.current_epoch
                    )
                # remove fake image features
                self.inception_scores.fake_features.clear()
