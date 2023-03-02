import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Optional

class DCGeneratorThree(nn.Module):
    def __init__(self, latent_dim: int = 100, ngf=64, img_shape: tuple = (3, 32, 32)):
        super(DCGeneratorThree, self).__init__()

        # dummy so we can get device and type
        self.dummy = nn.Linear(1, 1)

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, img_shape[0], 3, 1, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )
        self.example_input_array = torch.rand(img_shape)

    def forward(self, x):
        return self.model(x.unsqueeze(2).unsqueeze(3))


class DCDiscriminatorThree(nn.Module):
    def __init__(self, img_shape, dropout=0.2):
        super(DCDiscriminatorThree, self).__init__()

        nc = img_shape[0]
        ndf = 64
        self.model = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 1 x 32
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
        )
        self.example_input_array = torch.rand(img_shape)

    def forward(self, img):
        out = self.model(img)  # batch x 1 x 1 x 1
        return out.view(out.shape[:2])  # batch x 1


class DCGenerator64(nn.Module):
    def __init__(self, latent_dim: int = 100, ngf=64, img_shape: tuple = (3, 32, 32)):
        super(DCGenerator64, self).__init__()

        # dummy so we can easily get device and type
        self.dummy = nn.Linear(1, 1)

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, img_shape[0], 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.example_input_array = torch.rand(img_shape)

    def forward(self, x):
        return self.model(x.unsqueeze(2).unsqueeze(3))


class DCDiscriminator64(nn.Module):
    def __init__(self, img_shape, dropout=0.2):
        super(DCDiscriminator64, self).__init__()

        nc = img_shape[0]
        ndf = 64
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, 128, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False)),
        )
        self.example_input_array = torch.rand(img_shape)

    def forward(self, img):
        out = self.model(img)  # batch x 1 x 1 x 1
        return out.view(out.shape[:2])  # batch x 1


class InfoDCDiscriminator(nn.Module):
    def __init__(self, img_shape, n_codes=10, dropout=0.2):
        super(InfoDCDiscriminator, self).__init__()

        nc = img_shape[0]
        ndf = 64
        self.trunk = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 1 x 32
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
        )

        self.disc_head = nn.utils.spectral_norm(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

        self.code_head = nn.utils.spectral_norm(
            nn.Conv2d(ndf * 8, n_codes, 4, 1, 0, bias=False)
        )

    def forward(self, img):
        out = self.trunk(img)  # encode
        out = self.disc_head(out)  # predict validity
        return out.view(out.shape[:2])

    def get_features(self, img):
        out = self.trunk(img)  # encode
        shape = out.shape
        return out.view(shape[0], shape[1] * shape[2] * shape[3])

    def get_code_features(self, img, freeze=False):
        out = self.trunk(img)  # encode
        shape = out.shape
        features = out.view(shape[0], shape[1] * shape[2] * shape[3])
        if freeze:
            # only head will be tuned
            out = self.code_head(out.detach())  # predict codes
        else:
            out = self.code_head(out)  # predict codes
        code = out.view(out.shape[:2])
        return code, features

    def predict_code(self, img):
        out = self.trunk(img)  # encode
        out = self.code_head(out)  # predict codes
        return out.view(out.shape[:2])

    def predict_all(self, img):
        out = self.trunk(img)  # encode
        valid = self.disc_head(out)
        codes = self.code_head(out)  # predict codes
        return valid.view(valid.shape[:2]), codes.view(codes.shape[:2])


class InfoDCDiscriminator64(nn.Module):
    def __init__(self, img_shape, n_codes=10, dropout=0.2):
        super(InfoDCDiscriminator64, self).__init__()

        nc = img_shape[0]
        self.trunk = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(nc, 64, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
        )

        self.disc_head = nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False))

        self.code_head = nn.utils.spectral_norm(
            nn.Conv2d(512, n_codes, 4, 1, 0, bias=False)
        )

    def forward(self, img):
        out = self.trunk(img)  # encode
        out = self.disc_head(out)  # predict validity
        return out.view(out.shape[:2])

    def get_features(self, img):
        out = self.trunk(img)  # encode
        shape = out.shape
        return out.view(shape[0], shape[1] * shape[2] * shape[3])

    def get_code_features(self, img, freeze=False):
        out = self.trunk(img)  # encode
        shape = out.shape
        features = out.view(shape[0], shape[1] * shape[2] * shape[3])
        if freeze:
            # only head will be tuned
            out = self.code_head(out.detach())  # predict codes
        else:
            out = self.code_head(out)  # predict codes
        code = out.view(out.shape[:2])
        return code, features

    def predict_code(self, img):
        out = self.trunk(img)  # encode
        out = self.code_head(out)  # predict codes
        return out.view(out.shape[:2])

    def predict_all(self, img):
        out = self.trunk(img)  # encode
        valid = self.disc_head(out)
        codes = self.code_head(out)  # predict codes
        return valid.view(valid.shape[:2]), codes.view(codes.shape[:2])


class LabelModel(nn.Module):
    def __init__(
        self,
        inp_size,
        cardinality,
        class_balance: Optional[List] = None,
        acc_activation: str = "Sigmoid",
    ):
        super(LabelModel, self).__init__()

        self.cardinality = cardinality
        if class_balance is None:
            class_balance = [1 / cardinality for _ in range(cardinality)]

        p = torch.log(torch.tensor(class_balance, requires_grad=False))
        self.register_buffer("p_const", p)
        self.nlf = inp_size

        # initialize LFs equally weighted
        self.acc = nn.Parameter(torch.ones(1, 1, inp_size) * 0.5)
        # nn.Parameter(torch.rand(1, 1, inp_size))

        # let model learn a weight for the class prior.
        self.class_prior_weight = nn.Parameter(torch.rand(1))
        self.class_prior_weight_act = nn.Sigmoid()

        # encode that LFs are believed better than random
        if acc_activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        elif acc_activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif acc_activation.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError("selected accuracy activation not implemented")

    def forward(self, L, get_accuracies=False):
        """
        Parameters
        ----------
        L
            An (n, m, C) tensor with values in {0,1}, i.e. the one-hot encoded LF matrix
        """
        aggregation = torch.matmul(self.activation(self.acc), L).squeeze(1)
        aggregation += (
            self.class_prior_weight_act(self.class_prior_weight) * self.p_const
        )  # add weighted class balance
        Y = F.softmax(aggregation, dim=1)

        if get_accuracies:
            return Y, self.acc.detach().clone().reshape(self.nlf)
        return Y

    def __str__(self):
        return "LabelModel"

class MLP(nn.Module):
    """simple linear pytorch model with sigmoid outputs

    Args:
        input_dim (int): input dimension
        output_dim (int): output dimension
    """

    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.0):
        super(MLP, self).__init__()
        if hidden_dims:
            modules = []
            indim = input_dim
            for out_dim in hidden_dims:
                modules.append(nn.Linear(indim, out_dim))
                modules.append(nn.ReLU())
                if dropout > 0:
                    modules.append(nn.Dropout2d(dropout))
                indim = out_dim
            modules.append(nn.Linear(indim, output_dim))
            self.linear = nn.Sequential(*modules)
        else:
            self.linear = nn.Linear(input_dim, output_dim)
        if output_dim == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

    def predict_proba(self, x):
        outputs = self.linear(x)
        return self.activation(outputs)


class EncoderLabelmodel(nn.Module):
    """simple linear pytorch model with sigmoid outputs

    Args:
        input_dim (int): input dimension
        output_dim (int): output dimension
    """

    def __init__(
        self,
        Num_LFs,
        num_features,
        cardinality,
        hidden_dims,
        class_balance: Optional[List] = None,
        acc_activation: str = "Sigmoid",
        modeltype: str = "vector",
    ):
        super(EncoderLabelmodel, self).__init__()
        self.modeltype = modeltype
        if modeltype == "encoder":
            input_dim = Num_LFs * cardinality + num_features
            output_dim = Num_LFs
            self.mlp = MLP(input_dim, output_dim, hidden_dims)
        elif modeltype == "encoderX":
            input_dim = num_features
            output_dim = Num_LFs
            self.mlp = MLP(input_dim, output_dim, hidden_dims)
        elif modeltype == "encoderL":
            input_dim = Num_LFs * cardinality
            output_dim = Num_LFs
            self.mlp = MLP(input_dim, output_dim, hidden_dims)
        elif modeltype == "vector":
            self.acc = nn.Parameter(torch.ones(1, 1, Num_LFs) * 0.5)
        else:
            raise NotImplementedError("LabelModel type unknown %s" % modeltype)

        # fix class balance but register it
        self.cardinality = cardinality
        if class_balance is None:
            class_balance = [1 / cardinality for _ in range(cardinality)]
        p = torch.tensor(class_balance, requires_grad=False)
        self.register_buffer("p_const", p)
        self.nlf = Num_LFs
        # let model learn a weight for the class prior.
        self.class_prior_weight = nn.Parameter(torch.rand(1))
        self.class_prior_weight_act = nn.Sigmoid()

        # encode that LFs are believed better than random
        if acc_activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        elif acc_activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif acc_activation.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError("selected accuracy activation not implemented")

    def forward(self, X, L, get_accuracies=False):
        """
        Parameters
        ----------
        L
            An (n, m, C) tensor with values in {0,1}, i.e. the one-hot encoded LF matrix
        """
        batchsize = L.shape[0]
        if self.modeltype == "encoder":
            acc = self.mlp(
                torch.cat((X, L.view(batchsize, self.nlf * self.cardinality)), 1)
            )
            aggregation = torch.matmul(
                self.activation(acc.view(batchsize, 1, self.nlf)), L
            ).squeeze(1)
        elif self.modeltype == "encoderX":
            acc = self.mlp(X)
            aggregation = torch.matmul(
                self.activation(acc.view(batchsize, 1, self.nlf)), L
            ).squeeze(1)
        elif self.modeltype == "encoderL":
            acc = self.mlp(L.view(batchsize, self.nlf * self.cardinality))
            aggregation = torch.matmul(
                self.activation(acc.view(batchsize, 1, self.nlf)), L
            ).squeeze(1)
        else:
            aggregation = torch.matmul(self.activation(self.acc), L).squeeze(1)

        aggregation += (
            self.class_prior_weight_act(self.class_prior_weight) * self.p_const
        )  # add weighted class balance

        Y = F.softmax(aggregation, dim=1)

        if get_accuracies:
            if self.modeltype == "vector":
                return Y, self.acc.reshape(self.nlf)
            else:
                return Y, acc
        return Y

    def __str__(self):
        return "ThetaEncoder"
