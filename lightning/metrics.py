# Code below adapted from metrics code released by The PyTorch Lightning team
# https://github.com/PyTorchLightning/metrics
# originally released under Apache License, Version 2.0 (the "License") http://www.apache.org/licenses/LICENSE-2.0

from torchmetrics.metric import Metric
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.image.fid import _compute_fid
from torchmetrics.image.kid import poly_mmd
from sklearn.metrics import adjusted_rand_score

from torch.nn import Module

from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE


def _cl_update(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    """Check if the 2 input tenseor have
    the same number of elements and if they are 1d.
    Args:
        x: x-coordinates
        y: y-coordinates
    """

    if x.ndim > 1:
        x = x.squeeze()

    if y.ndim > 1:
        y = y.squeeze()

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError(
            f"Expected both `x` and `y` tensor to be 1d, but got tensors with dimension {x.ndim} and {y.ndim}"
        )
    if x.numel() != y.numel():
        raise ValueError(
            f"Expected the same number of elements in `x` and `y` tensor but received {x.numel()} and {y.numel()}"
        )
    return x, y


class ARI(Metric):
    r"""
    Computes adjusted rand index
    Args:
        reorder: AUC expects its first input to be sorted. If this is not the case,
            setting this argument to ``True`` will use a stable sorting algorithm to
            sort the input in descending order
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the ``allgather`` operation on the metric state. When ``None``, DDP
            will be used to perform the ``allgather``.
    """
    is_differentiable = False
    x: List[Tensor]
    y: List[Tensor]

    def __init__(
        self,
        reorder: bool = False,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.reorder = reorder

        self.add_state("x", default=[], dist_reduce_fx="cat")
        self.add_state("y", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
        """
        x, y = _cl_update(preds, target)

        self.x.append(x)
        self.y.append(y)

    def compute(self) -> Tensor:
        """Computes AUC based on inputs passed in to ``update`` previously."""
        x = dim_zero_cat(self.x).cpu().numpy().astype(int)
        y = dim_zero_cat(self.y).cpu().numpy().astype(int)
        # TODO could compute multiple metrics and call in on epoch end to
        # then log them separately
        return torch.tensor(adjusted_rand_score(x, y))


def inception_transforms(imgs):
    # transform image so that incpetion network accepts it
    if imgs.shape[1] == 1:
        # deal with grayscale
        imgs = imgs.repeat(1, 3, 1, 1)
    # our images are normalized to [-1,1]
    # transform to [0, 255] with torch.uint8 as dtype
    imgs = ((imgs + 1.0) * 255.0 / 2.0).type(torch.uint8)
    return imgs


class InceptionScores(Metric):
    r"""
    Calculates multiple different inception scores so that we don't compute features for each score separately
    Args:
        feature:
            Either an str, integer or ``nn.Module``:
            - an str or integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              'logits_unbiased', 64, 192, 768, 2048
            - an ``nn.Module`` for using a custom feature extractor. Expects that its forward method returns
              an ``[N,d]`` matrix where ``N`` is the batch size and ``d`` is the feature size.
        splits:
            Integer determining how many splits the inception score calculation should be split among
        subsets:
            Number of subsets to calculate the mean and standard deviation scores over
        subset_size:
            Number of randomly picked samples in each subset
        degree:
            Degree of the polynomial kernel function
        gamma:
            Scale-length of polynomial kernel. If set to ``None`` will be automatically set to the feature size
        coef:
            Bias term in the polynomial kernel.
        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather
    Raises:
        ValueError:
            If ``feature`` is set to an ``int`` (default settings) and ``torch-fidelity`` is not installed
        ValueError:
            If ``feature`` is set to an ``int`` not in [64, 192, 768, 2048]
        ValueError:
            If ``subsets`` is not an integer larger than 0
        ValueError:
            If ``subset_size`` is not an integer larger than 0
        ValueError:
            If ``degree`` is not an integer larger than 0
        ValueError:
            If ``gamma`` is niether ``None`` or a float larger than 0
        ValueError:
            If ``coef`` is not an float larger than 0
    """
    real_features: List[Tensor]
    fake_features: List[Tensor]

    def __init__(
        self,
        feature: Union[str, int, torch.nn.Module] = 2048,
        splits: int = 10,
        subsets: int = 100,
        subset_size: int = 1000,
        degree: int = 3,
        gamma: Optional[float] = None,  # type: ignore
        coef: float = 1.0,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        rank_zero_warn(
            "Metric `KID` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        if isinstance(feature, (str, int)):
            if not _TORCH_FIDELITY_AVAILABLE:
                raise RuntimeError(
                    "KID metric requires that Torch-fidelity is installed."
                    " Either install as `pip install torchmetrics[image]`"
                    " or `pip install torch-fidelity`"
                )
            valid_int_input = ("logits_unbiased", 64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input},"
                    f" but got {feature}."
                )

            self.inception: Module = NoTrainInceptionV3(
                name="inception-v3-compat", features_list=[str(feature)]
            )
        elif isinstance(feature, Module):
            self.inception = feature
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not (isinstance(subsets, int) and subsets > 0):
            raise ValueError("Argument `subsets` expected to be integer larger than 0")
        self.subsets = subsets

        if not (isinstance(subset_size, int) and subset_size > 0):
            raise ValueError(
                "Argument `subset_size` expected to be integer larger than 0"
            )
        self.subset_size = subset_size

        if not (isinstance(degree, int) and degree > 0):
            raise ValueError("Argument `degree` expected to be integer larger than 0")
        self.degree = degree

        if gamma is not None and not (isinstance(gamma, float) and gamma > 0):
            raise ValueError(
                "Argument `gamma` expected to be `None` or float larger than 0"
            )
        self.gamma = gamma

        if not (isinstance(coef, float) and coef > 0):
            raise ValueError("Argument `coef` expected to be float larger than 0")
        self.coef = coef

        self.splits = splits

        self.real_features = []
        self.fake_features = []

        # states for extracted features
        # self.add_state("real_features", [], dist_reduce_fx=None)
        # self.add_state("fake_features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor, real: bool) -> None:  # type: ignore
        """Update the state with extracted features.
        Args:
            imgs: tensor with images feed to the feature extractor
            real: bool indicating if imgs belong to the real or the fake distribution
        """
        features = self.inception(imgs).cpu()
        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self) -> Tuple[Tensor, Tensor]:
        #####
        # KID
        #####
        # First, compute  KID score based on accumulated extracted features from the two distributions.
        # Mean and standard deviation of KID scores are calculated on subsets of extracted features.

        real_features = dim_zero_cat(self.real_features)
        fake_features = dim_zero_cat(self.fake_features)

        n_samples_real = real_features.shape[0]
        if n_samples_real < self.subset_size:
            raise ValueError(
                "Argument `subset_size` should be smaller than the number of samples"
            )
        n_samples_fake = fake_features.shape[0]
        if n_samples_fake < self.subset_size:
            raise ValueError(
                "Argument `subset_size` should be smaller than the number of samples"
            )

        #####
        # FID
        #####

        # computation is extremely sensitive so we use double precision
        orig_dtype = real_features.dtype
        real_features = real_features.double()
        fake_features = fake_features.double()

        # calculate mean and covariance
        n = real_features.shape[0]
        mean1 = real_features.mean(dim=0)
        mean2 = fake_features.mean(dim=0)
        diff1 = real_features - mean1
        diff2 = fake_features - mean2
        cov1 = 1.0 / (n - 1) * diff1.t().mm(diff1)
        cov2 = 1.0 / (n - 1) * diff2.t().mm(diff2)
        fid = _compute_fid(mean1, cov1, mean2, cov2).to(orig_dtype)


        scores = {
            "FID": fid
            }

        return scores
