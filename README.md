# Weakly Supervised GAN (WSGAN)
This repository contains code to train a weakly supervised generative adversarial network (WSGAN) as presented in: 

**Generative Modeling Helps Weak Supervision (and Vice Versa)**<br>
Benedikt Boecking, Nicholas Roberts, Willie Neiswanger, Stefano Ermon, Frederic Sala, Artur Dubrawski<br>
International Conference on Learning Representations (ICLR) (2023)<br>
<a class="" href="https://arxiv.org/abs/2203.12023">[arXiv]</a> <a class="" href="https://openreview.net/forum?id=3OaBBATwsvP">[OpenReview]</a>

The base GAN network used for the WSGAN model in this repository is a simple DCGAN. This allows us to explore the joint training of GANs and label models without requiring costly high-end GPU resources. If you are interested in training a more sophisticated GAN architecture  (e.g. on images of higher resolution), please check out our StyleWSGAN repository with code used in our ablations to train a StyleWSGAN network (based on StyleGAN2-ADA): https://github.com/benbo/stylewsgan 

The style of weak supervision we consider in this repository is that of programmatic weak supervision (data programming), where we learn from multiple sources of imperfect, partial labels. These sources are referred to as labeling functions (LFs). 

# Dependencies
To allow you to run this code easily, we have updated it slightly from the version used in our paper, which relied on old versions of PyTorch, Pytorch Lightning, cuda etc. We recommend that you set up the environment as follows.

You can use the `environment.yml` file (please make sure that it installs pytorch and the cuda toolkit correctly for your system):
```bash
conda env create --file environment.yml --name WSGANpaper
```
Or you can set the environment up manually using the following commands (again, make sure you install pytorch and the cuda toolkit correctly for your system):
```bash
$ conda create -n WSGANpaper python=3.9
$ conda activate WSGANpaper
$ conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
$ conda install scikit-learn -c conda-forge 
$ conda install pytorch-lightning -c conda-forge
$ pip install torch-fidelity
```

The latest version of the code in this repository was tested with:
```bash
python=3.9.16
pytorch=1.13.1
pytorch-cuda=11.7
torchvision=0.14.1
pytorch-lightning=1.9.4
torchmetrics=0.11.3
torch-fidelity=0.3.0
```


### Versions used in the WSGAN paper
Please note that we ran most WSGAN experiments with DCGAN networks with the following versions: 
```bash
python=3.7
pytorch=1.8.0
torchvision=0.9.0
pytorch-lightning==1.3.8
```
We ran some experimemnts at later stages of the project with packages updated to the following versions (and experienced slightly different training dynamics compared to prior runs): 
```bash
pytorch=1.10.1
torchvision=0.11.2
pytorch-lightning=1.5.8
```

# To train a WSGAN on a weakly supervised dataset 
Export the `WSGAN-paper` repository path:
```bash
$ export PYTHONPATH="${PYTHONPATH}:/pathtodirectory/WSGAN-paper"
```

Then use `main_labelmodel.py` in the lightning folder, e.g.

```bash
$ conda activate WSGAN
$ python lightning/main_labelmodel.py --dataset CIFAR10 --gpus 3 --batch_size 16 --lffname /pathtodirectory/WSGAN-paper/data/CIFAR10/fixed_LFs.pth --max_epochs 150 --whichmodule GANLabelModel --ganenctype encoderX --storedir /outputs/ --data_path /pathtodownloaddirectory/
###
# !!before running the command above, please adapt it to the correct directories on your system!!
###
# --dataset # specifies dataset to use
# --gpus # specifies gpu IDs to use
# --batch_size # specifies batch size
# --lffname # specifies path to file with weak labels
# --max_epochs # specifies number of epochs to train for
# --whichmodule # specifies which module to train (GANLabelModel is a WSGAN, but you can also train an InfoGAN)
# --ganenctype # specifies which encoder to use for the label model (encoderX is recommended)
# --storedir # specifies where to store the model checkpoints and logs
# --data_path # specifies where to find the dataset or where to download it to
```

NOTE: the code will seeminlgy hang for a while after the first epoch (epoch 0) . This is because FID is computed for the first time, which takes a while as the dataset and fake images are passed through an inception network. By default, FID will be computed every 10 epochs. 

Use tensorboard to view the training progress:
```bash
$ conda activate WSGAN
$ tensorboard --logdir /pathtooutputdirectory/wsganlogs/
```


# Datasets used in the WSGAN paper
NOTE: when using these sets of labeling functions (LFs), please double check the accuracy and coverage of each LF (i.e. compare their votes to the ground truth), to ensure that you spot any errors stemming from the LF votes being associated with the wrong samples. Because we are using publicly available datasets with our own LFs, we cannot guarantee that the datasets will remain unchanged or that images will be loaded the same way on all systems.

We are able to release some of the LF sets used in the WSGAN paper in a way that they can be associated with publicly available datasets, without having to rehost the original image datasets. You can use these LF sets to train a WSGAN by loading the following files that contain the weak labels and sample indices:
- **CIFAR10-B**: use ``data/CIFAR10/ssl_lfs.pth`` for weak labels (``--lffname``) and set ``--numlfs 20``
- **MNIST**: use ``data/MNIST/ssl_lfs.pth`` for weak labels (``--lffname``) and set ``--numlfs 29``
- **FashionMNIST**: use ``data/FashionMNIST/ssl_lfs.pth`` for weak labels (``--lffname``) and set ``--numlfs 23``
- **GTSRB**: use ``data/GTSRB/ssl_lfs.pth`` for weak labels (``--lffname``) and set ``--numlfs 100``
- **CIFAR10-A**: use ``data/CIFAR10/synthetic_lfs.pth`` for weak labels (``--lffname``) and set ``--numlfs 20``

