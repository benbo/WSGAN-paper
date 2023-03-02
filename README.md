# WSGAN
This repository contains code to train a weakly supervised generative adversarial network (WSGAN) as presented in: 
**Generative Modeling Helps Weak Supervision (and Vice Versa)**<br>
B Boecking, W Neiswanger, N Roberts, S Ermon, F Sala, A Dubrawski<br>
International Conference on Learning Representations (ICLR) (2023)<br>
<a class="" href="https://arxiv.org/abs/2203.12023">[arXiv]</a> <a class="" href="https://openreview.net/forum?id=3OaBBATwsvP">[OpenReview]</a>

The base GAN network for the WSGAN model in this repository is a simple DCGAN. This allows us to explore the joint training of GANs and label models without requiring costly high-end GPU resources. If you are interested in training a more sophisticated GAN architecture  (e.g. on images of higher resolution), please check out our other repository with code used in our ablations to train a StyleWSGAN network (based on StyleGAN2-ADA): https://github.com/benbo/stylewsgan 


# Dependencies
```bash
conda install pip
# check correct installation for your system: https://pytorch.org
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install pytorch-lightning
pip install lightning-bolts  
```

Please note that we ran most experiments with the following versions: 
```bash
pytorch 1.8.0
torchvision  0.9.0
pytorch-lightning 1.3.8
```

After uprading to the versions below, using the same settings as previously, we experienced slightly different training dynamics: 
```bash
pytorch 1.10.1
torchvision  0.11.2
pytorch-lightning 1.5.8
```
# To train a WSGAN on a weakly supervised dataset 
Export path
```bash
$ export PYTHONPATH="${PYTHONPATH}:/pathtodirectory/WSGAN"
```

Use `main_labelmodel.py` in the lightning folder, e.g.
```bash
python lightning/main_labelmodel.py --dataset CIFAR10 --gpus 3 --batch_size 16 --lffname ./data/CIFAR10/fixed_LFs.pth --max_epochs 150 --whichmodule GANLabelModel --ganenctype encoderX --storedir /outputs/
```
