# Unsupervised Domain Adaptation for Image Classification

## Installation

It’s suggested to use **pytorch==1.7.1** and torchvision==0.8.2 in order to reproduce the benchmark results.

Example scripts support all models in [PyTorch-Image-Models](https://github.com/rwightman/pytorch-image-models). You
also need to install timm to use PyTorch-Image-Models.

```
pip install timm
```

## Dataset

Following datasets can be downloaded automatically:

- [Office31](https://www.cc.gatech.edu/~judy/domainadapt/)
- [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
- [VisDA2017](http://ai.bu.edu/visda-2017/)
- [DomainNet](http://ai.bu.edu/M3SDA/)


## Usage

The shell files give the script to reproduce the benchmark with specified hyper-parameters. For example, if you want to
train cdan_v1.py on OfficeHome, use the following script

```shell script
CUDA_VISIBLE_DEVICES=0 python cdan_v1.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Ar2Cl
```

Note that ``-s`` specifies the source domain, ``-t`` specifies the target domain, and ``--log`` specifies where to store
results.

Directory that stores datasets will be named as
``examples/domain_adaptation/image_classification/data/<dataset name>``.

You can also watch these results in the log file ``logs/dann/Office31_A2W/log.txt``.

After training, you can test your algorithm's performance by passing in ``--phase test``.

```
CUDA_VISIBLE_DEVICES=0 python cdan_v1.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 --seed 0 --log logs/cdan/OfficeHome_Ar2Cl --phase test
```
