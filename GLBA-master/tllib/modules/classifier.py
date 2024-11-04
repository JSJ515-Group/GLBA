"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch

import torch.nn.functional as F
from tllib.modules.grl import WarmStartGradientReverseLayer
from timm.models.layers import trunc_normal_

__all__ = ['Classifier', 'Classifier_v1']


class Classifier(nn.Module):
    """A generic Classifier class for domain adaptation.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True

    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)

    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        predictions = self.head(f)
        if self.training:
            return predictions, f
        else:
            return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params


class Classifier_v1(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, embed_dim: Optional[int] = -1, channel_ratio=4):
        super(Classifier_v1, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        # Classifier head
        self.bottleneck_t = nn.Linear(embed_dim, 256)
        # self.bottleneck_t = nn.Sequential(
        #     nn.Linear(embed_dim, 384),
        #     nn.BatchNorm1d(384),
        #     nn.ReLU()
        # )
        # self.bottleneck_t = nn.Linear(embed_dim, 1024)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.Linear(int(256 * channel_ratio), 256)
        # self.bottleneck = nn.Sequential(
        #     nn.Linear(int(256 * channel_ratio), 384),
        #     nn.BatchNorm1d(384),
        #     nn.ReLU()
        # )
        # self.bottleneck = nn.Linear(int(256 * channel_ratio), 1024)

        self.cls_head = nn.Linear(256, num_classes)
        # self.cls_head = nn.Linear(1024, num_classes)

        # self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # self.pooling = nn.AdaptiveAvgPool2d(1)
        # self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        # f = self.pool_layer(self.backbone(x))
        # f = self.bottleneck(f)
        # predictions = self.head(f)
        # if self.training:
        #     return predictions, f
        # else:
        #     return predictions

        x, x_t = self.backbone(x)
        x_p = self.pooling(x).flatten(1)

        x_p = self.bottleneck(x_p)
        conv_cls = self.cls_head(x_p)
        # conv classification
        # conv_cls = self.conv_cls_head(x_p)

        x_t = self.bottleneck_t(x_t)
        tran_cls = self.cls_head(x_t)
        # trans classification
        # tran_cls = self.trans_cls_head(x_t)

        if self.training:
            return conv_cls, x_p, tran_cls, x_t
        else:
            return conv_cls, tran_cls

    def get_parameters(self, base_lr=0.2) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.bottleneck_t.parameters(), "lr": 1.0 * base_lr},
            {"params": self.cls_head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params


class ImageClassifier(Classifier):
    pass

