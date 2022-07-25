from argparse import Namespace
from .base_classification_model import BaseClassificationModel
import torch as t
from .modules.conv1d.conv1d import FATConv1dClassifier
from .modules.lstm import LSTMClassifier
from .modules.transformer import Transformer
from .modules.conv2d.resnet import ResNetClassifier
from .modules.conv2d.resnet_official import wide_resnet101_2, ResNet, Bottleneck
# from .modules.resnet.resnet import ResNet
import ipdb

class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        ipdb.set_trace()
        # self.resnet = ResNet(*params.resnet)
        self.classifier = ResNetClassifier(
            params,
            num_blocks=[4, 6, 4, 6],
            c_hidden=[64, 128, 256, 512],
        )

        # ResNet(
        #     params,
        #     Bottleneck,
        #     [3, 4, 6, 3],
        #     width_per_group=64,
        #     replace_stride_with_dilation=[True, True, True],
        # )
