from argparse import Namespace
from .base_classification_model import BaseClassificationModel
import torch as t
from .modules.conv1d.conv1d import FATConv1dClassifier
from .modules.lstm import LSTMClassifier
from .modules.transformer import Transformer
from .modules.conv2d.resnet import ResNetClassifier
from .modules.conv2d.resnet_official import Bottleneck, ResNet
class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.classifier = ResNetClassifier(
            params,
            [4, 4, 4, 4],
            [32, 64, 128, 256],
            in_channel=13,
            num_classes=1
        )
        # self.classifier = ResNet(
        #     Bottleneck, layers = [4,4,4,4], num_classes=1
        # )