from argparse import Namespace
from .base_gan_model import BaseGANModel
import torch as t
from .modules.conv1d.conv1d import Conv1DGenerator, Conv1dClassifier, FATConv1dClassifier, Conv1Discriminator
from .modules.lstm import LSTMClassifier
from .modules.transformer_g import Transformer
from .modules.transformer import TransformerDiscriminator

class Model(BaseGANModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.discriminator = Conv1Discriminator(params)
        # self.generator = Transformer(params)
        self.classifier = FATConv1dClassifier()