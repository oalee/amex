import ipdb
import torch
import torch as t
import torch.nn.functional as F
from torch import nn
from ..conv1d.conv1d import Conv1DLayers, GaussianNoise
from ..transformer import Transformer


class ResNetBlock(nn.Module):
    def __init__(
        self, c_in, act, subsample=False, c_out=-1, droupout=0.1, double_dropout=False
    ):

        super().__init__()

        # the first layer has c_in input channels and c_out output channels
        # other layers have c_out input channels and c_out output channels
        if not subsample:
            c_out = c_in

        self.double_dropout = double_dropout

        self.do = nn.Dropout(droupout)

        self.net = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=3,
                padding=1,
                stride=1 if not subsample else 2,
                bias=False,
            ),  # No bias needed as the Batch Norm handles it
            nn.BatchNorm2d(c_out),
            act,
            self.do,
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = (
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        )
        self.act = act

    def forward(self, x):

        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act(out)
        if self.double_dropout:
            out = self.do(out)
        return out


class ResNetClassifier(nn.Module):
    def __init__(
        self,
        params,
        num_blocks=[4, 4, 4, 4],
        c_hidden=[32, 64, 128, 256],
        in_channel=1,
        num_classes=1,
    ):
        super().__init__()
        self.params = params
        self.c_hidden = c_hidden
        self.num_blocks = num_blocks
        self.in_channel = in_channel
        self.num_classes = num_classes

        # creates the layers
        self.create_network()

        # init the weights for better initial guess
        self.init_params()

        self.act = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def create_network(self):

        # array of hidden channels
        c_hidden = self.c_hidden

        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
            nn.Conv2d(
                self.in_channel,
                c_hidden[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(c_hidden[0]),
            nn.ReLU(),
        )

        # Creating the ResNet blocks
        blocks = []
        # ipdb.set_trace()
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                subsample = (
                    bc == 0 and block_idx > 0
                )  # Subsample the blocks of each group, except the very first one.
                blocks.append(
                    # first block is subsampled, others have input and output channels equal
                    ResNetBlock(
                        c_hidden[block_idx if not subsample else (block_idx - 1)],
                        nn.ReLU(),
                        subsample,
                        c_hidden[block_idx],
                    )
                )

        self.blocks = nn.Sequential(*blocks)

        # Mapping to classification output
        # Average pooling to get the final feature vector and reduce dimensionality
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # nn.Linear(c_hidden[-1], self.num_classes),
        )

        self.out_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_hidden[-1] * 3 + 188, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.conv1d = Conv1DLayers(6, 13, 256, 0.2)
        self.conv1dt = Conv1DLayers(6, 188, 256, 0.2)

        self.transformer = Transformer(self.params)
        self.noise = GaussianNoise(0.002)

    def init_params(self):
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # input size is B, 13, 188
        B, T, D = x.shape
        pad_dim = 256 - 188
        noise = t.randn(x.shape[0], T, pad_dim, device=x.device)
        conv1 = self.conv1d(self.noise(x))
        x_t = x.permute(0, 2, 1)
        conv1t = self.conv1dt(self.noise(x_t))

        transformer_h = self.transformer.hid(self.noise(x))

        conv1 = t.max(conv1, dim=2)[0]
        conv1t = t.max(conv1t, dim=2)[0]

        x = t.cat([x, noise], dim=2)

        x = x.reshape(B, T, 16, 16)

        # changes the channels of the input image
        x = self.input_net(self.noise(x))
        # ResNet blocks
        x = self.blocks(x)
        # Classification output
        # ipdb.set_trace()
        x = self.output_net(x)
        # ipdb.set_trace()

        x = t.cat([x, conv1, conv1t, transformer_h], dim=1)
        x = self.out_net(x)
        x = self.act(x)

        return x.squeeze(1)
