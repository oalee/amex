from turtle import forward
import torch as t
from torch import nn
import ipdb
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        drouput=0.1,
        residual=False,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(drouput)
        self.init_weights()

    def forward(self, x):

        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.do(x)
        return x

    def init_weights(self):

        # init weights of conv1d

        nn.init.kaiming_normal_(self.conv1d.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv1d.bias, 0)

        # init weights of bn

        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


class Conv1DLayers(nn.Module):
    def __init__(self, layers, in_channels, out_channels, dropout=0.2) -> None:
        super().__init__()

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.append(
                Conv1DBlock(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    groups=1,
                    bias=True,
                    drouput=dropout,
                    residual=False,
                )
            )
            in_channels = out_channels

    def forward(self, x):
        x = self.layers(x)
        return x


class Conv1Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.embedding = nn.Embedding(2, self.params.hparams.in_features)
        self.layers = Conv1DLayers(5, 13, 512, dropout=0.2)
        # self.t_layers = Conv1DLayers(5, 376, 512, dropout=0.5)
        self.fc = nn.Linear(512, 1)
        self.act = nn.Sigmoid()
        self.noise = GaussianNoise(0)

    def forward(self, x, y):

        # y = y.unsqueeze(1)
        cond = self.embedding(y.int())
        cond = cond.repeat(1, x.shape[1], 1)
        y = y.repeat(1, x.shape[1]).unsqueeze(-1)
        # x_t = self.noise(x)
        x = self.noise(x)
        # ipdb.set_trace()
        x = t.cat([x, cond], dim=2)
        # x_t = t.cat([x_t, cond], dim=2)
        # ipdb.set_trace()
        # x_t = x_t.permute(0, 2, 1)
        x = self.layers(x)
        # x_t = self.t_layers(x_t)

        # ipdb.set_trace()
        x = t.max(x, dim=2)[0]
        # x_t = t.max(x_t, dim=2)[0]
        x = x  
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.act(x)
        return x


class Conv1DGenerator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.z_dim = 64
        self.embedding = nn.Embedding(2, 47)
        self.noise_layer = nn.Sequential(nn.Linear(self.z_dim, 3 * 47), nn.LeakyReLU())

        self.conv1 = nn.Sequential(
            Conv1DBlock(4, 32, 3, 1, 1),
            Conv1DBlock(32, 64, 3, 1, 1),
            Conv1DBlock(64, 128, 3, 1, 1),
        )
        # 128 x 47
        self.conv2 = nn.ConvTranspose1d(128, 256, kernel_size=2, stride=2, padding=0)
        # 256 x 94
        self.conv3 = Conv1DBlock(256, 512, 3, 1, 1)
        self.conv4 = nn.ConvTranspose1d(512, 512, kernel_size=2, stride=2, padding=0)
        # 512 x 188
        self.conv5 = nn.Sequential(
            Conv1DBlock(512, 256, 3, 1, 1),
            Conv1DBlock(256, 128, 3, 1, 1),
            Conv1DBlock(128, 64, 3, 1, 1),
            Conv1DBlock(64, 13, 3, 1, 1),
        )

        # 4 x 47
        #

    def forward(self, y):
        z = t.randn(y.shape[0], self.z_dim, device=y.device)
        y = y.unsqueeze(1)
        y = self.embedding(y.int())
        z = self.noise_layer(z)
        z = z.view(z.shape[0], 3, 47)

        z = t.cat([y, z], dim=1)
        # ipdb.set_trace()

        z = self.conv1(z)
        x = self.conv2(z)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # ipdb.set_trace()

        return x


# Conv1d Classifier
class Conv1dClassifier(nn.Module):
    def __init__(self, in_channels=248, num_class=4):
        super(Conv1dClassifier, self).__init__()

        self.conv1 = Conv1DBlock(
            in_channels, 256, kernel_size=self.kernel_size, stride=1, padding=1
        )
        self.conv2 = Conv1DBlock(
            256, 256, kernel_size=self.kernel_size, stride=1, padding=1
        )
        self.conv3 = Conv1DBlock(
            256, 256, kernel_size=self.kernel_size, stride=1, padding=1
        )
        self.conv4 = Conv1DBlock(
            256, 256, kernel_size=self.kernel_size, stride=1, padding=1, drouput=0
        )
        # spacial attention
        self.classifier = nn.Linear(256, num_class)
        self.noise_layer = GaussianNoise(0.01)

    def forward(self, x):
        # ipdb.set_trace()
        x = x.squeeze(1)
        x = self.noise_layer(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # ipdb.set_trace()
        x = x.max(dim=2)[0]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.softmax(x, dim=1)


# Conv1d Classifier
class FitConv1dClassifier(nn.Module):
    def __init__(self, in_channels=248, num_class=4):
        super(FitConv1dClassifier, self).__init__()

        self.dim = 64
        self.kernel_size = 4
        self.droupout = 0.01
        # 248x195
        self.conv1 = Conv1DBlock(
            in_channels,
            self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            drouput=0,
        )
        # 256x195
        self.conv2 = Conv1DBlock(
            self.dim,
            self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            drouput=0.05,
        )
        # 256x98
        self.conv3 = Conv1DBlock(
            self.dim,
            self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            drouput=self.droupout,
        )
        # 256x49
        # self.conv4 = Conv1DBlock(self.dim, self.dim, kernel_size=self.kernel_size, stride=1, padding=1, drouput=0.1)
        # spacial attention
        self.classifier = nn.Linear(self.dim, num_class)
        # self.bclassifier = nn.Sequential(
        #     nn.Linear(34368, 2000), nn.ReLU(), nn.Linear(2000, num_class)
        # )
        self.noise_layer = GaussianNoise(0.01, False)

    def forward(self, x):
        # ipdb.set_trace()
        x = x.squeeze(1)
        x = self.noise_layer(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # ipdb.set_trace()
        x = x.max(dim=2)[0]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.softmax(x, dim=1)


class GaussianNoise(nn.Module):
    def __init__(self, stddev, minmax=True):
        super().__init__()
        self.stddev = stddev
        self.minmax = minmax

    def forward(self, x):
        if self.training:
            if self.minmax:
                range = x.max() - x.min()
                noise = t.randn(x.shape).to(x.device) * range * self.stddev
            else:
                sign = t.randn(x.shape).to(x.device)
                sign[sign > 0.5] = 1
                sign[sign < 0.5] = -1
                noise = t.randn(x.shape).to(x.device) * self.stddev * sign
            return x + noise
        else:
            return x


# FAT Conv1d Classifier
class FATConv1dClassifier(nn.Module):
    def __init__(
        self, in_channels=13, hidden_channel=512, num_layers=6, dropout=0.3, num_class=1
    ):
        super(FATConv1dClassifier, self).__init__()

        for i in range(num_layers):

            # # droput for the last layer is 0.05, first one 0 and the the rest 0.15
            # if i == 0:
            #     drouput = 0.05
            # elif i == num_layers - 1:
            #     drouput = 0.05
            # else:
            #     drouput = 0.50

            setattr(
                self,
                f"conv{i}",
                Conv1DBlock(
                    in_channels,
                    hidden_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    drouput=dropout,
                ),
            )
            in_channels = hidden_channel

        self.conditional_embedding = nn.Embedding(2, hidden_channel)

        self.classifier = nn.Linear(hidden_channel, num_class)
        self.num_layers = num_layers
        self.noise_layer = GaussianNoise(0, False)

    def forward(self, x):
        # ipdb.set_trace()
        x = self.noise_layer(x)
        x = x.squeeze(1)
        rand_noise = t.randn(*x.shape[:2], 64).to(x.device)
        # ipdb.set_trace()
        x = t.cat([x, rand_noise], dim=2)
        for i in range(self.num_layers):
            x = getattr(self, f"conv{i}")(x)

        x = x.max(dim=2)[0]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.sigmoid(x)


# Conv1d Classifier
class Res1D(nn.Module):
    def __init__(self, in_channels=248, num_class=4):
        super(Res1D, self).__init__()

        self.dim = 256
        self.kernel_size = 1
        self.droupout = 0.01
        # 248x195
        self.conv1 = Conv1DResidual(
            in_channels,
            self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            drouput=self.droupout,
        )
        # 256x195
        self.conv2 = Conv1DResidual(
            self.dim,
            self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            drouput=self.droupout,
        )
        # 256x98
        self.conv3 = Conv1DResidual(
            self.dim,
            self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            drouput=self.droupout,
        )
        # 256x49
        # self.conv4 = Conv1DBlock(self.dim, self.dim, kernel_size=self.kernel_size, stride=1, padding=1, drouput=0.1)
        # spacial attention
        self.classifier = nn.Linear(self.dim, num_class)
        # self.bclassifier = nn.Sequential(
        #     nn.Linear(34368, 2000), nn.ReLU(), nn.Linear(2000, num_class)
        # )
        self.noise_layer = GaussianNoise(2, False)

    def forward(self, x):
        # ipdb.set_trace()
        x = x.squeeze(1)
        # ipdb.set_trace()
        x = self.noise_layer(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # ipdb.set_trace()
        x = x.max(dim=2)[0]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.softmax(x, dim=1)


class Conv1DResidual(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=16, stride=1, padding=1, drouput=0
    ):
        super(Conv1DResidual, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, 16, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drouput = drouput
        self.dropout = nn.Dropout(self.drouput)
        self.do = nn.Dropout(drouput)

    def forward(self, x):
        residual = x
        re_res = self.do(x)
        x = self.conv1(x)
        # residual = self.do(x)
        x = self.conv2(x)
        # ipdb.set_trace()
        if residual.shape == x.shape:
            x = x + residual
        # if re_res.shape == x.shape:
        #     x = x + re_res

        x = self.bn1(x)
        x = self.relu(x)
        x = self.do(x)
        return x
