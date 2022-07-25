from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace

import matplotlib.pyplot as plt
import ipdb
import os
from torchmetrics import Accuracy

import mate
import numpy as np
import monai


class BaseClassificationModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.classifier: t.nn.Module
        self.critarion = t.nn.BCEWithLogitsLoss(pos_weight=t.tensor([3]))
        self.loss = lambda x, y: self.critarion(x, y.unsqueeze(1))

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, labels = batch
        y_pred = self.classifier(x)
        loss = self.loss(y_pred, labels)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, prog_bar=True)

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, labels = batch
        y_pred = self.classifier(x)
        loss = self.loss(y_pred, labels)

        amex = self.amex_metric_pytorch(labels, y_pred)
        # amex = t.tensor(amex)

        return {"val_loss": loss, "amex": amex}

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        amex = t.stack([x["amex"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("amex", amex, prog_bar=True)
        return {"val_loss": avg_loss, "amex": amex}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):

        x, labels = batch
        y_pred = self.classifier(x)
        loss = self.loss(y_pred, labels)

        # amex = self.amex_metric(labels.cpu().numpy(), y_pred.cpu().numpy())
        amex = self.amex_metric_pytorch(labels, y_pred)

        return {"test_loss": loss, "amex": amex}

    def test_epoch_end(self, outputs):
        avg_loss = t.stack([x["test_loss"] for x in outputs]).mean()
        amex = t.stack([x["amex"] for x in outputs]).mean()

        return {"test_loss": avg_loss, "amex": amex}

    def configure_optimizers(self):
        # import torch_optimizer as optim
        # optimizer = optim.lamb.Lamb(self.parameters(), lr=0.03)

        # return optimizer

        return mate.Optimizer(self.params.optimizer, self.classifier).get_optimizer()

    def amex_metric_pytorch(self, y_true: t.Tensor, y_pred: t.Tensor):

        # convert dtypes to float64
        y_true = y_true.flatten().double()
        y_pred = y_pred.flatten().double()

        # count of positives and negatives
        n_pos = y_true.sum()
        n_neg = y_pred.shape[0] - n_pos

        # sorting by descring prediction values
        indices = t.argsort(y_pred, dim=0, descending=True)
        preds, target = y_pred[indices], y_true[indices]

        # filter the top 4% by cumulative row weights
        weight = 20.0 - target * 19.0
        cum_norm_weight = (weight / weight.sum()).cumsum(dim=0)
        four_pct_filter = cum_norm_weight <= 0.04

        # default rate captured at 4%
        d = target[four_pct_filter].sum() / n_pos

        # weighted gini coefficient
        lorentz = (target / n_pos).cumsum(dim=0)
        gini = ((lorentz - cum_norm_weight) * weight).sum()

        # max weighted gini coefficient
        gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

        # normalized weighted gini coefficient
        g = gini / gini_max

        return 0.5 * (g + d)

    def amex_metric(self, y_true, y_pred):

        y_true = y_pred.flatten()
        y_pred = y_pred.flatten()

        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, 1].argsort()[::-1]]
        weights = np.where(labels[:, 0] == 0, 20, 1)
        cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
        top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])

        gini = [0, 0]
        for i in [1, 0]:
            labels = np.transpose(np.array([y_true, y_pred]))
            labels = labels[labels[:, i].argsort()[::-1]]
            weight = np.where(labels[:, 0] == 0, 20, 1)
            weight_random = np.cumsum(weight / np.sum(weight))
            total_pos = np.sum(labels[:, 0] * weight)
            cum_pos_found = np.cumsum(labels[:, 0] * weight)
            lorentz = cum_pos_found / total_pos
            gini[i] = np.sum((lorentz - weight_random) * weight)
        print(
            "G: {:.6f}, D: {:.6f}, ALL: {:6f}".format(
                gini[1] / gini[0], top_four, 0.5 * (gini[1] / gini[0] + top_four)
            )
        )
        return 0.5 * (gini[1] / gini[0] + top_four)
