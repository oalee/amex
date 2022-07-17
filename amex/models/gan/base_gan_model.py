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


class BaseGANModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        # self.generator: t.nn.Module
        self.discriminator: t.nn.Module
        self.classifier = t.nn.Module

        self.criterion = t.nn.BCELoss()
        self.loss = lambda x, y: self.criterion(x.flatten(), y.flatten())

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        return out

    def __adversarial_loss(self, y_pred: t.Tensor, y_true: t.Tensor):
        return F.binary_cross_entropy(y_pred.flatten(), y_true.flatten()).mean()

    def __discriminator_loss(self, x, y):

        pred_label = self.classifier(x)

        pred_label_idx = (
            (pred_label.round().int() != y.squeeze().int()).nonzero().squeeze()
        )

        loss = t.tensor(0, device = x.device)
        if pred_label_idx.numel() > 0:
            # ipdb.set_trace()

            pred_fake_x = self.discriminator(
                x[pred_label_idx, ...], pred_label[pred_label_idx, ...]
            )
            try:
                b_size = pred_fake_x.shape[0]
            except:
                b_size = 1
            fake_t_label = t.zeros(b_size).to(x.device)
            loss = self.__adversarial_loss(pred_fake_x, fake_t_label)

        pred_real_x = self.discriminator(x, y)

        # real invert 0 and 1
        y_invert = 1 - y
        pred_y_invernt = self.discriminator(x, y_invert)

        real_label = t.ones(x.shape[0]).to(x.device)
        fake_label = t.zeros(x.shape[0]).to(x.device)

        loss = [
            self.__adversarial_loss(pred_real_x, real_label),
            loss,
            self.__adversarial_loss(pred_y_invernt, fake_label),
        ]

        t_loss = t.stack(loss).sum() / 3
        self.log("D_loss", t_loss, prog_bar=True)
        self.log("D_Real_loss", loss[0], prog_bar=True)
        self.log("D_Fake_loss", loss[1], prog_bar=True)
        # self.log("D_Invert_loss", loss[2], prog_bar=True)
        return t_loss

    def __classifier_loss(self, x: t.Tensor, y):

        pred_label = self.classifier(x)

        true_label = t.ones(x.shape[0]).to(x.device)

        pred_label_idx = (
            (pred_label.round().int() != y.squeeze().int()).nonzero().squeeze()
        )

        loss = t.tensor(0, device = x.device)
        if pred_label_idx.numel() > 0:
            # ipdb.set_trace()
            pred_fake_x = self.discriminator(
                x[pred_label_idx, ...], pred_label[pred_label_idx, ...]
            )
            try:
                b_size = pred_fake_x.shape[0]
            except:
                b_size = 1
            t_label = t.ones(b_size).to(x.device)
            loss = self.__adversarial_loss(pred_fake_x, t_label)

        loss = (
            self.params.hparams.adversarial_loss_weight
            * self.__adversarial_loss(self.discriminator(x, pred_label), true_label)
            + self.params.hparams.classification_loss_weight
            * self.__adversarial_loss(pred_label, y)
            + loss
        )

        return loss

    def training_step(
        self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int, optimizer_idx: int
    ):
        x, labels = batch

        B, T, D = x.shape

        # # train generator
        # if optimizer_idx == 0:

        #     pred_label = self.discriminator(fake_x, labels)

        #     true_label = t.ones(B).to(x.device)
        #     loss = self.__adversarial_loss(pred_label, true_label)
        #     self.log("G_Loss", loss, prog_bar=True)
        #     return {"loss": loss}

        # train discriminator
        if optimizer_idx == 0:
            loss = self.__discriminator_loss(x, labels)
            return {"loss": loss}

        # train classifier
        if optimizer_idx == 1:

            # pred_label = self.classifier(x)
            loss = self.__classifier_loss(x, labels)
            self.log("C_Loss", loss, prog_bar=True)
            return {"loss": loss}
            return {"loss": loss}

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x[0]["loss"] for x in outputs]).mean()
        # self.log("train_loss", avg_loss, prog_bar=True)

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):

        # val_loss = discriminator_loss
        x, labels = batch

        pred = self.classifier(x)
        metric = self.amex_metric_pytorch(labels, pred)
        # metric_two = self.amex_metric_pytorch(labels, prop_two)
        loss = self.loss(pred, labels)

        # self.log("amex", metric, prog_bar=True)

        return {"val_loss": loss, "amex": metric}

        pass
        # x, labels = batch
        # y_pred = self.classifier(x)
        # loss = self.loss(y_pred, labels)

        # amex = self.amex_metric_pytorch(labels, y_pred)
        # amex = t.tensor(amex)

        # return {"val_loss": loss, "amex": amex}

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        avg_amex = t.stack([x["amex"] for x in outputs]).mean()
        # avg_amex_two = t.stack([x["amex_two"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("amex", avg_amex, prog_bar=True)
        # self.log("amex_two", avg_amex_two, prog_bar=True)

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        return

        x, labels = batch
        y_pred = self.classifier(x)
        loss = self.loss(y_pred, labels)

        # amex = self.amex_metric(labels.cpu().numpy(), y_pred.cpu().numpy())
        amex = self.amex_metric_pytorch(labels, y_pred)

        return {"test_loss": loss, "amex": amex}

    def test_epoch_end(self, outputs):
        return
        avg_loss = t.stack([x["test_loss"] for x in outputs]).mean()
        amex = t.stack([x["amex"] for x in outputs]).mean()

        return {"test_loss": avg_loss, "amex": amex}

    def configure_optimizers(self):

        # generate = mate.Optimizer(self.params.optimizer, self.generator).get_optimizer()
        discriminator = mate.Optimizer(
            self.params.optimizer, self.discriminator
        ).get_optimizer()

        classifier = mate.Optimizer(
            self.params.optimizer, self.classifier
        ).get_optimizer()

        return [discriminator, classifier]

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
