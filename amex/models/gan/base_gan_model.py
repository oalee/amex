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
        self.generator: t.nn.Module
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

        pred_label = self.discriminator(x, y)

        rand_y = t.rand_like(y)
        rand_y = (rand_y > 0.5).float()

        fake_x = self.generator(rand_y)
        pred_fake_x = self.discriminator(fake_x, rand_y)

        true_label = t.ones(*y.shape).to(x.device)
        fake_label = t.zeros(*y.shape).to(x.device)

        fake_loss = self.__adversarial_loss(pred_fake_x, fake_label)
        real_loss = self.__adversarial_loss(pred_label, true_label)

        # flip 0s and 1s in y
        y_flip = 1 - y
        pred_label_flip = self.discriminator(x, y_flip)
        real_loss_flip = self.__adversarial_loss(pred_label_flip, fake_label)

        self.log("F_Flip_Loss", real_loss_flip, prog_bar=True)
        self.log("D_Fake_Loss", fake_loss, prog_bar=True)
        self.log("D_Real_Loss", real_loss, prog_bar=True)

        loss = fake_loss + 2* real_loss + real_loss_flip
        return loss / 4

        return t_loss

    def __classifier_loss(self, x: t.Tensor, y):

        if self.current_epoch > self.params.hparams.warmup_epochs:
            # randomly select x or generate x
            prob = t.rand(1)[0]
            if prob > 0.5:
                random_labels = t.rand_like(y, device=x.device)
                x = self.generator(random_labels)
                d_pred = self.discriminator(x, random_labels)

                # select instances where discriminator predicts as real
                x = x[d_pred.flatten() > 0.5]
                y = random_labels[d_pred.flatten() > 0.5]

        pred_label = self.classifier(x)

        loss = self.__adversarial_loss(pred_label, y)

        return loss

    def __generator_loss(self, y):
        fake_x = self.generator(y)
        pred_fake_x = self.discriminator(fake_x, y)
        true_label = t.ones(*y.shape).to(y.device)

        loss = self.__adversarial_loss(pred_fake_x, true_label)
        return loss

    def training_step(
        self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int, optimizer_idx: int
    ):
        x, labels = batch

        B, T, D = x.shape

        random_labels = t.rand_like(labels)
        # convert to zero and ones
        random_labels = (random_labels > 0.5).float()

        # # train generator
        if optimizer_idx == 0:

            loss = self.__generator_loss(random_labels)
            self.log("G_Loss", loss, prog_bar=True)
            return {"loss": loss}

        # train discriminator
        if optimizer_idx == 1:
            loss = self.__discriminator_loss(x, labels)
            # self.log("D_Loss", loss, prog_bar=True)
            return {"loss": loss}

        # train classifier
        if optimizer_idx == 2:

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
        p_2 = pred
        metric = self.amex_metric_pytorch(labels, pred)
        # metric_two = self.amex_metric_pytorch(labels, prop_two)
        loss = self.loss(pred, labels)

        # self.log("amex", metric, prog_bar=True)

        return {"val_loss": loss, "amex": metric}

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        avg_amex = t.stack([x["amex"] for x in outputs]).mean()

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("amex", avg_amex, prog_bar=True)

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

        generator = mate.Optimizer(
            self.params.optimizer, self.generator
        ).get_optimizer()
        discriminator = mate.Optimizer(
            self.params.optimizer, self.discriminator
        ).get_optimizer()

        classifier = mate.Optimizer(
            self.params.optimizer, self.classifier
        ).get_optimizer()

        return [generator, discriminator, classifier]

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
