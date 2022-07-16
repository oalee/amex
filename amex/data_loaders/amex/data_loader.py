import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset

import pandas as pd
import numpy as np
import os
import ipdb
import torch as t


class CustomDataModule(pl.LightningDataModule):
    def load_train_df(self, train_file, train_labels):
        # ipdb.set_trace()
        train = pd.read_parquet(train_file, engine="fastparquet")
        train["S_2"] = pd.to_datetime(train["S_2"])
        tmp = train[["customer_ID", "S_2"]].groupby("customer_ID").count()

        missing_cids = []
        for nb_available_rows in range(1, 14):
            cids = tmp[tmp["S_2"] == nb_available_rows].index.values
            batch_missing_cids = [
                cid for cid in cids for _ in range(13 - nb_available_rows)
            ]
            missing_cids.extend(batch_missing_cids)

        train_part2 = train.iloc[: len(missing_cids)].copy()
        train_part2.loc[:] = np.nan
        train_part2["customer_ID"] = missing_cids

        train = pd.concat([train_part2, train])

        train = train.sort_values("customer_ID")

        train_labels = pd.read_csv(train_labels)
        train = pd.merge(train, train_labels, how="inner", on="customer_ID")

        train = train.sort_values("customer_ID")
        return train

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.data_location = params.data_location
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size

        # num workers = number of cpus to use
        # get number of cpu's on this device
        num_workers = os.cpu_count()
        self.num_workers = num_workers
        self.sc = StandardScaler()

        print("Loading data...")
        # self.all_data = self.load_train_df(
        #     os.path.join(self.data_location, "train.parquet"),
        #     os.path.join(self.data_location, "train_labels.csv"),
        # )
        self.prepare_tensor_data()
        print("Setting up data...")
        # self.prepare_data()
        print("Data loaded.")

    def load_torch_tensor(self):
        tensor_dict = torch.load(os.path.join(self.data_location, "tensor_dict.pt"))
        return tensor_dict["x"], tensor_dict["y"]

    def prepare_tensor_data(self):
        tensors = self.load_torch_tensor()

        # ipdb.set_trace()
        x = tensors[0]

        if self.params.contains("normalization"):

            maxes = torch.tensor(
                [torch.max(tensors[0][:, :, i]) for i in range(tensors[0].shape[2])]
            )
            mins = torch.tensor(
                [torch.min(tensors[0][:, :, i]) for i in range(tensors[0].shape[2])]
            )
            normalize = lambda x: 2 * (x - mins) / (maxes - mins) - 1

            if self.params.normalization == "tanh":
                x = normalize(x)
            elif self.params.normalization == "sigmoid":
                normalize = lambda x: (x - mins) / (maxes - mins)
                x = normalize(x)

        y = tensors[1]

        all_false = t.stack([x[i] for i in range(x.shape[0]) if y[i] == 0])
        all_true = t.stack([x[i] for i in range(x.shape[0]) if y[i] == 1])

        all_true_train, all_true_test = train_test_split(
            all_true, test_size=0.1, random_state=1
        )
        all_false_train, all_false_test = train_test_split(
            all_false, test_size=0.1, random_state=1
        )

        # ipdb.set_trace()
        self.train_tensor = CustomDataset(all_true_train, all_false_train)

        self.val_tensor = CustomDataset(all_true_test, all_false_test)

        self.test_tensor = CustomDataset(all_true_test, all_false_test)

        print("Shape of train data: ", tensors[0].shape)

    def _prepare_data(self):
        # All data comumns except customer_ID, target, and S_2 are features
        features = self.all_data.columns[2:-1]
        self.all_data[features] = self.sc.fit_transform(self.all_data[features])
        self.all_data[features] = self.all_data[features].fillna(0)

        # https://www.kaggle.com/competitions/amex-default-prediction/discussion/327828 !! Many Thanks @Chris Deotte for your sharing
        all_tensor_x = torch.reshape(
            torch.tensor(self.all_data[features].to_numpy()), (-1, 13, 188)
        ).float()
        all_tensor_y = torch.tensor(
            self.all_data.groupby("customer_ID").first()["target"].to_numpy()
        ).float()

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            all_tensor_x, all_tensor_y, test_size=0.1, random_state=1
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.1, random_state=1
        )

        # save tensors to file
        tensor_dict = {
            "x": all_tensor_x,
            "y": all_tensor_y,
        }
        torch.save(tensor_dict, os.path.join(self.data_location, "tensor_dict.pt"))

        # TRAIN
        self.train_tensor = CustomDataset(X_train, y_train)
        # VAL
        self.val_tensor = CustomDataset(X_val, y_val)
        # TEST
        self.test_tensor = CustomDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_tensor,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_tensor,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_tensor,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )


class CustomDataset(Dataset):
    def __init__(self, false_samples: t.Tensor, true_samples: t.Tensor):

        self.false_samples = false_samples
        self.true_samples = true_samples

        # length is minimum of the two
        self.length = min(self.false_samples.shape[0], self.true_samples.shape[0])
        self.true_sample_size = self.true_samples.shape[0]
        self.false_sample_size = self.false_samples.shape[0]
        print("False samples: ", self.false_samples.shape)
        print("True samples: ", self.true_samples.shape)

    def __len__(self):
        return 100000

    def __getitem__(self, idx):

        # randomly return false or true
        if torch.rand(1)[0] > 0.5:
            idx = torch.randint(0, self.false_sample_size, (1,))[0]
            return self.false_samples[idx], t.zeros(1)

        else:
            idx = torch.randint(0, self.true_sample_size, (1,))[0]
            return self.true_samples[idx], t.ones(1)

        return false_sample, true_sample

        return self.data[idx], self.target[idx]
