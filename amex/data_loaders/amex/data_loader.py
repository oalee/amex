import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler

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

        self.randomize_split = False
        # num workers = number of cpus to use
        # get number of cpu's on this device
        num_workers = os.cpu_count()
        self.num_workers = num_workers
        # self.sc = StandardScaler()
        # self._prepare_data()
        self.prepare_tensor_data()
        return
        print("Loading data...")

        self.prepare_tensor_data()
        print("Setting up data...")
        # self.prepare_data()
        print("Data loaded.")

    def load_torch_tensor(self):
        tensor_dict = torch.load(os.path.join(self.data_location, "tensor.pt"))
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

        if self.randomize_split:

            false_indxes = (y == 0).nonzero().squeeze()
            true_indxes = (y == 1).nonzero().squeeze()

            all_false = x[false_indxes]
            all_true = x[true_indxes]
            # ipdb.set_trace()

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

        else:

            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

            self.train_tensor = TensorDataset(train_x, train_y)
            self.val_tensor = TensorDataset(test_x, test_y)
            self.test_tensor = TensorDataset(test_x, test_y)

        print("Shape of train data: ", tensors[0].shape)

    def _prepare_data(self):
        # All data comumns except customer_ID, target, and S_2 are features
        self.all_data = self.load_train_df(
            os.path.join(self.data_location, "train.parquet"),
            os.path.join(self.data_location, "train_labels.csv"),
        )
        features = self.all_data.columns[2:-1]

        # drop columns with more than 60% missing values
        min_count = int(0.6 * self.all_data.shape[0] + 1)
        self.all_data = self.all_data.dropna(axis=1, thresh=min_count)
        ipdb.set_trace()

        # drop all columns with same value in all rows
        nonunique = self.all_data.nunique(0)
        z_unique = nonunique[nonunique == 0].index
        cols_to_drop = nonunique[nonunique == 1].index
        tabular_cols = nonunique[nonunique < 10].index
        ipdb.set_trace()

        # self.all_data = self.all_data.loc[

        # tabular_featue_columns = ["D_63" ,"D_64","D_68", "B_30" , "B_38", "D_114","D_117", "D_120", "D_126"
        # binary_feature_columns = ["D_114", "D_120"]

        # ipdb.set_trace()
        self.all_data[features] = self.sc.fit_transform(self.all_data[features])
        self.all_data[features] = self.all_data[features].fillna(-100)

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
        torch.save(tensor_dict, os.path.join(self.data_location, "tensor_n_dict.pt"))
        print("Tensor saved.")

        # TRAIN
        self.train_tensor = TensorDataset(X_train, y_train)
        # VAL
        self.val_tensor = TensorDataset(X_val, y_val)
        # TEST
        self.test_tensor = TensorDataset(X_test, y_test)

    def train_dataloader(self):
        # sampler =WeightedRandomSampler()
        # class_one = t.sum(self.train_tensor.tensors[1] == 1)
        # class_zero = t.sum(self.train_tensor.tensors[1] == 0)
        # print("Class one: ", class_one)
        # print("Class zero: ", class_zero)
        # tot = class_one + class_zero

        # # weight so that we sample equally from both classes
        # w_class_1 = class_zero / class_one
        # w_class_0 = 1

        # weights = torch.tensor(
        #     [0.3 if y == 1 else 0.1 for y in self.train_tensor.tensors[1]]
        # )

        # # ipdb.set_trace()

        # sampler = WeightedRandomSampler(
        #     weights, num_samples=len(weights), replacement=True
        # )

        return DataLoader(
            self.train_tensor,
            # sampler=sampler,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
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
