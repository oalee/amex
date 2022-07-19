import pandas as pd
import numpy as np
import ipdb
from pyparsing import col
import torch as t

data_location = "train_data.parquet"
train_labels = "train_labels.csv"


def load_data():
    df = pd.read_parquet(data_location)
    return df


"""
fills the rest of missing sequences by NA
"""


def fill_missing_time(df):
    df["S_2"] = pd.to_datetime(df["S_2"])
    tmp = df[["customer_ID", "S_2"]].groupby("customer_ID").count()

    missing_cids = []
    for nb_available_rows in range(1, 14):
        cids = tmp[tmp["S_2"] == nb_available_rows].index.values
        batch_missing_cids = [
            cid for cid in cids for _ in range(13 - nb_available_rows)
        ]
        missing_cids.extend(batch_missing_cids)

    train_part2 = df.iloc[: len(missing_cids)].copy()
    train_part2.loc[:] = np.nan  # fill the rest with nan
    train_part2["customer_ID"] = missing_cids

    train = pd.concat([train_part2, df])

    train = train.sort_values("customer_ID")

    print("Filled missing time:", len(missing_cids))

    return train


def drop_na_column(df: pd.DataFrame, thresh=0.6):

    thresh = int(thresh * df.shape[0])
    df = df.dropna(axis=1, thresh=thresh)
    # df.fillna(value=None)  # Fill None with nan!
    return df


def process_tabular(df):

    print("Processing tabular data")

    nununique = df.nunique(0)

    max = 12
    columns = []
    for i in range(max):
        idx = (nununique[nununique == i + 1]).index

        if len(idx) > 0:
            process_tabular_column(df, idx, i + 1)
            columns.append((idx, i + 1))

    c_ordered = ["customer_ID", "S_2"]
    try:
        for i in range(len(columns)):
            c_ordered.extend(columns[i][0].tolist())
            print(columns[i][0], columns[i][1])

        remaining_columns = [col for col in df.columns[2:] if col not in c_ordered]
        c_ordered.extend(remaining_columns)

        df = df.loc[:, c_ordered]
    except:
        ipdb.set_trace()

    print("Processed tabular:", len(c_ordered))

    return df


def process_tabular_column(df: pd.DataFrame, columns, enum_classes):

    # reformat values in df[column] to start from 0 and end to number of enum_classes
    for column in columns:
        uniques = df[column].unique().tolist()
        uniques = [item for item in uniques if not pd.isna(item)]
        uniques = sorted(uniques)

        reformated_values = np.arange(enum_classes)

        for i in range(len(uniques)):
            try:
                if uniques[i] == reformated_values[i]:
                    continue
            except:
                ipdb.set_trace()

            df[column] = df[column].replace(uniques[i], reformated_values[i])


def save_tensor(df):
    labels = pd.read_csv(train_labels)
    labels = labels.sort_values("customer_ID")
    df = df.sort_values(["customer_ID", "S_2"])  # sort by customer_ID and time
    df.fillna(value=np.nan)  # Fill None with nan
    tensor = df[df.columns[2:]].to_numpy()
    # ipdb.set_trace()
    tensor = t.reshape(t.tensor(tensor, dtype=t.float), (-1, 13, 157))
    all_tensor_y = t.tensor(labels["target"].to_numpy(), dtype=t.float)
    dict = {"x": tensor, "y": all_tensor_y}
    t.save(dict, "tensor.pt")

    print("Saved tensor")
    return df


def preprocess():
    df = load_data()
    df = fill_missing_time(df)
    df = drop_na_column(df)
    df = process_tabular(df)
    save_tensor(df)
    return df


preprocess()
