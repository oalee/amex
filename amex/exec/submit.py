from torch import nn
import torch as t
import pandas as pd
import numpy as np
import ipdb
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def run(model: nn.Module):

    model.eval()
    model.to("cuda")

    input = t.load("./amex/exec/test_tensor.pt")
    df = pd.read_csv("./amex/exec/test_customer_ids.csv")
    batch_size = 32
    tot_size = input.shape[0]
    num_batches = int(tot_size / batch_size)
    preds = []

    with t.no_grad():
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch = input[start:end].to("cuda")
            pred = model(batch)
            preds += [pred.flatten().cpu()]
            del batch

    preds = t.stack(preds).flatten()
    
    # ipdb.set_trace()
    remaining = tot_size - num_batches * batch_size
    if remaining > 0:
        batch = input[-remaining:].to("cuda")
        pred = model(batch)
        preds = t.cat([preds, pred.flatten().cpu()], dim=0)

    # ipdb.set_trace()
    # all_preds = t.stack(preds[:-1])
    preds = preds.flatten().detach().numpy()

    # ipdb.set_trace()
    df["prediction"] = preds
    df.to_csv("./amex/exec/submission.csv", index=False)


def load_test_df(test_file):
    test = pd.read_parquet(test_file)
    # ipdb.set_trace()
    test["S_2"] = pd.to_datetime(test["S_2"])

    # count of unique customer_ID's
    tmp = test[["customer_ID", "S_2"]].groupby("customer_ID").count()
    print(tmp.size)

    # ipdb.set_trace()

    # missing_cids = []
    # for nb_available_rows in range(1, 14):
    #     cids = tmp[tmp["S_2"] == nb_available_rows].index.values
    #     batch_missing_cids = [
    #         cid for cid in cids for _ in range(13 - nb_available_rows)
    #     ]
    #     missing_cids.extend(batch_missing_cids)

    # train_part2 = test.iloc[: len(missing_cids)].copy()
    # train_part2.loc[:] = np.nan
    # train_part2["customer_ID"] = missing_cids

    # test =  pd.concat([train_part2, test])

    # # test = test.sort_values("customer_ID")

    # sc = StandardScaler()
    # features = test.columns[2:]
    # test[features] = sc.fit_transform(test[features])
    # test[features] = test[features].fillna(0)

    # test = test.sort_values("customer_ID")
    # # tmp = test[["customer_ID", "S_2"]].groupby("customer_ID").count()
    # # print(tmp.size)
    # # ipdb.set_trace()

    # all_tensor_x = t.reshape(
    #     t.tensor(test[features].to_numpy()), (-1, 13, 188)
    # ).float()
    # t.save(all_tensor_x, "./test_tensor.pt")

    # first column is customer_ID, save it to a separate file
    # only unique customer_ID's
    # ipdb.set_trace()
    df = pd.DataFrame()
    df["customer_ID"] = test["customer_ID"].unique()
    df.to_csv("./test_customer_ids.csv", index=False)

    return test


# test_file = "test.parquet"
# test = load_test_df(test_file)
