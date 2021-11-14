import argparse
from datetime import datetime
from functools import partial
import json
import os
import time

from loguru import logger
import numpy as np
import pandas as pd

from datahandlers.dataset import MyDataset
from datahandlers.utils import encode_data, filter_column, get_label, get_n_users_n_items


def read_data_and_split(path, item_col, nrows, params_path):

    ds = pd.read_csv(path, sep=";", nrows=nrows)

    if item_col == "mcc":
        ds["itemId"] = ds["MCC_CODE"].copy()
    elif item_col == "merchant":
        ds["itemId"] = ds["LOCATION_NAME"].apply(lambda x: x.split("\\")[0])
        ds = ds.loc[~ds.LOCATION_NAME.isna()]
    else:
        raise ValueError(f"wrong item_Col {item_col}")

    ds.rename({"ID": "userId", "TRANS_DTTM": "timestamp"}, axis=1, inplace=True)

    get_n_users_n_items(ds)

    ds = filter_column(ds, col="itemId", min_freq=50)
    ds = filter_column(ds, col="userId", min_freq=10)

    get_n_users_n_items(ds)

    if ds.shape[0] == 0:
        raise ValueError("Not enough rows for creating models")

    ds, user2idx, item2idx = encode_data(ds)

    """ Train/valid/test split"""

    ds_grouped = (
        ds.groupby("userId")
        .apply(lambda x: [y for y, _ in sorted(zip(x.itemId, x.timestamp), key=lambda x: x[1])])
        .reset_index()
    )
    ds_grouped.rename({0: "items"}, axis=1, inplace=True)
    ds_grouped["len_items"] = ds_grouped["items"].apply(lambda x: len(x))

    ds_grouped["items"] = ds_grouped["items"].apply(lambda x: get_label(x))
    ds_grouped["train_items"] = ds_grouped["items"].apply(lambda x: ";".join(map(str, x[0])))
    ds_grouped["valid_items"] = ds_grouped["items"].apply(lambda x: ";".join(map(str, x[1])))
    ds_grouped["test_items"] = ds_grouped["items"].apply(lambda x: ";".join(map(str, x[2])))
    ds_grouped.drop(["items"], axis=1, inplace=True)

    params = {"n_items": len(item2idx)}
    with open(params_path.format(item_col), "w") as f:
        json.dump(params, f)

    with open(f"../Data/artifacts/user2idx_{item_col}.json", "w") as f:
        json.dump(user2idx, f)

    with open(f"../Data/artifacts/item2idx_{item_col}.json", "w") as f:
        json.dump(item2idx, f)

    return ds_grouped


def createParser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions_path", type=str, default="../Data/train_1.csv")
    parser.add_argument("--item_col", type=str, default="mcc")
    parser.add_argument("--min_user_freq", type=int, default=10)
    parser.add_argument("--nrows", type=int, default=-1)
    parser.add_argument("--output_path", type=str, default="../Data/processed_df.csv.gz")
    parser.add_argument("--params_path", type=str, default="../Data/artifacts/params_{}.json")

    return parser


if __name__ == "__main__":

    parser = createParser()
    args, _ = parser.parse_known_args()

    if args.nrows == -1:
        args.nrows = None

    logfile = os.path.join("logs", f"last_logs.log")

    if os.path.exists(logfile):
        os.remove(logfile)

    logger.add(logfile)
    logger.info("Делаем препроцессинг над файлом с траназкциями")

    ds_grouped = read_data_and_split(
        path=args.transactions_path,
        item_col=args.item_col,
        nrows=args.nrows,
        params_path=args.params_path,
    )
    logger.info("Grouped dataframe:")
    logger.info(str(ds_grouped))

    ds_grouped.to_csv(args.output_path, compression="gzip", index=None)
    logger.info("Transaction data is ready")
