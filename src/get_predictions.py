import argparse
from datetime import datetime
from functools import partial
import json
import os
import time

from catalyst import dl, metrics
from catalyst.contrib.datasets import MovieLens
from catalyst.utils import get_device, set_global_seed
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_normal_
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from datahandlers.dataset import collate_fn_train, collate_fn_valid, MyDataset
from datahandlers.utils import encode_data, filter_column, get_label, get_n_users_n_items
from models.multivae import MultiVAE, RecSysRunner


def read_data_and_split(path, item_col, nrows, params_path):

    ds = pd.read_csv(path, sep=";")

    ds["itemId"] = ds["MCC_CODE"].copy()

    ds.rename({"ID": "userId", "TRANS_DTTM": "timestamp"}, axis=1, inplace=True)

    ds, user2idx, item2idx = encode_data(ds)

    """ Train/valid/test split"""

    ds_grouped = (
        ds.groupby("userId")
        .apply(lambda x: [y for y, _ in sorted(zip(x.itemId, x.timestamp), key=lambda x: x[1])])
        .reset_index()
    )
    ds_grouped.rename({0: "items"}, axis=1, inplace=True)
    ds_grouped["len_items"] = ds_grouped["items"].apply(lambda x: len(x))

    return ds_grouped


def createParser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction_path", type=str, default="../Data/test_1.csv")
    parser.add_argument("--item_col", type=str, default="mcc")
    parser.add_argument("--min_user_freq", type=int, default=10)
    parser.add_argument("--nrows", type=int, default=-1)
    parser.add_argument("--output_path", type=str, default="../Data/predictions.csv.gz")
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
        path=args.transaction_path,
        item_col=args.item_col,
        nrows=args.nrows,
        params_path=args.params_path,
    )
    ds_grouped.rename({"items": "train_items"}, axis=1, inplace=True)
    ds_grouped["valid_items"] = ds_grouped["train_items"].copy()

    model = MultiVAE([200, 600, 50], dropout=0.0)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    engine = dl.DeviceEngine("cpu")
    hparams = {
        "anneal_cap": 0.2,
        "total_anneal_steps": 6000,
    }

    callbacks = [
        dl.NDCGCallback("logits", "targets", [20, 50, 100]),
        # dl.MAPCallback("logits", "targets", [20, 50, 100]),
        # dl.MRRCallback("logits", "targets", [20, 50, 100]),
        # dl.HitrateCallback("logits", "targets", [20, 50, 100]),
        dl.OptimizerCallback("loss", accumulation_steps=1),
        dl.EarlyStoppingCallback(
            patience=2, loader_key="valid", metric_key="ndcg20", minimize=False
        ),
    ]

    train = MyDataset(ds=ds_grouped, num_items=50, phase="train")

    inference_loader = DataLoader(train, batch_size=256, collate_fn=collate_fn_train)
    dumm_loader = DataLoader(train, batch_size=256, collate_fn=collate_fn_valid)
    loaders = {"train": inference_loader, "valid": dumm_loader}

    runner = RecSysRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        engine=engine,
        hparams=hparams,
        scheduler=lr_scheduler,
        loaders=loaders,
        num_epochs=0,
        verbose=True,
        timeit=False,
        callbacks=callbacks,
        # logdir="./logs",
    )
    model.load_state_dict(torch.load("../Data/artifacts/multvae_mcc"))
    model.eval()

    preds = []

    for prediction in runner.predict_loader(loader=inference_loader):
        preds.extend(prediction.detach().cpu().numpy().tolist())

    assert len(preds) == ds_grouped.shape[0]

    ds_grouped["preds_multivae"] = preds
    ds_grouped["recs_multivae"] = ds_grouped["preds_multivae"].apply(
        lambda x: np.argsort(-np.array(x))[:50]
    )
    ds_grouped = ds_grouped[["userId", "recs_multivae"]].explode("recs_multivae")
    ds_grouped["recs_multivae"] = ds_grouped["recs_multivae"].astype(np.int32)
    ds_grouped["rank"] = ds_grouped.groupby("userId")["recs_multivae"].cumcount() + 1

    with open("../Data/artifacts/item2idx_mcc.json", "r") as f:
        enc = json.load(f)

    dec = {k: v for v, k in enc.items()}

    ds_grouped["recs_multivae"] = ds_grouped["recs_multivae"].apply(lambda x: dec.get(x, -1))
    ds_grouped.to_csv(args.output_path, index=None, compression="gzip")
    print(ds_grouped.head())
