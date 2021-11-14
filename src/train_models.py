from typing import ByteString, Tuple
import argparse
from datetime import datetime
from functools import partial
import json
import os
import time

import lightgbm
from lightgbm import LGBMClassifier
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score as auc
from torch.utils.data import DataLoader

from datahandlers.dataset import collate_fn_train, collate_fn_valid, MyDataset
from datahandlers.utils import get_interaction_matrix_from_grouped_data
from metrics import compute_metrics
from models.ease import train_ease
from models.multivae import train_multivae


def get_cands(x, cols=["recs_multivae"]):

    cands = []
    for col in cols:
        cands.extend(x[col])

    cands = list(set(cands))
    cands = [y for y in cands if y not in x.train_items]

    return cands


def get_scores(x, col):

    scores = []
    for cand in x.all_candidates:
        scores.append(x[col][cand])

    return scores


def get_target(second_level_ds, label_col):

    target = ds_grouped[["userId", label_col]].explode(label_col)
    target["label"] = 1
    target.rename({label_col: "itemId"}, axis=1, inplace=True)
    second_level_ds_labeled = second_level_ds.merge(target, how="left")
    second_level_ds_labeled.label.fillna(0, inplace=True)

    return second_level_ds_labeled


def plotImp(model, X, path, num=20, fig_size=(40, 20)):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": X})
    feature_imp = feature_imp.sort_values(by="Value", ascending=False)[0:num]

    plt.figure(figsize=fig_size)
    sns.set(font_scale=5)
    sns.barplot(
        x="Value", y="Feature", data=feature_imp,
    )
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

    feature_imp.set_index("Value", inplace=True)
    logger.info(f"Feature importance of the final model: {str(feature_imp)}")


def createParser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions_path", type=str, default="../Data/processed_df.csv.gz")
    parser.add_argument("--item_col", type=str)
    parser.add_argument("--params_path", type=str, default="../Data/artifacts/params_{}.json")

    return parser


if __name__ == "__main__":

    parser = createParser()
    args, _ = parser.parse_known_args()

    logfile = os.path.join("logs", f"last_logs.log")
    # os.path.join("logs", f"{str(round(time.time()))}.log")

    with open(args.params_path.format(args.item_col), "r") as f:
        params = json.load(f)

    logger.add(logfile)
    logger.info("Делаем препроцессинг над файлом с траназкциями")

    if not os.path.exists(args.transactions_path):
        raise ValueError("The preprocessed file does not exist")

    ds_grouped = pd.read_csv(args.transactions_path)
    for col in ["train_items", "valid_items", "test_items"]:
        ds_grouped[col] = ds_grouped[col].apply(lambda x: list(map(int, x.split(";"))))

    n_items = params["n_items"]
    train = MyDataset(ds=ds_grouped, num_items=n_items, phase="train")
    valid = MyDataset(ds=ds_grouped, num_items=n_items, phase="valid")

    loaders = {
        "train": DataLoader(train, batch_size=128, collate_fn=collate_fn_train),
        "valid": DataLoader(valid, batch_size=512, collate_fn=collate_fn_valid),
    }

    logger.info("The grouped data is loaded")

    runner = train_multivae(
        path_to_save=f"../Data/artifacts/multvae_{args.item_col}",
        n_items=n_items,
        loaders=loaders,
    )

    interaction_matrix, train_inters = get_interaction_matrix_from_grouped_data(
        ds_grouped, n_items
    )

    matrix = train_ease(
        interaction_matrix=interaction_matrix,
        path_to_save=f"../Data/artifacts/ease_args{args.item_col}",
    )

    inference_loader = DataLoader(valid, batch_size=256, collate_fn=collate_fn_valid)

    preds = []

    for prediction in runner.predict_loader(loader=inference_loader):
        preds.extend(prediction.detach().cpu().numpy().tolist())

    assert len(preds) == ds_grouped.shape[0]

    ds_grouped["preds_multivae"] = preds
    ds_grouped["recs_multivae"] = ds_grouped["preds_multivae"].apply(
        lambda x: np.argsort(-np.array(x))[:50]
    )
    ds_grouped["all_candidates"] = ds_grouped.apply(lambda x: get_cands(x), axis=1)

    second_level_ds = pd.DataFrame()

    for col in ["preds_multivae"]:

        ds_grouped["scores"] = ds_grouped.apply(lambda x: get_scores(x, col=col), axis=1)
        ds_grouped["joined"] = ds_grouped.apply(
            lambda x: [f"{y1}_{y2}" for y1, y2 in zip(x.all_candidates, x.scores)], axis=1,
        )
        user_item = ds_grouped[["userId", "joined"]].explode("joined")
        user_item["itemId"] = user_item.joined.apply(lambda x: int(x.split("_")[0]))
        user_item[col] = user_item.joined.apply(lambda x: float(x.split("_")[1]))
        user_item.drop(["joined"], axis=1, inplace=True)

        if second_level_ds.shape[0] == 0:
            second_level_ds = user_item.copy()
        else:
            second_level_ds = second_level_ds.merge(user_item)

    user_info = pd.read_csv("../Data/train_2.csv", sep=";")

    USER_FEATURES = [
        "MM_IN_BANK",
        "MM_W_CARD",
        "AGE",
        "GENDER",
        "EDUCATION_LEVEL",
        "MARITAL_STATUS",
        "DEPENDANT_CNT",
        "INCOME_MAIN_AMT",
    ]

    for col in USER_FEATURES:
        user_info[col] = user_info[col].apply(lambda x: float(str(x).replace(",", ".")))

    with open(f"../Data/artifacts/user2idx_{args.item_col}.json", "r") as f:
        user2idx = json.load(f)

    user_info.rename({"ID": "userId"}, axis=1, inplace=True)
    user_info["userId"] = user_info["userId"].astype(str)
    user_info = user_info.loc[user_info.userId.isin(user2idx.keys())]
    user_info["userId"] = user_info["userId"].apply(lambda x: user2idx[x])

    second_level_ds = second_level_ds.merge(user_info[USER_FEATURES + ["userId"]])
    second_level_ds.fillna(0, inplace=True)
    second_level_ds.head()

    merch_info = (
        train_inters.merge(user_info)
        .groupby("itemId")
        .agg({k: "mean" for k in USER_FEATURES})
        .reset_index()
    )

    merch_info.rename({k: f"merch_mean_{k}" for k in USER_FEATURES}, axis=1, inplace=True)
    MERCH_FEATURES = [f"merch_mean_{k}" for k in USER_FEATURES]
    second_level_ds = second_level_ds.merge(merch_info)
    second_level_ds.head()

    train_df = get_target(second_level_ds, "valid_items")
    test_df = get_target(second_level_ds, "test_items")

    FEATURES = ["preds_multivae"] + USER_FEATURES + MERCH_FEATURES

    bst = lightgbm.LGBMClassifier()

    train_df["INCOME_MAIN_AMT"] = train_df["INCOME_MAIN_AMT"].astype(np.float32)
    test_df["INCOME_MAIN_AMT"] = test_df["INCOME_MAIN_AMT"].astype(np.float32)

    bst.fit(train_df[FEATURES], train_df["label"])

    test_df["preds_boosting"] = bst.predict_proba(test_df[FEATURES])[:, 1]
    auc_value = auc(test_df["label"], test_df["preds_boosting"])
    print(f"auc on test: {auc_value}")

    plotImp(
        model=bst, X=FEATURES, path=f"../Data/artifacts/feature_importance_{args.item_col}.png",
    )
    bst.booster_.save_model(f"../Data/artifacts/boosting_{args.item_col}.txt")

    logger.info(
        compute_metrics(
            test_df.sort_values(by="userId")[:100000], models=["preds_multivae", "preds_boosting"],
        )
    )
