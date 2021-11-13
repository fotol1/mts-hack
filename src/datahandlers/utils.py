from loguru import logger
import pandas as pd
from scipy import sparse as sps


def filter_column(ds, col, min_freq, free_col="timestamp"):

    temp = ds.groupby(col, as_index=False).agg({free_col: "count"})
    filtered = temp.loc[temp[free_col] >= min_freq, col].values

    ds = ds.loc[ds[col].isin(filtered)].copy()

    return ds


def get_n_users_n_items(ds):
    n_users = ds.userId.nunique()
    n_items = ds.itemId.nunique()
    logger.info(f"N_users: {n_users}, n_items: {n_items}")


def encode_data(ds):

    ds["userId"] = ds["userId"].astype(str)
    ds["itemId"] = ds["itemId"].astype(str)

    user2idx = {v: k for k, v in enumerate(ds.userId.unique())}
    item2idx = {v: k for k, v in enumerate(ds.itemId.unique())}

    ds["userId"] = ds.userId.apply(lambda x: user2idx[x])
    ds["itemId"] = ds.itemId.apply(lambda x: item2idx[x])
    logger.info("Items and users are encoded")

    return ds, user2idx, item2idx


def get_label(x):

    l = len(x)

    train_labels = []
    valid_labels = []
    test_labels = []

    phase = "train"
    for i in range(len(x)):
        if phase == "train":
            train_labels.append(x[i])
        elif phase == "valid":
            valid_labels.append(x[i])
        else:
            test_labels.append(x[i])

        if phase == "train" and i + 1 >= 0.8 * l:
            phase = "valid"
        elif phase == "valid" and i + 1 >= 0.9 * l:
            phase = "test"

    assert len(train_labels) * len(valid_labels) * len(test_labels) > 0, l

    return (train_labels, valid_labels, test_labels)


def get_interaction_matrix_from_grouped_data(ds_grouped, n_items):

    train_inters = []
    for idx, row in ds_grouped.iterrows():
        train_interactions = row.train_items

        train_inters.extend([[row.userId, x] for x in train_interactions])

    train_inters = pd.DataFrame(train_inters, columns=["userId", "itemId"])
    train_inters["freq"] = 1
    train_inters = (
        train_inters.groupby(["userId", "itemId"]).agg({"freq": "sum"}).reset_index()
    )

    matrix = sps.coo_matrix(
        (train_inters["freq"], (train_inters["userId"], train_inters["itemId"])),
        shape=(ds_grouped.shape[0], n_items),
    )

    return matrix, train_inters
