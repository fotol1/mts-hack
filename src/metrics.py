from typing import List

import numpy as np
import pandas as pd


def average_precision(is_relevant, pos_items_num):

    if len(is_relevant) == 0:
        a_p = 0.0
    else:
        p_at_k = (
            is_relevant
            * np.cumsum(is_relevant, dtype=np.float32)
            / (1 + np.arange(is_relevant.shape[0]))
        )
        a_p = np.sum(p_at_k) / np.min([pos_items_num, is_relevant.shape[0]])

    assert 0 <= a_p <= 1, a_p
    return a_p


def recall(is_relevant, pos_items_num):

    recall_score = np.sum(is_relevant, dtype=np.float32) / pos_items_num

    assert 0 <= recall_score <= 1, recall_score
    return recall_score


def get_metric(x, feature: str, label: str, metric: str, at: int = 10):

    if feature not in x.columns:
        raise ValueError(f"{feature}-колонка со скорами ранжирования не в данных")
    if label not in x.columns:
        raise ValueError(f"Нет колонки label в финальных данных")

    x = x.sort_values(by=feature, ascending=False)

    if x.label.sum() == 0:
        return None
    elif metric == "map":
        return average_precision(x["label"].values[:at], pos_items_num=x.label.sum())
    elif metric == "recall":
        return recall(x["label"].values[:at], pos_items_num=x.label.sum())
    else:
        raise NotImplementedError(f"{metric} is not implemented")


def compute_metrics(
    test_df: pd.DataFrame,
    models: List[str],
    ks: List[int] = [1, 5, 10],
    metrics: List[str] = ["map", "recall"],
) -> pd.DataFrame:

    metrics_list = []

    for k in ks:
        for metric in metrics:
            for model in models:
                metric_value = (
                    test_df.groupby(["userId"])
                    .apply(
                        lambda x: get_metric(x, feature=model, label="label", metric=metric, at=k)
                    )
                    .mean()
                )
                metrics_list.append((f"{metric}_{k}", model, metric_value))

    metrics_df = pd.DataFrame(metrics_list, columns=["metric", "model", "value"])
    metrics_df = pd.pivot_table(metrics_df, columns="metric", values="value", index="model")

    metrics_df.sort_values(by="map_5", inplace=True)

    return metrics_df
