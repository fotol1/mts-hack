# flake8: noqa
from typing import Dict, List, Tuple

from catalyst import dl, metrics
from catalyst.contrib.datasets import MovieLens
from catalyst.utils import get_device, set_global_seed
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_normal_
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

set_global_seed(42)
device = get_device()


class MyDataset(Dataset):
    def __init__(self, ds, num_items, phase="valid"):
        super().__init__()
        self.ds = ds
        self.phase = phase
        self.n_items = num_items

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

        row = self.ds.iloc[idx]

        x_input = np.zeros(self.n_items)
        x_input[row["train_items"]] = 1

        seq_input = row["train_items"][:-1]
        seq_target = row["train_items"][1:]

        targets = np.zeros(self.n_items)

        if self.phase == "train":
            return (x_input, seq_input, seq_target)
        elif self.phase == "valid":
            targets[row["valid_items"]] = 1
        else:
            targets[row["test_items"]] = 1

        return (x_input, targets, seq_input, seq_target)


def collate_fn_train(batch: List[Tuple[torch.Tensor]]) -> Dict[str, torch.Tensor]:

    x, seq_i, seq_t = zip(*batch)

    inputs = pad_sequence([torch.Tensor(t) for t in x]).T
    seq_len = torch.Tensor([len(x) for x in seq_i])
    seq_i = pad_sequence([torch.Tensor(t) for t in seq_i]).T
    seq_t = pad_sequence([torch.Tensor(t) for t in seq_t]).T

    return {"inputs": inputs, "seq_i": seq_i, "seq_t": seq_t, "seq_len": seq_len}


def collate_fn_valid(batch: List[Tuple[torch.Tensor]]) -> Dict[str, torch.Tensor]:

    x, y, seq_i, seq_t = zip(*batch)

    inputs = pad_sequence([torch.Tensor(t) for t in x]).T
    seq_len = torch.Tensor([len(x) for x in seq_i])
    seq_i = pad_sequence([torch.Tensor(t) for t in seq_i]).T
    seq_t = pad_sequence([torch.Tensor(t) for t in seq_t]).T

    targets = pad_sequence([torch.Tensor(t) for t in y]).T

    return {
        "inputs": inputs,
        "targets": targets,
        "seq_i": seq_i,
        "seq_t": seq_t,
        "seq_len": seq_len,
    }
