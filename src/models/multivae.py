# flake8: noqa
from typing import Dict, List, Tuple

from catalyst import dl, metrics
from catalyst.contrib.datasets import MovieLens
from catalyst.utils import get_device, set_global_seed
from loguru import logger
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_normal_
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

set_global_seed(42)
device = get_device()


class MultiVAE(nn.Module):
    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super().__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])]
        )
        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]
        )

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        mu, logvar = self.encode(input)  # (bs, num_items)
        z = self.reparameterize(mu, logvar)  # (bs, hidden_him)
        return self.decode(z), mu, logvar

    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, : self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1] :]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            xavier_normal_(layer.weight.data)
            constant_(layer.bias.data, 0)

        for layer in self.p_layers:
            xavier_normal_(layer.weight.data)
            constant_(layer.bias.data, 0)


class RecSysRunner(dl.Runner):
    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss_ae", "loss_kld", "loss"]
        }

    def handle_batch(self, batch):

        x = batch["inputs"]  # (bs, num_items)

        if "targets" in batch:
            x_true = batch["targets"]

        x_recon, mu, logvar = self.model(x)

        anneal = min(
            self.hparams["anneal_cap"],
            self.global_batch_step / self.hparams["total_anneal_steps"],
        )

        loss_ae = -torch.mean(torch.sum(F.log_softmax(x_recon, 1) * x, -1))
        loss_kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        loss = loss_ae + anneal * loss_kld

        self.batch.update({"logits": x_recon, "inputs": x, "targets": x})
        if "targets" in batch:
            self.batch.update({"targets": batch["targets"]})

        self.batch_metrics.update({"loss_ae": loss_ae, "loss_kld": loss_kld, "loss": loss})
        for key in ["loss_ae", "loss_kld", "loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

    def on_loader_end(self, runner):
        for key in ["loss_ae", "loss_kld", "loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def predict_batch(self, batch):
        x = batch["inputs"]
        x_recon, mu, logvar = self.model(x)
        return x_recon


def train_multivae(path_to_save, n_items, loaders):

    model = MultiVAE([200, 600, n_items], dropout=0.5)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    engine = dl.DeviceEngine(device)
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

    runner = RecSysRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        engine=engine,
        hparams=hparams,
        scheduler=lr_scheduler,
        loaders=loaders,
        num_epochs=10,
        verbose=True,
        timeit=False,
        callbacks=callbacks,
        # logdir="./logs",
    )

    torch.save(model.state_dict(), path_to_save)
    logger.info("Multivae is ready")

    return runner
