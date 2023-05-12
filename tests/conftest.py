import pytorch_lightning as pl
import pytest

from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.nn.functional as F

import pytorch_thunder


@pytest.fixture
def linear_model():
    class LitModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(2, 4)
            self.l2 = nn.Linear(4, 4)
            self.l3 = nn.Linear(4, 1)

        def forward(self, x):
            h1 = torch.relu(self.l1(x.view(x.size(0), -1)))
            h2 = torch.relu(self.l2(h1))
            h3 = torch.relu(self.l3(h2))

            return h3

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.mse_loss(y, y_hat)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.mse_loss(y, y_hat)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)

    return LitModel()


@pytest.fixture
def linear_thunder_silent_model():
    class LitModel(pytorch_thunder.modules.ThunderModule):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(2, 4)
            self.l2 = nn.Linear(4, 4)
            self.l3 = nn.Linear(4, 1)

        def forward(self, x):
            h1 = torch.relu(self.l1(x.view(x.size(0), -1)))
            h2 = torch.relu(self.l2(h1))
            h3 = torch.relu(self.l3(h2))

            return h3

        def mode_step(self, batch, batch_idx, mode):
            x, y = batch
            y_hat = self(x)
            loss = F.mse_loss(y, y_hat)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)

    return LitModel()


@pytest.fixture
def linear_thunder_storing_model():
    class LitModel(pytorch_thunder.modules.ThunderModule):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(2, 4)
            self.l2 = nn.Linear(4, 4)
            self.l3 = nn.Linear(4, 1)

        def forward(self, x):
            h1 = torch.relu(self.l1(x.view(x.size(0), -1)))
            h2 = torch.relu(self.l2(h1))
            h3 = torch.relu(self.l3(h2))

            return h3

        def mode_step(self, batch, batch_idx, mode):
            x, y = batch
            y_hat = self(x)
            loss = F.mse_loss(y_hat, y)

            if mode == pytorch_thunder.routines.inference_mode.InferenceMode.VALIDATION:
                self.store_output(y_hat)

            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)

    return LitModel()


@pytest.fixture
def xor_dataloader():
    inputs = torch.randint(0, 2, (4096, 2)).type(torch.FloatTensor)
    outputs = (inputs[:, 0] != inputs[:, 1]).type(torch.FloatTensor)

    dataset = torch.utils.data.TensorDataset(inputs, outputs)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    return dataloader
