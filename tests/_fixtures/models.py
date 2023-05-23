import pytest
import pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


@pytest.fixture
def linear_model():
    class LitModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(1, 1)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.mse_loss(y, y_hat)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)

    return LitModel()


@pytest.fixture
def sin_dataloader():
    inputs = torch.rand((4096,)) * 10 - 5
    outputs = torch.sin(inputs)

    dataset = torch.utils.data.TensorDataset(inputs, outputs)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    return dataloader
