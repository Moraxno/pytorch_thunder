import thunder_ml
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics as tm
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

inputs = torch.randint(0, 2, (4096, 2)).type(torch.FloatTensor)
outputs = (inputs[:, 0] != inputs[:, 1]).type(torch.LongTensor)


class LitStoringModel(thunder_ml.ThunderModule):
    def __init__(self, hidden_neurons=4, hidden_layers=1):
        super().__init__()
        self.model = nn.Sequential()

        self.model.append(nn.Linear(2, hidden_neurons))
        for _ in range(hidden_layers):
            self.model.append(nn.LazyLinear(hidden_neurons))
            self.model.append(nn.LazyBatchNorm1d())
            self.model.append(nn.LeakyReLU())
        self.model.append(nn.LazyLinear(2))

        self.save_hyperparameters()

    @property
    def example_input_array(self):
        return torch.zeros((64, 2))

    def forward(self, x):
        return self.model(x)

    def mode_step(self, batch, batch_idx, mode):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        if mode == thunder_ml.routines.inference_mode.InferenceMode.VALIDATION:
            self.store_output(y_hat)

        self.log("bce", loss)

        self.log(
            "acc",
            tm.functional.accuracy(y_hat, y, "multiclass", num_classes=2),
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


for _ in range(5):
    for i in range(2, 10, 2):
        for layer in range(5):
            dataset = torch.utils.data.TensorDataset(inputs, outputs)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

            cmc = thunder_ml.callbacks.ConfusionMatrixCallback()
            hpc = thunder_ml.callbacks.hyper_param.HyperparametersCallback()
            tb = TensorBoardLogger(".", default_hp_metric=False, log_graph=True)
            trainer = pl.Trainer(
                callbacks=[cmc, hpc], log_every_n_steps=5, logger=tb, max_epochs=8
            )

            trainer.fit(LitStoringModel(i, layer), dataloader, dataloader)
