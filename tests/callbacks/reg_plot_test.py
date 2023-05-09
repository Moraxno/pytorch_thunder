import pytest
import matplotlib

from thunderstorm.callbacks.reg_plot import RegressionPlotCallback
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl


@pytest.mark.parametrize(
    "cmap", ["rainbow", "Reds", matplotlib.colormaps.get_cmap("plasma")]
)
def test_regression_callback(linear_thunder_storing_model, xor_dataloader, cmap):
    rpc = RegressionPlotCallback(cmap=cmap)
    tb = TensorBoardLogger(".")
    trainer = pl.Trainer(
        callbacks=[rpc],
        logger=tb,
        max_epochs=16,
    )

    trainer.fit(linear_thunder_storing_model, xor_dataloader, xor_dataloader)
