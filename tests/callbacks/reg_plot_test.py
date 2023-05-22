import pytest
import matplotlib

from thunder_ml.callbacks.reg_plot import RegressionPlotCallback
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl

CMAPS = ["rainbow", "Reds", matplotlib.colormaps.get_cmap("plasma")]


@pytest.mark.parametrize("cmap", CMAPS)
def test_regression_callback(linear_thunder_storing_model, xor_dataloader, cmap):
    rpc = RegressionPlotCallback(cmap=cmap)
    tb = TensorBoardLogger(".")
    trainer = pl.Trainer(
        callbacks=[rpc],
        logger=tb,
        max_epochs=16,
    )

    trainer.fit(linear_thunder_storing_model, xor_dataloader, xor_dataloader)


@pytest.mark.parametrize("cmap", CMAPS)
def test_call_with_wrong_model(linear_model, xor_dataloader, cmap):
    rpc = RegressionPlotCallback(cmap=cmap)
    tb = TensorBoardLogger(".")
    trainer = pl.Trainer(
        callbacks=[rpc],
        logger=tb,
        max_epochs=16,
    )

    with pytest.raises(ValueError):
        trainer.fit(linear_model, xor_dataloader, xor_dataloader)
