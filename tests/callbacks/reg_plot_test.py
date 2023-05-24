import matplotlib
import pytest
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from thunder_ml.callbacks.reg_plot import RegressionPlotCallback

CMAPS = ["rainbow", "Reds", matplotlib.colormaps.get_cmap("plasma")]


def make_trainer(callbacks=None, logger=None):
    callbacks = [] if callbacks is None else callbacks
    return pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        log_every_n_steps=1,
    )


@pytest.mark.parametrize("cmap", CMAPS)
def test_regression_callback(linear_thunder_storing_model, xor_dataloader, cmap):
    rpc = RegressionPlotCallback(cmap=cmap)
    tb = TensorBoardLogger(".")
    trainer = make_trainer([rpc], tb)

    trainer.fit(linear_thunder_storing_model, xor_dataloader, xor_dataloader)


@pytest.mark.parametrize("cmap", CMAPS)
def test_call_with_wrong_model(linear_model, xor_dataloader, cmap):
    rpc = RegressionPlotCallback(cmap=cmap)
    tb = TensorBoardLogger(".")
    trainer = make_trainer([rpc], tb)

    with pytest.raises(ValueError):
        trainer.fit(linear_model, xor_dataloader, xor_dataloader)
