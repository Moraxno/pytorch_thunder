import os
from typing import Iterable

import matplotlib
import numpy as np
import pytest
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from thunder_ml.callbacks import ConfusionMatrixCallback
from thunder_ml.callbacks.conf_matrix import conf_matrix2figure


def artifact_path(filename: str):
    return os.path.join("./artifacts", filename)


def test_constructor_without_args():
    _ = ConfusionMatrixCallback()


@pytest.mark.parametrize("num_classes", [0, 1, 5, 42, -1, -9999])
def test_constructor_with_class_num(num_classes):
    cmc = ConfusionMatrixCallback(num_classes)

    if num_classes >= 0:
        assert len(cmc.class_names) == num_classes
        assert len(cmc.class_indices) == num_classes
    else:
        assert len(cmc.class_names) == 0
        assert len(cmc.class_indices) == 0


@pytest.mark.parametrize(
    "class_names",
    [
        ["good", "bad"],
        ["one", "two", "three", "four", "five", "six"],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ],
)
def test_constructor_with_class_names(class_names):
    cmc = ConfusionMatrixCallback(class_names)

    assert cmc.class_names == class_names
    assert len(cmc.class_indices) == len(class_names)


def test_call_with_wrong_model(linear_model, xor_dataloader):
    cmc = ConfusionMatrixCallback()
    trainer = pl.Trainer(callbacks=[cmc], max_epochs=1, log_every_n_steps=5)

    with pytest.raises(ValueError):
        trainer.fit(linear_model, xor_dataloader, xor_dataloader)


def test_call_with_lightning_model_raises(linear_model, xor_dataloader):
    cmc = ConfusionMatrixCallback()
    trainer = pl.Trainer(callbacks=[cmc], max_epochs=1, log_every_n_steps=5)

    with pytest.raises(ValueError):
        trainer.fit(linear_model, xor_dataloader, xor_dataloader)


def test_call_with_silent_thunder_model_raises(
    linear_thunder_silent_model, xor_dataloader
):
    cmc = ConfusionMatrixCallback()
    trainer = pl.Trainer(callbacks=[cmc], max_epochs=1)

    with pytest.raises(RuntimeError):
        trainer.fit(linear_thunder_silent_model, xor_dataloader, xor_dataloader)


@pytest.mark.parametrize("num_classes", [None, -1, 5, [], ["cat", "dog"]])
@pytest.mark.parametrize("cmap", ["Reds", matplotlib.colormaps["plasma"]])
def test_call_with_storing_thunder_module(
    linear_thunder_storing_model, xor_dataloader, num_classes, cmap
):
    cmc = ConfusionMatrixCallback(classes=num_classes, cmap=cmap)
    tb = TensorBoardLogger(".")
    trainer = pl.Trainer(
        callbacks=[cmc],
        logger=tb,
        max_epochs=1,
        log_every_n_steps=5,
    )

    trainer.fit(linear_thunder_storing_model, xor_dataloader, xor_dataloader)


@pytest.mark.parametrize("num_classes", [None, -1, 5, [], ["cat", "dog"]])
def test_construct_with_bad_cmap_raises(num_classes):
    with pytest.raises(TypeError):
        _ = ConfusionMatrixCallback(classes=num_classes, cmap=None)


@pytest.mark.visual
@pytest.mark.parametrize(
    "filename,matrix,classes,cmap",
    [
        (
            "main_diag.png",
            np.eye(8),
            ["one", "two", "three", "four", "five", "six", "seven", "eight"],
            matplotlib.colormaps.get_cmap("Reds"),
        ),
        (
            "random.png",
            np.random.random((8, 8)),
            "ABCDEFGH",
            matplotlib.colormaps.get_cmap("Greys"),
        ),
    ],
)
def test_render_a_matrix(
    filename: str,
    matrix: np.ndarray,
    classes: Iterable,
    cmap: matplotlib.cm.colors.Colormap,
):
    fig = conf_matrix2figure(matrix, classes, cmap)
    fig.savefig(artifact_path(filename))
