import pytest
from thunderstorm.callbacks import ConfusionMatrixCallback
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib


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
    trainer = pl.Trainer(callbacks=[cmc], max_epochs=1)

    with pytest.raises(ValueError):
        trainer.fit(linear_model, xor_dataloader, xor_dataloader)


def test_call_with_lightning_model_raises(linear_model, xor_dataloader):
    cmc = ConfusionMatrixCallback()
    trainer = pl.Trainer(callbacks=[cmc], max_epochs=1)

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
@pytest.mark.parametrize("cmap", ["Reds", matplotlib.cm.get_cmap("plasma")])
def test_call_with_storing_thunder_module(
    linear_thunder_storing_model, xor_dataloader, num_classes, cmap
):
    cmc = ConfusionMatrixCallback(classes=num_classes, cmap=cmap)
    tb = TensorBoardLogger(".")
    trainer = pl.Trainer(
        callbacks=[cmc],
        logger=tb,
        max_epochs=1,
    )

    trainer.fit(linear_thunder_storing_model, xor_dataloader, xor_dataloader)


@pytest.mark.parametrize("num_classes", [None, -1, 5, [], ["cat", "dog"]])
def test_construct_with_bad_cmap_raises(num_classes):
    with pytest.raises(TypeError):
        cmc = ConfusionMatrixCallback(classes=num_classes, cmap=None)
