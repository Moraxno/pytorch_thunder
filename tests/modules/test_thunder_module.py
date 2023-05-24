import pytest
import torch

from thunder_ml.modules.thunder_module import InferenceMode, ThunderModule


def test_module_construction():
    _ = ThunderModule()


def test_module_saving_output_while_not_training_throws():
    t = ThunderModule()
    with pytest.raises(RuntimeError):
        t.store_output(torch.zeros((32,)))


def test_module_saving_output_while_training_throws():
    t = ThunderModule()
    t.inference_mode = InferenceMode.TRAINING
    t.current_batch = torch.ones((32,))
    t.store_output(torch.zeros((32,)))

    assert len(t.outputs[InferenceMode.TRAINING]) > 0


def test_module_clearing_after_epoch():
    t = ThunderModule()
    t.inference_mode = InferenceMode.TRAINING
    t.current_batch = torch.ones((32,))
    t.outputs[InferenceMode.TRAINING] = torch.zeros((32,))

    t.reset_outputs(InferenceMode.TRAINING)

    assert len(t.outputs[InferenceMode.TRAINING]) == 0
    assert t.inference_mode == InferenceMode.TRAINING

    t.reset_volatile_data()
    assert t.inference_mode == InferenceMode.UNDEFINED
    assert t.current_batch is None
