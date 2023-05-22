import pytorch_lightning as pl
from thunder_ml.routines import InferenceMode

from typing import Any


class ThunderModule(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.outputs = {
            InferenceMode.TRAINING: [],
            InferenceMode.VALIDATION: [],
            InferenceMode.TESTING: [],
            InferenceMode.PREDICTION: [],
        }

        self.inference_mode = InferenceMode.UNDEFINED
        self.current_batch = None

    def reset_outputs(self, mode):
        self.outputs[mode] = []

    def reset_volatile_data(self):
        self.inference_mode = InferenceMode.UNDEFINED
        self.current_batch = None

    def has_tensorboard(self):
        return isinstance(self.logger, pl.loggers.TensorBoardLogger)

    def _log_any(self, log_method_name, *args, **kwargs):
        if not self.has_tensorboard():
            raise RuntimeError(
                f"Can't use '{log_method_name}'. No TensorBoardLogger attached."
            )

        log_method = getattr(self.logger.experiment, log_method_name, None)
        if log_method is None:
            raise RuntimeError(
                f"TensorBoardLogger has no method named '{log_method_name}'."
            )

        if "global_step" not in kwargs or kwargs["global_step"] is None:
            kwargs["global_step"] = self.global_step

        return log_method(*args, **kwargs)

    def add_image(
        self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW"
    ):
        return self._log_any(
            "add_image",
            tag=tag,
            img_tensor=img_tensor,
            global_step=global_step,
            walltime=walltime,
            dataformats=dataformats,
        )

    def add_figure(self, tag, figure, global_step=None, walltime=None):
        return self._log_any(
            "add_figure",
            tag=tag,
            figure=figure,
            global_step=global_step,
            walltime=walltime,
        )

    def store_output(self, output):
        if self.inference_mode == InferenceMode.UNDEFINED:
            raise RuntimeError("Cannot call this outside of `..._step()`!")
        self.outputs[self.inference_mode].append((self.current_batch, output))

    def mode_step(self, batch, batch_idx, mode: InferenceMode):
        raise NotImplementedError

    def _wrapped_mode_step(self, batch, batch_idx, mode: InferenceMode):
        self.inference_mode = mode
        self.current_batch = batch

        if mode in self.outputs and batch_idx == 0:
            self.reset_outputs(mode)

        try:
            results = self.mode_step(batch, batch_idx, mode)
        finally:
            self.reset_volatile_data()

        return results

    def training_step(self, batch, batch_idx):
        return self._wrapped_mode_step(batch, batch_idx, InferenceMode.TRAINING)

    def validation_step(self, batch, batch_idx):
        return self._wrapped_mode_step(batch, batch_idx, InferenceMode.VALIDATION)

    def test_step(self, batch, batch_idx):
        return self._wrapped_mode_step(batch, batch_idx, InferenceMode.TESTING)

    def predict_step(self, batch, batch_idx):
        return self._wrapped_mode_step(batch, batch_idx, InferenceMode.PREDICTION)

    def configure_optimizers(self):
        raise NotImplementedError
