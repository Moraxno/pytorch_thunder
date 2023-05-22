import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from typing import Iterable
import numpy as np
import torch
import matplotlib
import itertools
import colorsys

from thunder_ml.routines import InferenceMode
from thunder_ml.modules import ThunderModule


def convert_distribution_to_predictions(batch: torch.Tensor):
    if len(batch.shape) > 2:
        raise ValueError("Distributions have at most 2 dimensions")

    if len(batch.shape) == 1 and torch.is_floating_point(batch):
        batch = batch.reshape(1, -1)

    predictions = torch.argmax(batch, dim=-1)

    return predictions


class ConfusionMatrixCallback(pl.Callback):
    def __init__(self, classes=None, cmap=plt.cm.Blues):
        # check if we have proper labels or just numbers for classes
        if isinstance(classes, int):
            self.class_indices = np.arange(0, classes)
            self.class_names = self.class_indices
        elif isinstance(classes, Iterable):
            self.class_indices = np.arange(0, len(list(classes)))
            self.class_names = classes
        else:
            self.class_indices = None
            self.class_names = None

        if isinstance(cmap, str):
            cmap = matplotlib.colormaps[cmap]
        elif not isinstance(cmap, matplotlib.cm.colors.Colormap):
            raise TypeError("`cmap` must be of type `str` or `Colormap`")

        self.cmap = cmap

    # do after each validation
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "ThunderModule"
    ):
        if not isinstance(pl_module, ThunderModule):
            raise ValueError("Module needs to be a ThunderModule!")

        # thanks to our ThunderModule, we have a history of inferences
        outputs = pl_module.outputs.get(InferenceMode.VALIDATION, [])
        ytrue = None
        ypred = None

        if len(outputs) == 0:
            raise RuntimeError("No inferences logged.")

        # make a long list of predictions and ground truths
        for (inputs, y_targets), y_distributions in outputs:
            y_predictions = convert_distribution_to_predictions(y_distributions)

            if ypred is None:
                ypred = y_predictions
            else:
                ypred = torch.cat([ypred, y_predictions], 0)

            if ytrue is None:
                ytrue = y_targets
            else:
                ytrue = torch.cat([ytrue, y_targets], 0)

        if self.class_indices is None or len(self.class_indices) == 0:
            class_indices = list(set(np.unique(ytrue)).union(set(np.unique(ypred))))
        else:
            class_indices = self.class_indices

        if self.class_names is None or len(self.class_names) == 0:
            class_names = class_indices
        else:
            class_names = self.class_names

        # let sklearn make the matrix and than we normalize it between zero and one
        cf_matrix = confusion_matrix(
            ytrue, ypred, labels=class_indices, normalize="true"
        ).astype(float)

        # create a figure in matplot
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cf_matrix, interpolation="nearest", cmap=self.cmap, vmin=0, vmax=1)
        plt.title("Confusion matrix")

        # add a scale
        plt.colorbar()

        # put proper labels to the matrix
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Use white text if squares are dark; otherwise black.
        threshold = cf_matrix.max() / 2.0

        # add percentages in each cell
        for i, j in itertools.product(
            range(cf_matrix.shape[0]), range(cf_matrix.shape[1])
        ):
            red, green, blue, _ = self.cmap(cf_matrix[i, j])
            _, lightness, _ = colorsys.rgb_to_hls(red, green, blue)

            color = "white" if lightness < threshold else "black"
            plt.text(
                j,
                i,
                f"{cf_matrix[i, j]:.2%}",
                horizontalalignment="center",
                color=color,
            )

        # finishing touches
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        # use thunder modules integration to tensorboard to add the figure directly
        pl_module.add_figure("conf_matrix", figure)

        # clean up
        plt.close(figure)
