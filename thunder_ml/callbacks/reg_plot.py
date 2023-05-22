import pytorch_lightning as pl
import matplotlib.pyplot as plt

import torch
import matplotlib

from thunder_ml.routines import InferenceMode
from thunder_ml.modules import ThunderModule


class RegressionPlotCallback(pl.Callback):
    def __init__(self, cmap="rainbow"):
        # check if we have proper labels or just numbers for classes

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
        for (inputs, y_targets), y_predictions in outputs:
            if ypred is None:
                ypred = y_predictions
            else:
                ypred = torch.cat([ypred, y_predictions], 0)

            if ytrue is None:
                ytrue = y_targets
            else:
                ytrue = torch.cat([ytrue, y_targets], 0)

        differences = torch.abs(ypred - ytrue)

        # create a figure in matplot
        figure = plt.figure(figsize=(8, 8))
        plt.scatter(ytrue, ypred, c=differences, cmap=self.cmap)
        plt.plot([ytrue.min(), ytrue.max()], [ytrue.min(), ytrue.max()], "k--")
        plt.title("Regression Plot")

        # add a scale
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("Absolute Error of Regression", rotation=270)

        # finishing touches
        plt.tight_layout()
        plt.xlabel("Ground Truth")
        plt.ylabel("Regression")

        # use thunder modules integration to tensorboard to add the figure directly
        pl_module.add_figure("Regression Plot", figure)

        # clean up
        plt.close(figure)
