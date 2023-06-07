from enum import IntEnum


class InferenceMode(IntEnum):
    """
    Indicator of the stage a (Lightning)Module is used.
    """

    TRAINING = 0
    VALIDATION = 1
    TESTING = 2
    PREDICTION = 3

    UNDEFINED = -1


InferenceModeNames = {
    InferenceMode.TRAINING: "train",
    InferenceMode.VALIDATION: "val",
    InferenceMode.TESTING: "test",
    InferenceMode.PREDICTION: "pred",
    InferenceMode.UNDEFINED: "undef",
}
