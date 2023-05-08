from enum import StrEnum


class InferenceMode(StrEnum):
    """
    Indicator of the stage a (Lightning)Module is used.
    """

    TRAINING = "train"
    VALIDATION = "val"
    TESTING = "test"
    PREDICTION = "pred"

    UNDEFINED = "undef"
