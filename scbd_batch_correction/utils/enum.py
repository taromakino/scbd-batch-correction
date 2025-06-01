from enum import Enum


class Dataset(Enum):
    FUNK22 = "funk22"
    CELLPAINTING2 = "cellpainting2"


class EncoderType(Enum):
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    DENSENET121 = "densenet121"


class ExperimentGroup(Enum):
    TREATMENT = "treatment"
    CONTROL = "control"