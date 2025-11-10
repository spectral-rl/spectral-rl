from .base import BaseVisualAlgorithm
from .ctrlsr.agent import CtrlSR_TD3
from .diffsr.agent import DiffSR_TD3
from .drqv2.agent import DrQv2
from .mulvrep.agent import MuLVRep_DrQv2

__all__ = (
    "BaseVisualAlgorithm",
    "DrQv2",
    "CtrlSR_TD3",
    "DiffSR_TD3",
)
