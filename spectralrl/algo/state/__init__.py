from .base import BaseStateAlgorithm
from .ctrlsr.agent import CtrlSR_TD3
from .diffsr.agent import DiffSR_TD3
from .lvrep.agent import LVRep_TD3
from .sac.agent import SAC
from .speder.agent import Speder_TD3
from .td3.agent import TD3

__all__ = (
    "BaseStateAlgorithm",
    "SAC",
    "CtrlSR_TD3",
    "LVRep_TD3",
    "TD3",
    "DiffSR_TD3",
    "Speder_TD3"
)
