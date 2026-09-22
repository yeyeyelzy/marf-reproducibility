"""MARC: Multi-Scale Adaptive Relation Calibration for longitudinal EHR prediction."""

from .config import MARCConfig
from .models.marc import MARC

__all__ = ["MARC", "MARCConfig"]
