# warp_templates/types.py
from typing import TypedDict, Dict, List, Union
import numpy as np

class FitLCResult(TypedDict, total=False):
    parameters: List[float]
    param_names: List[str]
    chisq: float
    ndof: int
    errors: Dict[str, float]
    success: bool

class CorrDataPerBand(TypedDict):
    wave: float
    phase: List[float]
    frac: List[float]
    err: List[float]
    tphase: np.ndarray
    tcorr: np.ndarray
    terr: np.ndarray

class TemplateCorrectionResult(TypedDict):
    model: str
    corrdata: Dict[str, CorrDataPerBand]
    corrmodel: Dict[str, Union[np.ndarray, List[float]]]
    dps_init: int
    dps_tcut: int
    dps_fcut: int
    dps_allcut: int
    success: bool
    absmag: float
    chidof: float
    lceval: Dict[str, int]
    