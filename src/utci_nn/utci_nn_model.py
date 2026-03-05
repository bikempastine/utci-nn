import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from typing import Union

# ── Neural network model definition ───────────────────────────────────────────
class UTCI_NN_Emulator(nn.Module):
    """
    Neural network emulator for Universal Thermal Climate Index (UTCI).
    """
    def __init__(self):
        super(UTCI_NN_Emulator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 79),
            nn.ReLU(),
            nn.Linear(79, 75),
            nn.ReLU(),
            nn.Linear(75, 39),
            nn.ReLU(),
            nn.Linear(39, 1)
        )

    def forward(self, x):
        return self.model(x)


# ── Load assets at import time ───────────────────────────────────────────────────
_ASSETS_DIR = Path(__file__).parent / "assets"
_model = UTCI_NN_Emulator()
_model.load_state_dict(torch.load(_ASSETS_DIR / "utci_nn_weights.pth", weights_only=True))
_model.eval()

_scaler = joblib.load(_ASSETS_DIR / "scaler.pkl")


# ── Valid input bounds, from training data ────────────────────────────────────
_BOUNDS = {
    "Ta":    (-50.0,  50.0),
    "Tr":   (-80.0, 120.0),
    "va":    (  0.5,  30.3),
    "rH":    (  5.0, 100.0),
}

# ── Main prediction function ─────────────────────────────────────────────────
def calculate_utci(
    Ta: Union[float, np.ndarray, pd.Series],
    Tr: Union[float, np.ndarray, pd.Series],
    va: Union[float, np.ndarray, pd.Series],
    rH: Union[float, np.ndarray, pd.Series],
    oob: str = "nan",
) -> np.ndarray:
    """
    Calculate UTCI using a pre-trained neural network approximator as described in Pastine et al.
    NN is trained on the same data as the polynomial approximation in Bröde et al. (2012) and covers the same input domain.

    Accepts scalars, numpy arrays, or pandas Series for all meterological inputs.

    Parameters
    ----------
    Ta  : Air temperature (°C),          valid range: -50 to +50
    Tr  : Mean radiant temperature (°C), valid range: Ta-80 to Ta+120
    va  : Wind speed at 10 m (m/s),      valid range: 0.5 to 30.3
    rH  : Relative humidity (%),         valid range: 5 to 100
    oob : Out-of-bounds handling strategy:
            "nan"   – return NaN for any out-of-bounds row (default)
            "clamp" – clamp each variable to its valid range before predicting

    Returns:
    -------
    np.ndarray
        UTCI values in °C defind as UTCI offset plus Air temperature. Shape matches the input arrays.

    References
    ----------
    Bröde, P., Fiala, D., Bla˙zejczyk, K. et al. Deriving the operational procedure
    for the Universal Thermal Climate Index (UTCI). Int. J. Biometeorol. 56, 481–494
    (2012).
    """
    if oob not in ("nan", "clamp"):
        raise ValueError(f"oob must be 'nan' or 'clamp', got '{oob}'")

    # ── Coerce all inputs to float32 numpy arrays ──────────────────────────────
    Ta    = np.asarray(Ta,    dtype=np.float32)
    Tr    = np.asarray(Tr,    dtype=np.float32)
    va    = np.asarray(va,    dtype=np.float32)
    rH    = np.asarray(rH,    dtype=np.float32)
    
    # ── Keep track if inputs were scalars to return scalar output ──────────────
    scalar_input = Ta.ndim == 0
    Ta, Tr, va, rH = (np.atleast_1d(a) for a in (Ta, Tr, va, rH))

    # ── Check that all inputs have the same length and issue warning ────────────
    shapes = {len(np.atleast_1d(a)) for a in (Ta, Tr, va, rH)}
    if len(shapes) > 1:
        raise ValueError(f"All inputs must have the same length, got shapes: {shapes}")



    # ── Run inference ───────────────────────────────────────────────────────────
    utci = np.full(len(Ta), np.nan)

    if oob == "nan":
        oob_mask = (
            (Ta < _BOUNDS["Ta"][0]) | (Ta > _BOUNDS["Ta"][1])
            | (Tr < _BOUNDS["Tr"][0]) | (Tr > _BOUNDS["Tr"][1])
            | (va < _BOUNDS["va"][0]) | (va > _BOUNDS["va"][1])
            | (rH < _BOUNDS["rH"][0]) | (rH > _BOUNDS["rH"][1])
        )
        valid_mask = ~oob_mask
        
        if not valid_mask.any():
            return utci[0] if scalar_input else utci  # early return, all NaN
        
        Tr_Ta = Tr - Ta
        X = pd.DataFrame({"Ta": Ta[valid_mask], "Tr-Ta": Tr_Ta[valid_mask], "va": va[valid_mask], "rH": rH[valid_mask]})
        X_scaled = _scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            predictions = _model(X_tensor).squeeze().numpy()
        utci[valid_mask] = predictions + Ta[valid_mask]  # only write valid rows

    elif oob == "clamp":
        Ta = np.clip(Ta, *_BOUNDS["Ta"])
        Tr = np.clip(Tr, *_BOUNDS["Tr"])
        va = np.clip(va, *_BOUNDS["va"])
        rH = np.clip(rH, *_BOUNDS["rH"])

        Tr_Ta = Tr - Ta
        X = pd.DataFrame({"Ta": Ta, "Tr-Ta": Tr_Ta, "va": va, "rH": rH})
        X_scaled = _scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            predictions = _model(X_tensor).squeeze().numpy()
        utci[:] = predictions + Ta  # all rows are valid after clamping

    return utci[0] if scalar_input else utci