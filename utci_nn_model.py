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


# ── Load once at import time ───────────────────────────────────────────────────
_model = UTCI_NN_Emulator()
_model.load_state_dict(torch.load('utci_nn_weights.pth', weights_only=True))
_model.eval()

_scaler = joblib.load('scaler.pkl')

# ── Valid input bounds, from training data ────────────────────────────────────
_BOUNDS = {
    "Ta":    (-50.0,  50.0),
    "Tr":   (-80.0, 120.0),
    "va":    (  0.5,  30.3),
    "rH":    (  5.0, 100.0),
}

# ── Main prediction function ─────────────────────────────────────────────────

def NN_UTCI(
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

    Returns
    -------
    np.ndarray
        UTCI values in °C. Shape matches the input arrays.

    References
    ----------
    Bröde et al. (2012). Int. J. Biometeorology, 56(3), 481-494.
    """
    if oob not in ("nan", "clamp"):
        raise ValueError(f"oob must be 'nan' or 'clamp', got '{oob}'")

    # ── Coerce all inputs to float32 numpy arrays ──────────────────────────────
    Ta    = np.asarray(Ta,    dtype=np.float32)
    Tr    = np.asarray(Tr,    dtype=np.float32)
    va    = np.asarray(va,    dtype=np.float32)
    rH    = np.asarray(rH,    dtype=np.float32)
    scalar_input = Ta.ndim == 0
    Ta, Tr, va, rH = (np.atleast_1d(a) for a in (Ta, Tr, va, rH))

    Tr_Ta = Tr - Ta  # MRT offset

    # ── Check that all inputs have the same length and issue warning ─────────────
    shapes = {len(np.atleast_1d(a)) for a in (Ta, Tr, va, rH)}
    if len(shapes) > 1:
        raise ValueError(f"All inputs must have the same length, got shapes: {shapes}")


    # ── Build feature matrix and scale ────────────────────────────────────────
    X = pd.DataFrame({"Ta": Ta, "Tr-Ta": Tr_Ta, "va": va, "rH": rH})
    X_scaled = _scaler.transform(X)

    # ── Run inference ──────────────────────────────────────────────────────────
    utci = np.full(len(Ta), np.nan)  # pre-fill with NaN

    if oob == "nan":
        # Filter out oob inputs before calculation
        oob_mask = (
            (Ta    < _BOUNDS["Ta"][0])    | (Ta    > _BOUNDS["Ta"][1])
            | (Tr_Ta < _BOUNDS["Tr_Ta"][0]) | (Tr_Ta > _BOUNDS["Tr_Ta"][1])
            | (va    < _BOUNDS["va"][0])    | (va    > _BOUNDS["va"][1])
            | (rH    < _BOUNDS["rH"][0])    | (rH    > _BOUNDS["rH"][1])
        )
        valid_mask = ~oob_mask
        if valid_mask.any():
            X_tensor = torch.tensor(X_scaled[valid_mask], dtype=torch.float32)
            with torch.no_grad():
                predictions = _model(X_tensor).squeeze().numpy()
            utci[valid_mask] = predictions + Ta[valid_mask]

    elif oob == "clamp":
        # Clamp to the bounds of UTCI before inferance
        Ta    = np.clip(Ta,    *_BOUNDS["Ta"])
        Tr_Ta = np.clip(Tr_Ta, *_BOUNDS["Tr_Ta"])
        va    = np.clip(va,    *_BOUNDS["va"])
        rH    = np.clip(rH,   *_BOUNDS["rH"])

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            predictions = _model(X_tensor).squeeze().numpy()
        utci = predictions + Ta

    return utci[0] if scalar_input else utci


