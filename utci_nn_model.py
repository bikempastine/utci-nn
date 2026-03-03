import torch.nn as nn
import torch
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from typing import Union, Tuple

class UTCIEmulator(nn.Module):
    """
    Neural network emulator for Universal Thermal Climate Index (UTCI).
    
    Input: 4 meteorological variables
        - Air Temperature (°C)
        - Relative Humidity (%)
        - Wind Speed (m/s)
        - Mean Radiant Temperature difference (°C) [Air Temperature - MRT]
    Output: UTCI value
    """
    def __init__(self):
        super(UTCIEmulator, self).__init__()
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
    



# def NN_UTCI(Ta, Tr, va, rH):
#     """
#     Calculate the Universal Thermal Climate Index (UTCI) using a neural network model by Pastine et al.
    
#     The UTCI is a biocliometric index that assesses thermal comfort in outdoor environments
#     by considering air temperature, radiant temperature, wind speed, and humidity (via
#     water vapor pressure). This function uses a pre-trained neural network emulator to
#     predict UTCI values efficiently.
    
#     Parameters
#     ----------
#     Ta : float
#         Air temperature in degrees Celsius (°C).
#         Valid range: -50°C to +50°C
#     Tr : float
#         Mean radiant temperature in degrees Celsius (°C).
#         Valid range: Ta - 30°C to Ta + 70°C
#     va : float
#         Wind speed at 10m height in meters per second (m/s).
#         Valid range: 0 m/s to 30.3 m/s
#     rH : float
#         Relative Humidity in percentage (%).
#         valid range: ~5-100% RH depending on temperature
        
#     Returns
#     -------
#     float
#         The calculated UTCI value in degrees Celsius (°C).
#         The UTCI represents the perceived temperature that accounts for
#         all meteorological factors affecting human thermal comfort.
        
#     Warnings
#     --------
#     UserWarning
#         Issued when any input parameter is outside the valid training bounds.
#         The function will still compute a prediction, but results may be unreliable.
        
#     Notes
#     -----
#     - The model predicts the difference (UTCI - Ta) and then adds Ta back
#       to obtain the final UTCI value
#     - The function requires pre-trained model weights at '../results/utci_nn_weights.pth'
#     - A fitted scaler object must exist at '../results/scaler.pkl'
#     - The model operates in evaluation mode (no gradient computation)
#     - Input features are standardized using the pre-fitted scaler before prediction
#     - **Extrapolation warning**: When inputs are outside training bounds, the neural
#       network may produce unreliable predictions
    
#     The neural network was trained to emulate the official UTCI calculation
#     algorithm, providing faster computation while maintaining accuracy within
#     the specified parameter ranges.
    
#     Examples
#     --------
#     >>> # Comfortable conditions
#     >>> utci = utci_nn_model(Ta=20.0, Tr=25.0, va=2.0, pa=15.0)
#     >>> print(f"UTCI: {utci:.2f}°C")
    
#     >>> # Hot conditions with high radiant temperature
#     >>> utci = utci_nn_model(Ta=30.0, Tr=45.0, va=1.0, pa=25.0)
#     >>> print(f"UTCI: {utci:.2f}°C")
    
#     >>> # Cold windy conditions
#     >>> utci = utci_nn_model(Ta=5.0, Tr=3.0, va=8.0, pa=8.0)
#     >>> print(f"UTCI: {utci:.2f}°C")
    
#     >>> # Out-of-bounds warning example
#     >>> utci = utci_nn_model(Ta=55.0, Tr=60.0, va=2.0, pa=15.0)
#     UserWarning: Input Ta=55.0°C is outside valid range [-50.0, 50.0]°C...
    
#     References
#     ----------
#     .. [1] Bröde, P., et al. (2012). "Deriving the operational procedure for the 
#            Universal Thermal Climate Index (UTCI)." International Journal of 
#            Biometeorology, 56(3), 481-494.
#      """

#     # Calculate Mean Radient Temperature difference
#     Tr_Ta_diff = Tr - Ta
    

    
#     # Load the model architecture and weights
#     model = UTCIEmulator()
#     model.load_state_dict(torch.load('../results/utci_nn_weights.pth'))
#     model.eval()
    
#     # Prepare the inputs
#     inputs = pd.DataFrame({
#         "Ta": [Ta],
#         "Tr-Ta": [Tr_Ta_diff],
#         "va": [va],
#         "rH": [rH]
#     })

#     # Scale the inputs
#     scaler = joblib.load('../results/scaler.pkl')
#     inputs_scaled = scaler.transform(inputs)
#     inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)

#     # Make prediction
#     #NEEDS TO BE VECTORIZED
#     if Ta < -50.0 or Ta > 50.0 or rH < 5.0 or rH > 100.0 or va < 0.0 or va > 30.3 or Tr_Ta_diff < -30.0 or Tr_Ta_diff > 70.0:
#        utci_prediction = np.nan # Assign NaN for out-of-bounds inputs

#     else:
#         with torch.no_grad():
#             prediction = model(inputs_tensor)  
#         utci_prediction = prediction.item() + Ta  # Add back the air temperature to get UTCI

#     return utci_prediction




class UTCIPredictor:
    """
    Vectorized UTCI predictor using the pre-trained neural network emulator.
    Loads model + scaler once. Call `.predict(...)` with numpy arrays of matching shape.
    """
    def __init__(self,
                 model_path: str = "../results/utci_nn_weights.pth",
                 scaler_path: str = "../results/scaler.pkl",
                 device: str = "cpu",
                 batch_size: int = 65536):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)

        # load model architecture and weights
        self.model = UTCIEmulator()  
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # load scaler (assumed to have .transform NumPy array -> scaled array)
        self.scaler = joblib.load(scaler_path)

    def _validate_and_flatten(self,
                              Ta: Union[np.ndarray, float],
                              Tr: Union[np.ndarray, float],
                              va: Union[np.ndarray, float],
                              rH: Union[np.ndarray, float]) -> Tuple[np.ndarray, Tuple[int,...]]:
        """Convert to np.ndarray and flatten while preserving original shape."""
        Ta = np.asarray(Ta)
        Tr = np.asarray(Tr)
        va = np.asarray(va)
        rH = np.asarray(rH)

        if not (Ta.shape == Tr.shape == va.shape == rH.shape):
            raise ValueError("All inputs must have identical shapes.")

        orig_shape = Ta.shape
        size = Ta.size

        Ta_f = Ta.reshape(size)
        Tr_f = Tr.reshape(size)
        va_f = va.reshape(size)
        rH_f = rH.reshape(size)

        return (Ta_f, Tr_f, va_f, rH_f), orig_shape

    # def predict(self,
    #             Ta: Union[np.ndarray, float],
    #             Tr: Union[np.ndarray, float],
    #             va: Union[np.ndarray, float],
    #             rH: Union[np.ndarray, float]) -> np.ndarray:
    #     """
    #     Vectorized prediction of UTCI for an entire grid.
    #     Returns an array of same shape as inputs with UTCI (°C) or np.nan for OOB.
    #     """
    #     (Ta_f, Tr_f, va_f, rH_f), orig_shape = self._validate_and_flatten(Ta, Tr, va, rH)

    #     # compute Tr-Ta difference
    #     TrTa_f = Tr_f - Ta_f

    #     # bounds check (vectorized)
    #     valid_mask = (
    #         (Ta_f >= -50.0) & (Ta_f <= 50.0) &
    #         (rH_f >= 5.0) & (rH_f <= 100.0) &
    #         (va_f >= 0.0) & (va_f <= 30.3) &
    #         (TrTa_f >= -30.0) & (TrTa_f <= 70.0)
    #     )

    #     N = Ta_f.size
    #     utci_out = np.full(N, np.nan, dtype=float)

    #     if valid_mask.sum() == 0:
    #         # nothing valid; return array of nans
    #         return utci_out.reshape(orig_shape)

    #     # prepare input matrix for valid entries only (columns: Ta, Tr-Ta, va, rH)
    #     inputs = np.column_stack((Ta_f[valid_mask],
    #                               TrTa_f[valid_mask],
    #                               va_f[valid_mask],
    #                               rH_f[valid_mask]))
        
        

    #     # scale (scaler.transform expects shape (n_samples, n_features))
    #     inputs_df = pd.DataFrame(inputs, columns=["Ta", "Tr-Ta", "va", "rH"])
    #     inputs_scaled = self.scaler.transform(inputs_df)

    #     # inference in batches to be memory safe
    #     n_valid = inputs_scaled.shape[0]
    #     preds = np.empty(n_valid, dtype=float)

    #     with torch.no_grad():
    #         start = 0
    #         while start < n_valid:
    #             end = min(start + self.batch_size, n_valid)
    #             batch_np = inputs_scaled[start:end].astype(np.float32)
    #             batch_tensor = torch.from_numpy(batch_np).to(self.device)
    #             out = self.model(batch_tensor)  # expects shape (batch, ) or (batch,1)
    #             out = out.detach().cpu().numpy().reshape(-1)
    #             preds[start:end] = out
    #             start = end

    #     # preds are (UTCI - Ta) according to your notes -> add Ta back
    #     Ta_valid = inputs[:, 0]  # Ta used in inputs
    #     utci_valid = preds + Ta_valid

    #     # fill results back into full output array
    #     utci_out[valid_mask] = utci_valid

    #     # reshape back to original grid shape
    #     return utci_out.reshape(orig_shape, 1)
    
    def predict(self,
                Ta: Union[np.ndarray, float],
                Tr: Union[np.ndarray, float],
                va: Union[np.ndarray, float],
                rH: Union[np.ndarray, float],
                clip_oob: bool = False) -> np.ndarray:
        (Ta_f, Tr_f, va_f, rH_f), orig_shape = self._validate_and_flatten(Ta, Tr, va, rH)

        # compute Tr-Ta difference
        TrTa_f = Tr_f - Ta_f

        # Optionally clip out-of-bounds values so we always produce finite outputs
        if clip_oob:
            Ta_f = np.clip(Ta_f, -50.0, 50.0)
            rH_f = np.clip(rH_f, 5.0, 100.0)
            va_f = np.clip(va_f, 0.0, 30.3)
            TrTa_f = np.clip(TrTa_f, -30.0, 70.0)

        # bounds check (vectorized)
        valid_mask = (
            (Ta_f >= -50.0) & (Ta_f <= 50.0) &
            (rH_f >= 5.0)   & (rH_f <= 100.0) &
            (va_f >= 0.0)   & (va_f <= 30.3) &
            (TrTa_f >= -30.0)& (TrTa_f <= 70.0)
        )

        N = Ta_f.size
        utci_out = np.full(N, np.nan, dtype=float)

        # If clip_oob==True then everything should be valid_mask==True; but keep logic robust
        if valid_mask.sum() == 0:
            return utci_out.reshape(orig_shape, 1)

        # prepare input matrix for valid entries only (columns: Ta, Tr-Ta, va, rH)
        inputs = np.column_stack((Ta_f[valid_mask],
                                TrTa_f[valid_mask],
                                va_f[valid_mask],
                                rH_f[valid_mask]))

        # scale and inference (same as before)...
        inputs_df = pd.DataFrame(inputs, columns=["Ta", "Tr-Ta", "va", "rH"])
        inputs_scaled = self.scaler.transform(inputs_df)
        
        # inference in batches to be memory safe
        n_valid = inputs_scaled.shape[0]
        preds = np.empty(n_valid, dtype=float)

        with torch.no_grad():
            start = 0
            while start < n_valid:
                end = min(start + self.batch_size, n_valid)
                batch_np = inputs_scaled[start:end].astype(np.float32)
                batch_tensor = torch.from_numpy(batch_np).to(self.device)
                out = self.model(batch_tensor)  # expects shape (batch, ) or (batch,1)
                out = out.detach().cpu().numpy().reshape(-1)
                preds[start:end] = out
                start = end

        # preds are (UTCI - Ta) according to your notes -> add Ta back
        Ta_valid = inputs[:, 0]  # Ta used in inputs
        #utci_valid = preds + Ta_valid

        # fill results back into full output array
        utci_out[valid_mask] = preds

        # reshape back to original grid shape
        return utci_out.reshape((*orig_shape, 1))
