# core/alignment.py

import numpy as np
from typing import Tuple
from scipy.signal import butter, filtfilt, correlate
from scipy.ndimage import uniform_filter1d
from core.config_loader import IDUQConfig

class PhaseAligner:
    """
    Handles viscoelastic phase delay compensation between robot commands and visual tissue response.
    Generates the interaction-driven residual R_phys representing acoustic slip or decoupling.
    """

    def __init__(self, config: IDUQConfig):
        self.cfg = config.alignment

    def _butter_lowpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Applies a zero-phase lowpass filter to remove high-frequency noise."""
        nyq = 0.5 * self.cfg['fs']
        normal_cutoff = self.cfg['cutoff_freq'] / nyq
        b, a = butter(self.cfg['filter_order'], normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def align_and_evaluate(self, X_raw: np.ndarray, Y_raw: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Finds the optimal phase lag using cross-correlation and computes the aligned residual.
        
        Args:
            X_raw (np.ndarray): 1D array of robot kinematics (e.g., Z-velocity).
            Y_raw (np.ndarray): 1D array of visual response (e.g., Divergence).
            
        Returns:
            Tuple[np.ndarray, int]: 
                - anomaly_score: The computed interaction residual R_phys.
                - best_lag: The optimal phase delay found (in frames).
        """
        # 1. Terminal Denoising
        X_clean = self._butter_lowpass_filter(X_raw)
        Y_clean = self._butter_lowpass_filter(Y_raw)

        # 2. Z-Score Standardization (to equalize metric scales)
        X_norm = (X_clean - np.mean(X_clean)) / (np.std(X_clean) + 1e-6)
        Y_norm = (Y_clean - np.mean(Y_clean)) / (np.std(Y_clean) + 1e-6)

        # 3. Cross-Correlation to find Viscoelastic Physical Lag
        correlation = correlate(Y_norm, X_norm, mode='full')
        lags = np.arange(-len(X_norm) + 1, len(Y_norm))
        
        # Constrain search window
        max_lag = self.cfg['max_lag']
        valid_idx = np.where((lags >= -max_lag) & (lags <= max_lag))[0]
        
        best_lag_idx = valid_idx[np.argmax(correlation[valid_idx])]
        best_lag = lags[best_lag_idx]

        # 4. Phase Compensation via Shift (np.roll)
        X_delayed = np.roll(X_norm, best_lag)
        if best_lag > 0:
            X_delayed[:best_lag] = X_delayed[best_lag]  # Pad edge to prevent wrap-around
        elif best_lag < 0:
            X_delayed[best_lag:] = X_delayed[best_lag-1]

        # 5. Residual Generation (Absolute tracking error post-alignment)
        residual = np.abs(Y_norm - X_delayed)
        
        # Smooth out single-frame jitter
        anomaly_score = uniform_filter1d(residual, size=self.cfg['residual_smooth_win'], mode='nearest')
        
        return anomaly_score, best_lag