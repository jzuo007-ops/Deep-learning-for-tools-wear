from typing import Dict

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    nonzero = np.abs(y_true) > 1e-12
    mape = float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100.0) if np.any(nonzero) else 0.0
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": float(r2),
        "MAPE_percent": mape,
    }

