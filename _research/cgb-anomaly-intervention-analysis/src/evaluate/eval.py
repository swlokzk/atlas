from typing import Tuple
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2
