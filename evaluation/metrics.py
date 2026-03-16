import numpy as np

def mape(y_true, y_pred):

    return np.mean(
        abs((y_true - y_pred) / y_true)
    ) * 100