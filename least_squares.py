import numpy as np
import pandas as pd

def least_squares_fit(x_vals, y_vals, degree):
    coeffs = np.polyfit(x_vals, y_vals, degree)
    p = np.poly1d(coeffs)
    y_pred = p(x_vals)
    return coeffs, y_pred

def least_squares_table(x_vals, y_vals, y_pred):
    data = {
        "x": x_vals,
        "y (actual)": y_vals,
        "y (predicted)": y_pred
    }
    return pd.DataFrame(data)
