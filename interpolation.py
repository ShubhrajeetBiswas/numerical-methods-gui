import pandas as pd

def lagrange_interpolation(x_vals, y_vals, x):
    n = len(x_vals)
    result = 0
    for i in range(n):
        term = y_vals[i]
        for j in range(n):
            if j != i:
                term *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        result += term
    return result

def lagrange_table(x_vals, y_vals):
    return pd.DataFrame({"x": x_vals, "y": y_vals})

def forward_diff_table(y_vals, h):
    return [(y_vals[i+1] - y_vals[i]) / h for i in range(len(y_vals)-1)]

def backward_diff_table(y_vals, h):
    return [(y_vals[i] - y_vals[i-1]) / h for i in range(1, len(y_vals))]

