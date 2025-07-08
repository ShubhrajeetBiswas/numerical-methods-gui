import pandas as pd

def bisection_method(f, a, b, tol, max_iter):
    data = []
    if f(a) * f(b) >= 0:
        raise ValueError("Function has same signs at interval endpoints")
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        data.append({"iteration": i+1, "a": a, "b": b, "c": c, "f(c)": fc})
        if abs(fc) < tol or (b - a)/2 < tol:
            break
        if f(a)*fc < 0:
            b = c
        else:
            a = c
    return pd.DataFrame(data)

def newton_raphson_method(f_expr, x0, tol, max_iter):
    import numpy as np
    import pandas as pd
    from sympy import symbols, diff, lambdify
    x = symbols('x')
    f_sym = eval(f_expr)
    df_sym = diff(f_sym, x)
    f = lambdify(x, f_sym, "numpy")
    df = lambdify(x, df_sym, "numpy")
    data = []
    xi = x0
    for i in range(max_iter):
        fxi = f(xi)
        dfxi = df(xi)
        if dfxi == 0:
            raise ZeroDivisionError("Derivative zero. No solution found.")
        x_next = xi - fxi/dfxi
        data.append({"iteration": i+1, "x": xi, "f(x)": fxi})
        if abs(x_next - xi) < tol:
            break
        xi = x_next
    return pd.DataFrame(data)

def regula_falsi_method(f, a, b, tol, max_iter):
    data = []
    if f(a) * f(b) >= 0:
        raise ValueError("Function has same signs at interval endpoints")
    c = a
    for i in range(max_iter):
        c_old = c
        c = b - (f(b)*(a - b)) / (f(a) - f(b))
        fc = f(c)
        data.append({"iteration": i+1, "a": a, "b": b, "c": c, "f(c)": fc})
        if abs(fc) < tol or abs(c - c_old) < tol:
            break
        if f(a)*fc < 0:
            b = c
        else:
            a = c
    return pd.DataFrame(data)

def secant_method(f, x0, x1, tol, max_iter):
    data = []
    for i in range(max_iter):
        if f(x1) - f(x0) == 0:
            raise ZeroDivisionError("Zero denominator in secant method")
        x2 = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))
        data.append({"iteration": i+1, "x0": x0, "x1": x1, "x2": x2, "f(x2)": f(x2)})
        if abs(x2 - x1) < tol:
            break
        x0, x1 = x1, x2
    return pd.DataFrame(data)
