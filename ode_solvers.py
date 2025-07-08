import pandas as pd

def euler_method(f, x0, y0, h, n):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    for _ in range(n):
        y += h * f(x, y)
        x += h
        xs.append(x)
        ys.append(y)
    return pd.DataFrame({"x": xs, "y": ys})

def modified_euler_method(f, x0, y0, h, n):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    for _ in range(n):
        k1 = f(x, y)
        k2 = f(x + h, y + h * k1)
        y += h * (k1 + k2) / 2
        x += h
        xs.append(x)
        ys.append(y)
    return pd.DataFrame({"x": xs, "y": ys})

def runge_kutta_2(f, x0, y0, h, n):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    for _ in range(n):
        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        y += h * k2
        x += h
        xs.append(x)
        ys.append(y)
    return pd.DataFrame({"x": xs, "y": ys})

def runge_kutta_4(f, x0, y0, h, n):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    for _ in range(n):
        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        k3 = f(x + h/2, y + h*k2/2)
        k4 = f(x + h, y + h*k3)
        y += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        x += h
        xs.append(x)
        ys.append(y)
    return pd.DataFrame({"x": xs, "y": ys})
