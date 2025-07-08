import numpy as np
import pandas as pd

def heat_equation_explicit(alpha, L, T, nx, nt, ic_func):
    dx = L / (nx - 1)
    dt = T / (nt - 1)
    u = np.zeros((nt, nx))
    x = np.linspace(0, L, nx)
    u[0, :] = ic_func(x)

    r = alpha * dt / (dx ** 2)
    if r > 0.5:
        raise ValueError("Stability condition violated: alpha*dt/dx^2 <= 0.5")

    for n in range(0, nt - 1):
        for i in range(1, nx - 1):
            u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
        u[n+1, 0] = 0
        u[n+1, -1] = 0

    return x, u

def heat_equation_table(x_vals, u_grid):
    df = pd.DataFrame(u_grid.T, index=x_vals)
    df.index.name = 'x'
    return df
