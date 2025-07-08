import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from root_finding import bisection_method, newton_raphson_method, regula_falsi_method, secant_method
from interpolation import (
    lagrange_interpolation,
    lagrange_table,
    forward_diff_table,
    backward_diff_table,
)
import pandas as pd

from least_squares import least_squares_fit, least_squares_table
from integration import trapezoidal_rule, simpsons_rule, forward_difference, backward_difference, central_difference
from ode_solvers import euler_method, modified_euler_method, runge_kutta_2, runge_kutta_4
from pde_solvers import heat_equation_explicit, heat_equation_table

st.set_page_config(page_title="Numerical Methods Toolbox", layout="wide")
st.title("🧮 Numerical Methods GUI")

def evaluate_function(expr, x, y=None):
    try:
        if y is None:
            return eval(expr, {"x": x, "np": np})
        else:
            return eval(expr, {"x": x, "y": y, "np": np})
    except:
        return np.nan

# --- Tabs by Category ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧮 Root-Finding",
    "📐 Interpolation",
    "📊 Least Squares",
    "📏 Integration & Differentiation",
    "🔁 ODE Solvers",
    "🌊 PDE Solvers"
])

# --- Root-Finding Tab ---
with tab1:
    st.subheader("🧮 Root-Finding Methods")
    method = st.selectbox("Select Method", ["Bisection", "Newton-Raphson", "Regula Falsi", "Secant"])
    f_expr = st.text_input("Function f(x)", "x**3 - x - 2")
    f = lambda x: evaluate_function(f_expr, x)

    if method == "Bisection":
        a = st.number_input("a", value=1.0)
        b = st.number_input("b", value=2.0)
        tol = st.number_input("Tolerance", value=1e-6, format="%.1e")
        max_iter = st.number_input("Max Iterations", value=100)
        if st.button("Run Bisection"):
            try:
                df = bisection_method(f, a, b, tol, int(max_iter))
                st.dataframe(df)
                st.success(f"Root ≈ {df['c'].iloc[-1]}")
            except Exception as e:
                st.error(f"Error: {e}")
    elif method == "Newton-Raphson":
        x0 = st.number_input("Initial Guess x₀", value=1.5)
        tol = st.number_input("Tolerance", value=1e-6, format="%.1e")
        max_iter = st.number_input("Max Iterations", value=100)
        if st.button("Run Newton-Raphson"):
            try:
                df = newton_raphson_method(f_expr, x0, tol, int(max_iter))
                st.dataframe(df)
                st.success(f"Root ≈ {df['x'].iloc[-1]}")
            except Exception as e:
                st.error(f"Error: {e}")
    elif method == "Regula Falsi":
        a = st.number_input("a", value=1.0)
        b = st.number_input("b", value=2.0)
        tol = st.number_input("Tolerance", value=1e-6, format="%.1e")
        max_iter = st.number_input("Max Iterations", value=100)
        if st.button("Run Regula Falsi"):
            try:
                df = regula_falsi_method(f, a, b, tol, int(max_iter))
                st.dataframe(df)
                st.success(f"Root ≈ {df['c'].iloc[-1]}")
            except Exception as e:
                st.error(f"Error: {e}")
    elif method == "Secant":
        x0 = st.number_input("x₀", value=1.0)
        x1 = st.number_input("x₁", value=2.0)
        tol = st.number_input("Tolerance", value=1e-6, format="%.1e")
        max_iter = st.number_input("Max Iterations", value=100)
        if st.button("Run Secant"):
            try:
                df = secant_method(f, x0, x1, tol, int(max_iter))
                st.dataframe(df)
                st.success(f"Root ≈ {df['x2'].iloc[-1]}")
            except Exception as e:
                st.error(f"Error: {e}")

# --- Interpolation Tab ---
with tab2:
    st.subheader("📐 Interpolation Methods")

    method = st.selectbox("Select Method", ["Lagrange", "Forward Difference", "Backward Difference"])

    x_data = st.text_input("x values (comma-separated)", "1,2,3,4")
    y_data = st.text_input("y values (comma-separated)", "1,4,9,16")

    try:
        x_vals = list(map(float, x_data.split(",")))
        y_vals = list(map(float, y_data.split(",")))

        if len(x_vals) > 1:
            h = x_vals[1] - x_vals[0]
        else:
            h = 1

        if method == "Lagrange":
            x_val = st.number_input("x to interpolate", value=2.5)
            y_interp = lagrange_interpolation(x_vals, y_vals, x_val)
            df_interp = lagrange_table(x_vals, y_vals)
            st.dataframe(df_interp)
            st.success(f"Interpolated value at x={x_val}: {y_interp:.6f}")

        elif method == "Forward Difference":
            fdiffs = forward_diff_table(y_vals, h)
            df_fd = pd.DataFrame({"Forward Differences": fdiffs})
            st.dataframe(df_fd)

        elif method == "Backward Difference":
            bdiffs = backward_diff_table(y_vals, h)
            df_bd = pd.DataFrame({"Backward Differences": bdiffs})
            st.dataframe(df_bd)

    except Exception as e:
        st.error(f"Error: {e}")

# --- Least Squares Tab ---
with tab3:
    st.subheader("📊 Least Squares Method")
    x_data_ls = st.text_input("x values (comma-separated)", "1,2,3,4,5")
    y_data_ls = st.text_input("y values (comma-separated)", "2.2,2.8,3.6,4.5,5.1")
    degree = st.number_input("Polynomial Degree", min_value=1, max_value=10, value=1)

    try:
        x_vals = list(map(float, x_data_ls.split(",")))
        y_vals = list(map(float, y_data_ls.split(",")))
        coeffs, y_pred = least_squares_fit(x_vals, y_vals, degree)
        df_ls = least_squares_table(x_vals, y_vals, y_pred)
        st.success("Fit successful!")
        st.dataframe(df_ls)
        x_range = np.linspace(min(x_vals), max(x_vals), 500)
        poly_fit = np.poly1d(coeffs)
        fig, ax = plt.subplots()
        ax.scatter(x_vals, y_vals, color='blue', label='Data')
        ax.plot(x_range, poly_fit(x_range), color='red', label=f'Fit (deg {degree})')
        ax.grid(True); ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error: {e}")

# --- Integration & Differentiation Tab ---
with tab4:
    st.subheader("📏 Numerical Integration and Differentiation")
    method = st.selectbox("Select Method", ["Trapezoidal Rule", "Simpson's Rule", "Forward Difference", "Backward Difference", "Central Difference"])
    f_expr = st.text_input("Function f(x)", "np.sin(x)")
    f = lambda x: evaluate_function(f_expr, x)

    if method in ["Trapezoidal Rule", "Simpson's Rule"]:
        a = st.number_input("a", value=0.0)
        b = st.number_input("b", value=np.pi)
        n = st.number_input("n (intervals)", min_value=2, value=10, step=2)
        if st.button("Compute Integral"):
            try:
                if method == "Trapezoidal Rule":
                    result = trapezoidal_rule(f, a, b, int(n))
                else:
                    if int(n) % 2 != 0:
                        st.error("n must be even for Simpson's Rule")
                        result = None
                    else:
                        result = simpsons_rule(f, a, b, int(n))
                if result is not None:
                    st.success(f"Approximate Integral = {result:.6f}")
            except Exception as e:
                st.error(str(e))
    else:
        x = st.number_input("x", value=1.0)
        h = st.number_input("Step size h", value=0.01)
        if st.button("Compute Derivative"):
            try:
                if method == "Forward Difference":
                    result = forward_difference(f, x, h)
                elif method == "Backward Difference":
                    result = backward_difference(f, x, h)
                else:
                    result = central_difference(f, x, h)
                st.success(f"Approximate Derivative = {result:.6f}")
            except Exception as e:
                st.error(str(e))

# --- ODE Solvers Tab ---
with tab5:
    st.subheader("🔁 ODE Solvers")
    method = st.selectbox("Select ODE Method", ["Euler Method", "Modified Euler Method", "Runge-Kutta 2nd Order", "Runge-Kutta 4th Order"])
    ode_expr = st.text_input("dy/dx = f(x, y)", "x + y")
    x0 = st.number_input("Initial x₀", value=0.0)
    y0 = st.number_input("Initial y₀", value=1.0)
    h = st.number_input("Step size h", value=0.1)
    n = st.number_input("Steps n", min_value=1, value=10)

    if st.button("Solve ODE"):
        f = lambda x, y: evaluate_function(ode_expr, x, y)
        try:
            if method == "Euler Method":
                df = euler_method(f, x0, y0, h, int(n))
            elif method == "Modified Euler Method":
                df = modified_euler_method(f, x0, y0, h, int(n))
            elif method == "Runge-Kutta 2nd Order":
                df = runge_kutta_2(f, x0, y0, h, int(n))
            else:
                df = runge_kutta_4(f, x0, y0, h, int(n))
            st.success("Solved!")
            st.dataframe(df)
            fig, ax = plt.subplots()
            ax.plot(df["x"], df["y"], marker='o', label="y(x)")
            ax.set_title(f"{method}")
            ax.grid(True); ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# --- PDE Solvers Tab ---
with tab6:
    st.subheader("🌊 PDE Solvers")
    st.markdown("Currently implemented: 1D Heat Equation (Explicit FTCS Method)")
    alpha = st.number_input("Thermal Diffusivity α", value=0.01)
    L = st.number_input("Length of Rod (L)", value=1.0)
    T = st.number_input("Total Time (T)", value=0.1)
    nx = st.number_input("Spatial Steps (nx)", min_value=10, value=20)
    nt = st.number_input("Time Steps (nt)", min_value=10, value=50)
    ic_expr = st.text_input("Initial Condition u(x,0) = f(x)", "np.sin(np.pi * x)")
    if st.button("Solve Heat Equation"):
        try:
            ic_func = lambda x: evaluate_function(ic_expr, x)
            x_vals, u_grid = heat_equation_explicit(alpha, L, T, int(nx), int(nt), ic_func)
            df_heat = heat_equation_table(x_vals, u_grid)
            st.success("Solved!")
            st.dataframe(df_heat)
            fig, ax = plt.subplots()
            for i in range(0, len(u_grid), max(1, len(u_grid)//5)):
                ax.plot(x_vals, u_grid[i], label=f"t={i}")
            ax.set_xlabel("x"); ax.set_ylabel("u(x, t)"); ax.set_title("Heat Equation Evolution")
            ax.grid(True); ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")
