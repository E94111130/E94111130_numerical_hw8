import math
import pandas as pd
import matplotlib.pyplot as plt

# Set pandas display format: float numbers rounded to 4 decimal places
pd.set_option('display.float_format', '{:.4f}'.format)

# Define the target function f(x) = x²·sin(x)
def f(x):
    return x**2 * math.sin(x)

# === Parameter Setup ===
a, b = 0, 1           # Interval [a, b]
m = 16                # Half the number of nodes (total nodes N = 2m)
N = 2 * m
s = 4                 # Maximum order of Fourier series terms
h = (b - a) / (N - 1) # Step size between nodes

# Generate N equally spaced nodes x, and corresponding z mapped to [-π, π]
x = [a + i * h for i in range(N)]
z = [math.pi * (2 * (xi - a) / (b - a) - 1) for xi in x]  # Linearly map [a, b] → [-π, π]

# Initialize Fourier coefficients A[k], B[k]
A = [0.0] * (s + 1)
B = [0.0] * (s + 1)

# Approximate the Fourier coefficients using a discrete integration (Simpson-like method)
for k in range(s + 1):
    for j in range(N):
        A[k] += f(x[j]) * math.cos(k * z[j]) / m
        B[k] += f(x[j]) * math.sin(k * z[j]) / m

# Define the approximation function S₄(z), truncated at s = 4 terms
def S4(z_val):
    return 0.5 * A[0] + sum(A[k] * math.cos(k * z_val) + B[k] * math.sin(k * z_val) for k in range(1, s + 1))

# (a)
print("(a)")
print(f"a0 = {A[0]:.5f}")
for k in range(1, s + 1):
    print(f"a{k} = {A[k]:.5f} , b{k} = {B[k]:.5f}")

# (b)
# Apply a trapezoidal-like rule for the approximation
approx_integral = sum(S4(z[i]) / (N - 1) for i in range(N - 1))
print(f"\n(b)\n∫₀¹ S4(x)dx = {approx_integral:.5f}")

# (c) 
fx_vals = [f(xi) for xi in x]             # True function values
s4_vals = [S4(zi) for zi in z]            # Approximated values from S₄(z)
pt_errors = [abs(fv - sv) for fv, sv in zip(fx_vals, s4_vals)]  # Absolute error at each point

# Exact value of the integral (from symbolic integration): ∫₀¹ x²·sin(x) dx = cos(1) + 2·sin(1) − 2
true_integral = math.cos(1) + 2 * math.sin(1) - 2
abs_err = abs(true_integral - approx_integral)          # Absolute error
rel_err = abs_err / abs(true_integral) * 100            # Relative error (in percent)

# Display values and errors using a pandas table
df = pd.DataFrame({
    'x':         x,
    'f(x)':      fx_vals,
    'S4(x)':     s4_vals,
    'Error':     pt_errors
})

print("\n(c)")
print(df.to_string(index=False))  # Clean table output

# Show integration and error results
print(f"\nTrue Integral: {true_integral:.5f} , S4 Integral: {approx_integral:.5f}")
print(f"Absolute Error: {abs_err:.5f} , Relative Error: {rel_err:.5f}%")

# === (d) Compute L2 norm of error (square error) ===
square_error = sum((fx_vals[i] - s4_vals[i])**2 for i in range(N))
print(f"\n(d)\nE(S4): {square_error:.5f}")


