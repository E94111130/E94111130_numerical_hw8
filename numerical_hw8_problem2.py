# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:38:53 2025

@author: user
"""

import numpy as np
from scipy.integrate import quad

# Define the function f(x)
def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2 * x)

# Basis functions for degree 2 polynomial
phi0 = lambda x: 1
phi1 = lambda x: x
phi2 = lambda x: x**2
basis = [phi0, phi1, phi2]

# Compute the elements of the normal equations
A = np.zeros((3, 3))
b = np.zeros(3)

# Fill matrix A and vector b
for i in range(3):
    for j in range(3):
        A[i, j], _ = quad(lambda x: basis[i](x) * basis[j](x), -1, 1)
    b[i], _ = quad(lambda x: f(x) * basis[i](x), -1, 1)

# Solve the normal equations to get coefficients
coeffs = np.linalg.solve(A, b)

# Format polynomial as string
poly_str = f"{coeffs[2]:.6f}x^2 + {coeffs[1]:.6f}x + {coeffs[0]:.6f}"

# Compute mean square error over [-1, 1]
def p(x):  # Polynomial approximation
    return coeffs[0] + coeffs[1] * x + coeffs[2] * x**2

error_integral, _ = quad(lambda x: (f(x) - p(x))**2, -1, 1)

# Print results
print("Least Squares Polynomial Approximation (degree 2) for f(x) on [-1, 1]:")
print(f"  p(x) ≈ {poly_str}")
print(f"  Mean squared error (integral of squared error): {error_integral:.6f}")
