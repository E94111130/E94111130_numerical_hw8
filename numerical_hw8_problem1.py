import numpy as np
from scipy.optimize import curve_fit

# Given data
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# (a) Polynomial (degree 2) least squares approximation
coeffs_poly2 = np.polyfit(x, y, 2)
poly2 = np.poly1d(coeffs_poly2)
y_poly2 = poly2(x)
error_poly2 = np.sum((y - y_poly2) ** 2)

# Format polynomial
a2, a1, a0 = coeffs_poly2
poly_str = f"{a2:.4f}x^2 + {a1:.4f}x + {a0:.4f}"

print("(a) Quadratic Polynomial Least Squares Approximation:")
print(f"  y ≈ {poly_str}")
print(f"  Sum of squared errors: {error_poly2:.5f}\n")

# (b) Exponential: y = b * e^(a * x)
# Linearize: ln(y) = ln(b) + a * x
log_y = np.log(y)
A_exp = np.vstack([x, np.ones_like(x)]).T
a_exp, log_b_exp = np.linalg.lstsq(A_exp, log_y, rcond=None)[0]
b_exp = np.exp(log_b_exp)
y_exp = b_exp * np.exp(a_exp * x)
error_exp = np.sum((y - y_exp) ** 2)

print("(b) Exponential Fit (y = b * e^(a * x)):")
print(f"  a ≈ {a_exp:.4f}")
print(f"  b ≈ {b_exp:.4f}")
print(f"  Sum of squared errors: {error_exp:.5f}\n")

# (c) Power function: y = b * x^r
# Linearize: ln(y) = ln(b) + r * ln(x)
log_x = np.log(x)
A_pow = np.vstack([log_x, np.ones_like(log_x)]).T
r_pow, log_b_pow = np.linalg.lstsq(A_pow, np.log(y), rcond=None)[0]
b_pow = np.exp(log_b_pow)
y_power = b_pow * x ** r_pow
error_power = np.sum((y - y_power) ** 2)

print("(c) Power Fit (y = b * x^n):")
print(f"  n ≈ {r_pow:.4f}")
print(f"  b ≈ {b_pow:.4f}")
print(f"  Sum of squared errors: {error_power:.5f}")


