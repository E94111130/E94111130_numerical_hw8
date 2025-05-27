import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Step 1: Setup
m = 16
n = 4
a, b = 0, 1
x = np.linspace(a, b, m)
f = lambda x: x**2 * np.sin(x)
y = f(x)

# Step 2: Transform [0,1] to [-π, π]
z = 2 * np.pi * x - np.pi  # map [0,1] -> [-π, π]

# Step 3: Compute coefficients for S_4
a0 = (1/m) * np.sum(y)
ak = [(2/m) * np.sum(y * np.cos(k * z)) for k in range(1, n)]
bk = [(2/m) * np.sum(y * np.sin(k * z)) for k in range(1, n)]

# Print coefficients
print(f"a0 / 2 = {a0/2:.6f}")
for k in range(1, n):
    print(f"a{k} = {ak[k-1]:.6f}, b{k} = {bk[k-1]:.6f}")

# Print polynomial expression
terms = [f"{a0/2:.6f}"]
for k in range(1, n):
    terms.append(f"{ak[k-1]:+.6f}*cos({k}z)")
    terms.append(f"{bk[k-1]:+.6f}*sin({k}z)")
print("\nS4(x) = " + " ".join(terms))
print("where z = 2πx - π")

# Define S4 on x in [0,1]
def S4(x_val):
    z_val = 2 * np.pi * x_val - np.pi
    sum_val = a0 / 2
    for k in range(1, n):
        sum_val += ak[k-1] * np.cos(k * z_val) + bk[k-1] * np.sin(k * z_val)
    return sum_val

# Step 4: Compute ∫ S_4(x) dx over [0,1]
S4_vec = np.vectorize(S4)
integral_S4, _ = quad(S4, 0, 1)
print(f"\nIntegral of S4(x) over [0,1]: {integral_S4:.8f}")

# Step 5: Compare to true integral ∫ x^2 sin(x) dx
true_integral, _ = quad(lambda x: x**2 * np.sin(x), 0, 1)
print(f"True integral of x^2*sin(x) over [0,1]: {true_integral:.8f}")

# Step 6: Compute error E(S_4)
S4_values = S4_vec(x)
error = np.sum((y - S4_values)**2)
print(f"Error E(S4): {error:.8f}")

# Optional: Plot the function and the approximation
x_plot = np.linspace(0, 1, 400)
plt.plot(x_plot, f(x_plot), label="f(x) = x² sin(x)")
plt.plot(x_plot, S4_vec(x_plot), label="S₄(x)", linestyle='--')
plt.legend()
plt.title("Discrete Least Squares Trigonometric Approximation")
plt.grid(True)
plt.show()


