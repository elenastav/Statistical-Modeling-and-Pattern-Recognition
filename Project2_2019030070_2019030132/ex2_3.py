import numpy as np

# Sample data for class ω1
samples_omega1 = np.array([
    [0.42, -0.087, 0.58],
    [-0.2, -3.3, -3.4],
    [1.3, -0.32, 1.7],
    [0.39, 0.71, 0.23],
    [-1.6, -5.3, -0.15],
    [-0.029, 0.89, -4.7],
    [-0.23, 1.9, 2.2],
    [0.27, -0.3, -0.87],
    [-1.9, 0.76, -2.1],
    [0.87, -1, -2.6]
])

# Calculate MLE for mean and variance for each characteristic
mu_hat = np.mean(samples_omega1, axis=0)
sigma_hat_squared = np.var(samples_omega1, axis=0, ddof=0)  # Using ddof=0 for MLE

print("Mean (μ̂) for each characteristic:", mu_hat)
print("Variance (σ̂^2) for each characteristic:", sigma_hat_squared)

# Calculate MLE for mean vector and covariance matrix for each pair of features
def mle_2d(samples):
    mu_hat = np.mean(samples, axis=0)
    sigma_hat = np.cov(samples, rowvar=False, ddof=0)  # Using ddof=0 for MLE
    return mu_hat, sigma_hat

pairs = [
    (0, 1),
    (0, 2),
    (1, 2)
]

for i, (idx1, idx2) in enumerate(pairs):
    pair_samples = samples_omega1[:, [idx1, idx2]]
    mu_hat, sigma_hat = mle_2d(pair_samples)
    print(f"Pair {i+1}: Features {idx1+1} and {idx2+1}")
    print("Mean vector (μ̂):", mu_hat)
    print("Covariance matrix (Σ̂):\n", sigma_hat)
    print()

# Calculate MLE for mean vector and covariance matrix for 3D features
mu_hat_3d = np.mean(samples_omega1, axis=0)
sigma_hat_3d = np.cov(samples_omega1, rowvar=False, ddof=0)  # Using ddof=0 for MLE

print("Mean vector (μ̂) for 3D:", mu_hat_3d)
print("Covariance matrix (Σ̂) for 3D:\n", sigma_hat_3d)

# Calculate MLE for mean vector and diagonal covariance matrix for 3D features
mu_hat_diag = np.mean(samples_omega1, axis=0)
sigma_hat_diag_squared = np.var(samples_omega1, axis=0, ddof=0)  # Using ddof=0 for MLE

print("Mean vector (μ̂) for diagonal covariance matrix:", mu_hat_diag)
print("Diagonal elements of covariance matrix (σ̂^2):", sigma_hat_diag_squared)

# Sample data for class ω2
samples_omega2 = np.array([
    [-0.4, 0.58, 0.089],
    [-0.31, 0.27, -0.04],
    [0.38, 0.055, -0.035],
    [-0.15, 0.53, 0.011],
    [-0.35, 0.47, 0.034],
    [0.17, 0.69, 0.1],
    [-0.011, 0.55, -0.18],
    [-0.27, 0.61, 0.12],
    [-0.065, 0.49, 0.0012],
    [-0.12, 0.054, -0.063]
])

# Part (a) for ω2
mu_hat_omega2 = np.mean(samples_omega2, axis=0)
sigma_hat_squared_omega2 = np.var(samples_omega2, axis=0, ddof=0)

print("ω2 - Mean (μ̂) for each characteristic:", mu_hat_omega2)
print("ω2 - Variance (σ̂^2) for each characteristic:", sigma_hat_squared_omega2)

# Part (b) for ω2
for i, (idx1, idx2) in enumerate(pairs):
    pair_samples = samples_omega2[:, [idx1, idx2]]
    mu_hat, sigma_hat = mle_2d(pair_samples)
    print(f"ω2 - Pair {i+1}: Features {idx1+1} and {idx2+1}")
    print("Mean vector (μ̂):", mu_hat)
    print("Covariance matrix (Σ̂):\n", sigma_hat)
    print()

# Part (c) for ω2
mu_hat_3d_omega2 = np.mean(samples_omega2, axis=0)
sigma_hat_3d_omega2 = np.cov(samples_omega2, rowvar=False, ddof=0)

print("ω2 - Mean vector (μ̂) for 3D:", mu_hat_3d_omega2)
print("ω2 - Covariance matrix (Σ̂) for 3D:\n", sigma_hat_3d_omega2)

# Part (d) for ω2
mu_hat_diag_omega2 = np.mean(samples_omega2, axis=0)
sigma_hat_diag_squared_omega2 = np.var(samples_omega2, axis=0, ddof=0)

print("ω2 - Mean vector (μ̂) for diagonal covariance matrix:", mu_hat_diag_omega2)
print("ω2 - Diagonal elements of covariance matrix (σ̂^2):", sigma_hat_diag_squared_omega2)
