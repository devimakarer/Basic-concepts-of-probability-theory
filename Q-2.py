import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as multivaritate_normal


mean = [0, 1]
covariance_matrix = [1, 0.1], [0.1, 2]

samples = np.random.multivariate_normal(mean, covariance_matrix, 200, check_valid='warn', tol=1e-8)

x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 5, 100))
pos = np.dstack((x, y))

rv = multivaritate_normal.multivariate_normal(mean, covariance_matrix)

plt.figure("Plot the PDF of the samples")
ax = plt.axes(projection = "3d")
ax.plot_surface(x, y, rv.pdf(pos), cmap = "viridis")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability Density')
ax.set_title('Bivariate Normal Distribution - Probability Density Function')

plt.show()


plt.hist( samples[:, 0], label='PDF of X')
plt.title('PDF of the First Variable (X)')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

plt.hist( samples[:, 1], label='PDF of Y')
plt.title('PDF of the Second Variable (Y)')
plt.xlabel('Y')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

correlation = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
print(f"Correlation between X and Y: {correlation:.3f}\n")

plt.scatter(samples[:, 0], samples[:, 1], marker='o', alpha=0.7)
plt.title('Scatter Plot of X and Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

z= samples[:, 0] + samples[:, 1]

print(f"Expected value of Z E[Z] = {np.mean(z):.3f}")
print(f"Variance of Z VAR[Z] = {np.var(z):.3f}\n")

plt.hist(z, label='PDF of z')
plt.xlabel('z')
plt.ylabel('Probability Density')
plt.legend()
plt.show()


correlation_XZ = np.corrcoef(samples[:, 0], z)[0, 1]
covariance_XZ = np.cov(samples[:, 0], z)[0, 1]

print(f"Correlation between X and Z: {correlation_XZ:.3f}")
print(f"Covariance between X and Z: {covariance_XZ:.3f}\n")

plt.scatter(samples[:, 0], z)
plt.title('Scatter Plot of X and Z')
plt.xlabel('X')
plt.ylabel('Z = X + Y')
plt.show()

correlation_XY = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
covariance_XY = np.cov(samples[:, 0], samples[:, 1])[0, 1]

correlation_XZ = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
covariance_XZ = np.cov(samples[:, 0], samples[:, 1])[0, 1]

print(f"Correlation between X and Y: {correlation_XY:.3f}")
print(f"Covariance between X and Y: {covariance_XY:.3f}")

print(f"Correlation between X and Z: {correlation_XZ:.3f}")
print(f"Covariance between X and Z: {covariance_XZ:.3f}")

