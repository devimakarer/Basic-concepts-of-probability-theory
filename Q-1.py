import numpy as np
import matplotlib.pyplot as plt

mu = 1
x = np.linspace(0, 2, 1000)
y = x**2 + 1

#pdf of x
def pdfx(x):
    return np.where((0 < x) & (x <= 2), 5/32*x**4, 0)

#cdf of x
def cdfx(x):
    return np.where((0 < x) & (x <= 2), 5/160*x**5, 0)

cdfx = [pdfx(xi) for xi in x]

mean_x = np.trapz(x * pdfx(x), x)

#pdf of y
def pdfy(y):
    return np.where((1 < y) & (y <= 5), (5/32) * np.sqrt(y - 1)**4 * 1/(2*np.sqrt(y - 1)), 0)

#cdf of y
def cdfy(y):
    return np.where((1 < y) & (y <= 5), ((y-1)**(5/2))/32, 0)

cdfy = [pdfy(yi) for yi in y]

mean_y = np.trapz(y * pdfy(y), y)


plt.plot(x, cdfx,label="CDF of X ", color="violet")
plt.axvline(mean_x, label = f"Mean of X: {mean_x:.3f}", color = "black" , linestyle = "--")
plt.xlabel('X')
plt.ylabel('Probability')
plt.title("CDF of X")
plt.legend()
plt.show()


plt.plot(y,cdfy,label="CDF of Y", color="green")
plt.axvline(mean_y, label = f"Mean of Y: {mean_y:.3f}", color = "pink" , linestyle = "--")
plt.xlabel('CDF(y)')
plt.ylabel('Probability')
plt.title("Y")
plt.legend()
plt.show()
