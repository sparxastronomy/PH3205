import numpy as np
from math import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# defing functions for integration

# defing the function to integrate


def fs(x):
    return np.exp(-x**2)

# simpson numerical integration


def simpson(f, a, b, n):
    """
    Simpson numerical integration
    Parameters
    ----------
    f : function to integrate
    a : lower limit of integration
    b : upper limit of integration
    n : number of points to use
    """

    h = (b-a)/n  # getting the step size
    # creating the array of x values
    x = np.linspace(a, b, n+1)
    # getting functional values
    y = f(x)
    # initializing the sum
    s = y[0]+y[-1]   # sum of first and last values
    for i in range(1, n):
        if i % 2 == 0:
            s += 2*y[i]
        else:
            s += + 4*y[i]
    s = s*h/3  # multiplying by the step size weight
    return h, s


# array of step sizes
n = np.array([2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120])

# array of simson sums and step sizes
s = np.zeros(len(n))
h = np.zeros(len(n))

# exact value of the integral = sqrt(pi)*erf(1)
exact = np.sqrt(np.pi)*erf(1)

# looping over the step sizes
for i in range(len(n)):
    h[i], s[i] = simpson(fs, -1, 1, int(n[i]))

fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
# plotting the error values with log scaling of the x-axis
ax[0].plot(n, np.abs(exact-s), 'o-', c='b', label='Simpson (number of points)')
ax[1].plot(h, np.abs(exact-s), 'o-', c='b', label='Simposon (Step Size)')
ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_yscale('log')

# formatting the output
ax[0].legend()
ax[1].legend()
ax[0].grid(alpha=0.2)
ax[1].grid(alpha=0.2)
ax[0].set_xlabel('Number of points (n)')
ax[1].set_xlabel('Step Size (h) ')
ax[0].set_ylabel('Error Value')
fig.suptitle('Error v/s step size ($log-log$ Scale)')
#plt.savefig('WS2_1_error.jpg', bbox_inches='tight', dpi=200)

# curve fitting
###############################################################################


def e_fit(x, a, b):
    return a*x + b


# converting the error and the step size to log
# error is taken to be the log of absolute value of the error
y_fit = np.log(np.abs((exact-s)))
x_fit = np.log(h)

# doing the fit
popt, pcov = curve_fit(e_fit, x_fit, y_fit)

# plotting the fit
plt.figure(figsize=(8, 5), dpi=100)
plt.plot(x_fit, e_fit(x_fit, *popt), 'g--',
         label='Fit\nSlope = %2.4f' % popt[0])
plt.plot(x_fit, y_fit, 'o', c='b', label='Error')
plt.legend()
plt.grid(alpha=0.2)
plt.xlabel('log(h)')
plt.ylabel('log(Error)')
plt.title(
    'Error v/s step size ($log-log$ Scale): $\\epsilon\\approx O(h^{%2.2f})$' % popt[0])
#plt.savefig('WS2_1_errorfitted.jpg', bbox_inches='tight', dpi=200)
plt.show()
