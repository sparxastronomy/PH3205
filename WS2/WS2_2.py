import matplotlib.pyplot as plt
import numpy as np

# defining the function to integrate


def fg(x):
    return (4*x + 4)*np.exp(4*x + 4)

# defining function for gaussian quadrature integration


def gauss(f, n):
    """
    Gaussian quadrature integration
    f: function to integrate
    n: number of points
    """
    # dictionary of weights and abscissas
    abscissas = {
        2: np.array([0.577350269189626, -0.577350269189626]),
        3: np.array([0.774596669241483, 0, -0.774596669241483]),
        4: np.array([0.861136311594053, 0.339981043584856, -0.339981043584856, -0.861136311594053]),
        5: np.array([0.906179845938664, 0.538469310105683, 0, -0.538469310105683, -0.906179845938664]),
    }
    weights = {
        2: np.array([1, 1]),
        3: np.array([0.555555555555556, 0.888888888888889, 0.555555555555556]),
        4: np.array([0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454]),
        5: np.array([0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189]),
    }
    if n > 5 or n < 2:
        print('n must be between 2 and 5')
        return
    else:
        # getting required weights
        w = weights[n]
        # getting the function values
        y = f(abscissas[n])
        # initializing the sum
        s = y*w
        return np.sum(s)


# exact value of the integral to compare with the gaussian quadrature
# I = Integral(te^(2t)) from 0 to 4
exact_gauss = (1/4)*(1 + (7*np.exp(8)))

# array of n values
n = np.array([2, 3, 4, 5])

gauss_integral = np.zeros(len(n))
for i in range(len(n)):
    gauss_integral[i] = gauss(fg, n[i])

# plotting the resulsts
fig, ax = plt.subplots(1, 2, figsize=(13, 6), dpi=100)
ax[0].axhline(y=exact_gauss, color='r', ls='--', label='Exact')
ax[0].plot(n, gauss_integral, 'o-', c='g', ms=7, label='Gaussian Integration')
ax[1].axhline(y=0, color='r', ls='--', label='Exact')
ax[1].plot(n, exact_gauss-gauss_integral, 'o-', c='b', ms=7, label='Error')

# formatting the output
ax[0].legend()
ax[1].legend()
ax[0].grid(alpha=0.2)
ax[1].grid(alpha=0.2)
ax[0].set_xticks(n)
ax[1].set_xticks(n)
ax[0].set_xlabel('Order of Gaussian Quadrature (n)', fontsize=16)
ax[1].set_xlabel('Order of Gaussian Quadrature (n)', fontsize=16)
ax[0].set_ylabel('Integral Value', fontsize=16)
ax[1].set_ylabel('Error (Exact-computed)', fontsize=16)
ax[0].set_title('Gaussian Quadrature Integration')
ax[1].set_title('Error v/s Order of Gaussian Quadrature')
fig.suptitle('Gaussian Quadrature Integration')
# plt.savefig('Gaussian_Quadrature_Integration.jpg',
#             bbox_inches='tight', dpi=200)

plt.show()
