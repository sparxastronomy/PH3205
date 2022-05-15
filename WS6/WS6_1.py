import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

# defining the function to be integrated


def f(x):
    val = np.zeros(len(x))
    a = x[np.where(x <= np.pi)[0]]
    # getting non-zero values
    val_nz = np.power(np.power(a, 2) + np.power(np.cos(a), 2), -1)
    val[np.where(x <= np.pi)[0]] = val_nz
    return val

# defining the weight function


def w(x, a):
    return a*np.exp(-1*x)

# defining function to generate ramdom variables from the weight functions


def r(N, a):
    y = np.random.uniform(0, 1, N)
    return np.log(1/(-y/a + 1))

# integrating by imporatance sampling


def imp_sampling(f, w, r, N, a):
    """
    Performs Monte-calro integration using importance sampling

    Paramters
    ========================
    f: function to be integrated
    w: weigth function
    r: random variable generator
    N: Number of points
    a: parameter of the weight function
    """
    y_i = r(N, a)
    y = f(y_i)
    w_x = w(y_i, a)

    # estimator of integral
    I = np.sum(y/w_x)/N

    # calulating variance from the estimator
    F = y/w_x
    var = np.sum(F**2)/N - I**2
    return np.array([I, var])


# calulating the integral using the imporance sampling for differnt N
N = np.linspace(100, 1000000, 101)
val = []
for i in range(len(N)):
    val.append(imp_sampling(f, w, r, int(N[i]), 1)[0])

exact_val = 1.58119

# plotting the results
plt.figure(figsize=(14, 6), dpi=100)
plt.plot(N, val, c='green', marker='o', mec='blue',
         label='MC Integration(Importance Sampling)')
plt.axhline(exact_val, ls='--', color='red', label='True Value')
plt.xlabel('Number of points', fontsize=14)
plt.ylabel('Estimated Integral', fontsize=14)
plt.legend()

#plt.savefig('WS5_1_a.jpg', bbox_inches='tight', dpi=300)

# calulating the integral using the imporance sampling method for differnt a
A = np.linspace(1, 2, 100)
val = []
var = []
for i in range(len(A)):
    results = imp_sampling(f, w, r, 100000, A[i])
    val.append(results[0])
    var.append(results[1])

# plotting the results
fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

# plotting values in subplot 1
ax[0].plot(A, val, marker='o', ms=2.5, mec='green',
           label='MC Integration(Importance Sampling)')
ax[0].axhline(exact_val, ls='--', label='True Value')
ax[0].set_xlabel('a', fontsize=14)
ax[0].set_ylabel('Estimated Intergral ($\\hat{I}$)', fontsize=14)
ax[0].legend(loc='best')

# plotting variance in subplot 2
ax[1].plot(A, var, marker='o', ms=2.5, mec='red',
           label='MC Integration(Importance Sampling)')
ax[1].axhline(0, ls='--', label='No Variance($\\sigma^2=0$)')
ax[1].set_xlabel('a', fontsize=14)
ax[1].set_ylabel('Variance $(\\sigma^2)$', fontsize=14)

plt.suptitle(
    "Comparing the result and variance for different values of 'a' in $w(x) = ae^{-x}$", fontsize=17)
#plt.savefig('WS5_1_b.jpg', bbox_inches='tight', dpi=300)

plt.show()
