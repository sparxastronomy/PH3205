import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
from numba import njit
from tqdm import tqdm

# defining function for unbiased random walk with equal probability


@njit(cache=True)
def random_walk(n, x0):
    """
    Returns the final postion on 1D lattice staring from x0
    p: probability of hopping to right = 0.5
    q: probabillity of hopping to left = 0.5

    prameters
    ========================================
    n: number of steps
    x0: initial position
    """
    x = x0
    r = np.random.random(n)
    # for i in range(n):
    # if r[i] < 0.5:
    # x += 1
    # else:
    # x -= 1

    P = len(np.where(r < 0.5)[0])
    Q = len(np.where(r >= 0.5)[0])
    x = x + (P-Q)
    return x

# part a
# storing the final positions of random walks in a variable


# number of random walks
M = 100000
# number of steps
N = 10000
# initial position
x0 = 0

x = np.zeros(M)
for i in tqdm(range(M), "Random Walks"):
    x[i] = random_walk(N, x0)

plt.figure(figsize=(10, 5), dpi=100)
plt.hist(x, bins=150, density=True, color='#3475D0',
         rwidth=0.5, label='Distribution of Final Positions')

# calulating mu and sigma of random walks
mu = np.mean(x)
sigma = np.std(x)

x = np.linspace(min(x), max(x), 100)
plt.plot(x, norm.pdf(x, mu, sigma), color='red',
         label='$\\mathcal{N}(%2.2f, %2.2f)$' % (mu, sigma))
plt.xlabel('x')
plt.ylabel('p(x) (PDF)')
plt.legend(loc='best')
plt.title("Random Walks: Step Size = %d, Number of Walks = %d" % (N, M))
# plt.savefig("Problem2_1.jpg", bbox_inches='tight', dpi=200)

# part b
# plotting std deviation of distribution for different number of steps
N = np.linspace(10, 100000, 100+1)
std = np.zeros(len(N))
# number of random walks
M = 1000
for i in range(len(N)):
    x = np.zeros(M)
    for j in range(M):
        x[j] = random_walk(int(N[i]), 0)
    std[i] = np.std(x)
# fitting the STD to square root function
# definging fitting functions


def fit(x, a, b):
    return a*np.sqrt(x)+b


def fit_log(x, a, b):
    return a*(x)+b


# fitting the standard deviation
popt, pcov = curve_fit(fit, N, std)
popt1, pcov1 = curve_fit(fit_log, np.log(N), np.log(std))
# plotting the data and the fitted function
fig, ax = plt.subplots(1, 3, figsize=(20, 5), dpi=100)
# plotting the data in the first subplot
ax[0].plot(N, std, color='#1485D0', label='Standard Deviation of Distribution')

# plotting the fitted function in the second subplot
ax[1].plot(N, std, color='#3475D0', label='Standard Deviation of Distribution')
ax[1].plot(N, fit(N, *popt), ls='--', color='red',
           label='$A\sqrt{N}+B~ (A=%2.2f, B=%2.2f)$' % (popt[0], popt[1]))
ax[1].legend(loc='best')
# plotting the fitted function in the third subplot
ax[2].plot(np.log(N), np.log(std), color='#3475D0',
           label='Standard Deviation of Distribution')
ax[2].plot(np.log(N), fit_log(np.log(N), *popt1), ls='--', color='red',
           label='$A\log(N)+B~ (A=%2.2f, B=%2.2f)$' % (popt1[0], popt1[1]))
ax[2].legend(loc='best')
for i in range(2):
    ax[i].set_xlabel('Step Size (N)', fontsize=15)
    ax[i].set_ylabel('Standard Deviation ($\\sigma$)', fontsize=15)
ax[2].set_xlabel('$\\log(N)$',  fontsize=15)
ax[2].set_ylabel('$\\log(\sigma)$', fontsize=15)
plt.suptitle('Standard Deviation of Distribution for different N ', fontsize=16)
# plt.savefig('STD_vs_N.jpg', bbox_inches='tight', dpi=200)
plt.show()
