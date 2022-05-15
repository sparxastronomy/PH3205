import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# defining the function for Box-Muller transformation


def BoxMuller(n):
    """
    n = number of samples
    """
    u = np.random.uniform(0, 1, n)
    v = np.random.uniform(0, 1, n)
    x = np.sqrt(-2*np.log(u))*np.cos(2*np.pi*v)
    y = np.sqrt(-2*np.log(u))*np.sin(2*np.pi*v)
    return np.array([x, y])

# genralizing the function for Box-Muller transformation


def BoxMuller_general(n, mu, sigma):
    """
    n = number of samples
    mu = mean
    sigma = standard deviation
    """
    u = np.random.uniform(0, 1, n)
    v = np.random.uniform(0, 1, n)
    x = np.sqrt(-2*np.log(u))*np.cos(2*np.pi*v)
    y = np.sqrt(-2*np.log(u))*np.sin(2*np.pi*v)
    return np.array([x, y])*sigma+mu


# generating N samples using Box-Muller transformation
N = 10000
x, y = BoxMuller(N)
# computing the histrogram and storing it in a variable
hist, bins = np.histogram(x, bins=100, density=True)
# the densitiy is calulated by the formula: hist/N*(bins[1]-bins[0])

# plotting the histogram
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(bins[:-1], hist, label='Distribution of Random Numbers')

# plotting a normal distribution with mean 0 and standard deviation 1
x = np.linspace(-5, 5, 100)
plt.plot(x, norm.pdf(x, 0, 1), color='red', label='$\\mathcal{N}(0,1)$')

plt.xlabel('x')
plt.ylabel('p(x) (PDF)')
plt.legend(loc='best')
plt.title("Random Numbers by using Box-Muller Transformation")

#plt.savefig("Problem1_1.jpg", bbox_inches='tight', dpi=200)

# part 2
# repeating the above process for different mean and standard deviation
mu = [1, 3, 5]
sigma = [2.5, 4, 6]

# number of samples
N = 20000
fig = plt.figure(figsize=(20, 15), dpi=100)
grid = plt.GridSpec(3, 3, wspace=0.2, hspace=0.2)
# slecting grids for individual plots
ax1 = fig.add_subplot(grid[2, 0])
ax2 = fig.add_subplot(grid[2, 1])
ax3 = fig.add_subplot(grid[2, 2])
# selecting grid for x-z plot
ax4 = fig.add_subplot(grid[0:2, :])

ax = [ax1, ax2, ax3]

for i in range(len(mu)):
    x, y = BoxMuller_general(N, mu[i], sigma[i])
    # getting the individual histograms
    hist, bins = np.histogram(x, bins=150, density=True)

    # plotting the histogram
    ax[i].hist(x, bins=150, density=True, rwidth=0.7, label='Distribution')
    ax4.plot(bins[:-1], hist,
             label='Random Number for: $\\mu=%2.2f, \\sigma=%2.2f$' % (mu[i], sigma[i]))

    # plotting the corresponding normal distribution
    x = np.linspace(min(x), max(x), 100)
    ax[i].plot(x, norm.pdf(x, mu[i], sigma[i]), ls='--', lw=1.5,
               label='PDF of $\\mathcal{N}('+str(mu[i])+','+str(sigma[i])+')$')
    ax4.plot(x, norm.pdf(x, mu[i], sigma[i]), ls='--', lw=1.5,
             label='PDF of $\\mathcal{N}('+str(mu[i])+','+str(sigma[i])+')$')

    # formatting the plots
    ax[i].legend(loc='best')
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('p(x) [PDF]')
    ax[i].set_title('Histogram for $\\mu=%2.2f, \\sigma=%2.2f$' %
                    (mu[i], sigma[i]), fontsize=12)

ax4.legend(loc='best')
ax4.set_title('All Plots together', fontsize=16)
plt.suptitle(
    "Random Numbers by using Box-Muller Transformation for different mean and standard deviation", fontsize=24)

#plt.savefig("Problem1_2.jpg", bbox_inches='tight', dpi=200)
plt.show()
