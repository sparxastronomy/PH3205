import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["font.family"] = "serif"

# defining the function to generate random numbers from the above distribution


def r(N):
    """
    Generates N random numbers from the distribution.
    N: number of random numbers to generate
    """
    y = np.random.uniform(0, 1, N)
    return np.power(y, 5/4)


# generating random numbers from the above distribution

# Number of random numbers
N = 500000

# generating random numbers
y = r(N)

# plotting the distribution
fig = plt.figure(figsize=(12, 6), dpi=100)
# bin edges using friedman diaconis rule
edges = np.histogram_bin_edges(y, bins='fd')
# getting midpoints of the bins
bin_centers = (edges[1:] + edges[:-1])/2

vals = plt.hist(y, bins=edges, density=True, color='green', edgecolor='blue', alpha=0.5,
                label='Distribution of values')
plt.plot(bin_centers, vals[0], 'o', ms=4, c='orange')

# plotting the distribution
x = np.linspace(1e-2, 1, 100)
y_dist = (4/5)*np.power(x, -1/5)
plt.plot(x, y_dist, color='red',
         label='Theoretical Distribution: $(4/5)*x^{-1/5}$')

plt.legend(loc='best')
plt.xlim(1e-3, 1)
# converting both the axis to log scale
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x', fontsize=14)
plt.ylabel('p(x) (PDF)', fontsize=14)
plt.title('Problem 1-a')
#plt.savefig('Problem_1a.jpg', bbox_inches='tight', dpi=300)


# part b
# ============================================================
# defining the function to be integrated
def f(x):
    val = np.zeros(len(x))
    a = x[np.where(x <= 1)[0]]
    # getting non-zero values
    val_nz = np.exp(-a)*np.power(a, -1/5)
    val[np.where(x <= np.pi)[0]] = val_nz
    return val

# defining the weight function


def w(x):
    return (4/5)*np.power(x, -1/5)

# integrating by imporatance sampling


def imp_sampling(f, w, r, N):
    """
    Paramters
    ========================
    f: function to be integrated
    w: weigth function
    r: random variable generator
    N: Number of points
    """
    y_i = r(N)
    y = f(y_i)
    w_x = w(y_i)

    # estimator of integral
    I = np.sum(y/w_x)/N

    # calulating variance from the estimator
    F = y/w_x
    var = np.sum(F**2)/N - I**2
    return np.array([I, var])


# calulating the integral using the imporance sampling
# Number of points
N = 100000

# number of iteration
M = 1000

# array to store the value of integral
I_arr = np.zeros(M)

# array to store the value of variance
var_arr = np.zeros(M)

# running the integral M times
for i in tqdm(range(M), 'Iterations'):
    I, var = imp_sampling(f, w, r, N)
    I_arr[i] = I
    var_arr[i] = var

print("Approximated value of integral upto 3 decimal places: ",
      round(np.mean(I_arr), 3))
print("True value of integral:", 0.836581)

plt.show()
