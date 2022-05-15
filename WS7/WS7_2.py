import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import factorial
from tqdm import tqdm

plt.rcParams["font.family"] = "serif"

# solving birth-death process
# defing the approximation of the Birth-Death equation


def BD(N0, K1, K2, T, N):
    """
    Returns array of values of the Birth-death equation in approximated  form

    Prarameters:
    ==========================
    N0:     initial number of people
    K1:     initial value of parameter K1
    K2:     initial value of parameter K2
    T:      time to maturity
    N:      number of time steps


    Returns:
    ==========================
    S:              Array of values of the Black-Scholes equation in approximated form
    t           :   Array of time steps (needed for plotting)
    """

    # defining the time steps
    dt = T/N
    t = np.linspace(0, T, N)  # time array

    z = np.random.normal(0, 1, N-1)  # generating white noise

    # initlaizing the arrays
    S = np.zeros(N)
    S[0] = N0

    # getting the approximated values of the Black-Scholes equation
    for i in range(0, N-1):
        S[i+1] = S[i] + (K1*dt) - (K2*S[i]*dt) + \
            (z[i]*np.sqrt(dt*(K1 + K2*S[i])))

    # # getting the analytical values of the Black-Scholes equation
    # S_analytical = (K1/K2)*(1-np.exp(-K2*t))

    return S, t

# analytical solution of Birth-Death Equation


def BD_analytical(K1, K2, T, N):
    """
    Returns array of values of the Birth-death equation in analytical form

    Prarameters:
    ==========================
    N0:     initial number of people
    K1:     initial value of parameter K1
    K2:     initial value of parameter K2
    T:      time to maturity
    N:      number of time steps


    Returns:
    ==========================
    S:              Array of values of the Black-Scholes equation in approximated form
    t           :   Array of time steps (needed for plotting)
    """

    # defining the time steps
    dt = T/N
    t = np.linspace(0, T, N)  # time array

    # getting the analytical values of the Black-Scholes equation
    S_analytical = (K1/K2)*(1-np.exp(-K2*t))

    return S_analytical


# Part A
# =============================================================================
# getting solution for birth-death process
# defining input parameters
K1 = 5
K2 = 0.2
N = 1000
T = 50

# Number of realizations to run
N_real = 10000

# deffining arrary of initial conditions
N0 = np.random.normal(0, 1, N_real)

# defining master array to store the solutions
N_master = []
# defining arrary to store final positions
N_final = []

# looping over the number of realizations
for i in tqdm(range(N_real), "Realizations: "):
    # getting the solution
    N_ar, time_ar = BD(N0[i], K1, K2, T, N)
    N_master.append(N_ar)
    N_final.append(N_ar[-1])
N_analytical = BD_analytical(K1, K2, T, N)

# taking the average of the solutions
N_avg = np.mean(N_master, axis=0)

# plotting the approximated and analytical solution
fig = plt.figure("Part a", figsize=(12, 9.5), dpi=100)
grid = plt.GridSpec(3, 1, wspace=0.4, hspace=0.31)

# slecting grids for solution and relative errors
ax1 = fig.add_subplot(grid[0:2, 0])
ax2 = fig.add_subplot(grid[2, 0])

# plotting the approximated and analytical solution
ax1.plot(time_ar, N_avg, lw=1.5, c='orange', label='Approximated')
ax1.plot(time_ar, N_analytical, lw=1.5, c='blue', ls='--', label='Analytical')
ax1.legend(fontsize=12)
ax1.set_xlabel('Time of evolution, t', fontsize=14)
ax1.set_ylabel('$N_t$', fontsize=16)
ax1.grid(alpha=0.3)
ax1.set_title(
    'Solution of the Birth-Death equation\n$\\bar{N}_0$= %d, $K_1=$%2.2f, $K_2=$%2.2f  Number of Realizations=%d' % (0, K1, K2, N_real), fontsize=16)

# plottting the relative error between the approximated and analytical solution
ax2.plot(time_ar[1:], (N_avg[1:]-N_ar[1:])/N_analytical[1:], lw=1.5, c='red')
ax2.set_xlabel('Time of evolution, t', fontsize=14)
ax2.set_ylabel('Relative error, $\epsilon(N_t)$', fontsize=12)
ax2.grid(alpha=0.2)

#plt.savefig('Problem_2a.jpg', dpi=300, bbox_inches='tight')

# Part B
# =============================================================================
# definging function for poisson fit


def poisson_fit(k, lamb):
    dr = np.ones(len(k))
    for i in range(len(k)):
        if k[i] > 0:
            dr[i] = factorial(k[i])
    return np.exp(-lamb)*(lamb**k)/dr


# plotting the distribution of final positions of the solution
fig = plt.figure("Part b", figsize=(12, 6), dpi=100)
# bin edges using friedman diaconis rule
edges = np.histogram_bin_edges(N_final, bins='fd')
# getting midpoints of the bins
bin_centers = (edges[1:] + edges[:-1])/2

vals = plt.hist(N_final, bins=edges, density=True, color='green', edgecolor='blue', alpha=0.5,
                label='Distribution of values\n $K_1=$%2.2f, $K_2=$%2.2f' % (K1, K2))
plt.plot(bin_centers, vals[0], 'o', ms=4, c='orange')

# fitting the distribution of final positions
popt, pcov = curve_fit(poisson_fit, bin_centers, vals[0], p0=[
                       np.mean(N_final)], maxfev=1000)

# plotting the fitted distribution
x_plot = np.linspace(np.min(N_final), np.max(N_final), 200)
plt.plot(x_plot, poisson_fit(x_plot, *popt), c='red',
         label='Fitted Poisson distribution\n $\\lambda=$%2.2f' % popt[0])

plt.legend(fontsize=12, loc='best')

plt.xlabel('Steady State values', fontsize=14)
plt.ylabel('PMF,  $P(x)$', fontsize=14)

plt.title("Distribution of Steady State value for Birth-Death Equation", fontsize=16)
#plt.savefig('Problem2_b.jpg', bbox_inches='tight', dpi=300)

plt.show()
