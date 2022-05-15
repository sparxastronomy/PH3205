import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from tqdm import tqdm
plt.rcParams["font.family"] = "serif"


# solving black-scholes equation
# defining noise function
def w(N, dt):
    """
    Returns array of white noise
    Parameters:
    ==========================
    N:      number of time steps
    dt:     time step

    Returns:
    ==========================
    W:      array of stochastics noise
    """

    W = np.zeros(N)  # initlaizing the array

    # getting the noise from the Wienner process
    w = np.random.normal(loc=0, scale=1, size=N-1)
    W[1:] = w*np.sqrt(dt)  # updating the array

    return W

# defing the approximation of the Black-Scholes equation


def BS(S0, T, N, mu, sigma):
    """
    Returns array of values of the Black-Scholes equation in both approximated and analytical form

    Prarameters:
    ==========================
    S0:     initial price/value of the underlying asset
    T:      time to maturity
    N:      number of time steps
    mu:     expected return of the underlying asset
    sigma:  volatility/variance of the underlying asset

    Returns:
    ==========================
    S:              Array of values of the Black-Scholes equation in approximated form
    S_analytical:   Array of values of the Black-Scholes equation in analytical form
    t           :   Array of time steps (needed for plotting)
    """

    # defining the time steps
    dt = T/N
    t = np.linspace(0, T, N)  # time array

    # Defining the stochastics part of the equation
    W = w(N, dt)  # getting the noise from wienner process
    W_T = np.cumsum(W)  # time-dependent white noise summed up i.e \eta(t)

    # initlaizing the arrays
    S = np.zeros(N)
    S[0] = S0

    # getting the approximated values of the Black-Scholes equation
    for i in range(0, N-1):
        S[i+1] = S[i] + (S[i]*mu*dt) + (S[i]*sigma*W[i+1])

    # getting the analytical values of the Black-Scholes equation
    S_analytical = S0*np.exp((mu - 0.5*sigma**2)*t + sigma*W_T)

    return S, S_analytical, t


# Part a
# ======================================================
# defining input parameters
S0 = 1
T = 10
N = 1000
mu = 1
sigma = 1.5

# getting solution of the Black-Scholes equation
print("Computing solution of Black-Scholes equation for 1 realization")
S, S_analytical, T = BS(S0, T, N, mu, sigma)

fig = plt.figure("Part A", figsize=(13, 8.5), dpi=100)
grid = plt.GridSpec(3, 1, wspace=0.4, hspace=0.3)

# slecting grids for solution and relative errors
ax1 = fig.add_subplot(grid[0:2, 0])
ax2 = fig.add_subplot(grid[2, 0])

# plotting the approximated and analytical solution
ax1.plot(T, S, lw=1.5, c='orange', label='Approximated')
ax1.plot(T, S_analytical, lw=1.5, c='blue', ls='--', label='Analytical')
ax1.legend(fontsize=12)
ax1.set_xlabel('Time of evolution, t', fontsize=14)
ax1.set_ylabel('$S_t$', fontsize=16)
ax1.grid(alpha=0.3)
ax1.set_title('Solution of the Black-Scholes equation\n$S_0$= %d, $\\mu=$%2.2f, $\\sigma=$%2.2f\n' %
              (S0, mu, sigma), fontsize=16)

# plottting the relative error between the approximated and analytical solution
ax2.plot(T, (S_analytical-S)/S_analytical, lw=1.5, c='red')
ax2.set_xlabel('Time of evolution, t', fontsize=14)
ax2.set_ylabel('Relative error, $\epsilon(S_t)$', fontsize=12)
ax2.grid(alpha=0.2)

#plt.savefig('Problem1_a.jpg', dpi=300, bbox_inches='tight')

# Part B
# ======================================================
# defining input parameters
S0 = 1
T = 10
N = 1000
mu = 1
sigma = 1.5

# defining square root fitting function


def sqrt_fit(x, a, b):
    return a*np.sqrt(x) + b


# array of time steps
Dt = np.arange(0.0001, 0.1, 0.001)
N_ar = 10/Dt


# defining a master array to RMS value for many realizations
Master_RMS = []

# number of realizations
N_real = 100

# looping over the number of realizations
print("Computing Error for %d realizations" % N_real)
for i in tqdm(range(N_real), "Realizations: "):
    # getting RMS error for varrying  time steps
    RMS = np.zeros(len(N_ar))
    for i in range(len(N_ar)):
        S, S_analytical, T_ar = BS(S0, T, int(N_ar[i]), mu, sigma)
        # getting the relative error
        Error = (S_analytical-S)/S_analytical
        RMS[i] = np.sqrt(np.mean(Error**2))
    Master_RMS.append(RMS)

# taking the average of the RMS values
RMS_avg = np.mean(Master_RMS, axis=0)

# fitting RMS error to the square root(dt) function
popt, pcov = curve_fit(sqrt_fit, Dt, RMS_avg)

fig, ax = plt.subplots(2, 1, figsize=(12, 9), dpi=100,
                       gridspec_kw={'hspace': 0.3})

# plotting the RMS error
ax[0].plot(N_ar, RMS_avg, marker='.', lw=1.5,
           c='green', label='Average RMS error')
ax[0].loglog()
ax[0].set_xlabel('log($N$)', fontsize=14)
ax[0].set_ylabel('log($\epsilon(S_t)$)', fontsize=12)
ax[0].grid(alpha=0.2)
ax[0].legend(fontsize=12, loc='best')
ax[0].set_title(
    'Average RMS error of Relative Error for different $N$', fontsize=15)


ax[1].plot(10/N_ar, RMS_avg, marker='.', lw=1.5,
           c='green', label='Average RMS error')
# plotting the fitted function
ax[1].plot(Dt, sqrt_fit(Dt, *popt), lw=1.5, c='red',
           label='Fitted function: $k\\sqrt{dt}$')
ax[1].set_xlabel('$\Delta t$', fontsize=14)
ax[1].set_ylabel('$\epsilon(S_t)$', fontsize=12)
ax[1].grid(alpha=0.2)
ax[1].legend(fontsize=12, loc='best')
ax[1].set_title(
    'Average RMS error of Relative Error for different $\\Delta t$', fontsize=15)

#plt.savefig('Problem1_b.jpg', dpi=300, bbox_inches='tight')
plt.show()
