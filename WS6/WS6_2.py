import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["font.family"] = "serif"

# defining the target distribution
mu = 3  # mean of proposed/target distribution
sigma = 1  # standard deviation of propsed/target destribution


def targetdist(x):
    """
    Returns the PDF of a general normal distribution.
    It is our target distribution.
    """
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-1*(x-mu)**2/(2*sigma**2))

# defining the Metropolis Hasting steping


def MHstep(x0, sigma_p, targetdist):
    """
    Returns the new value of x after a Metropolis Hasting step

    Parameters
    ========================
    x0: current value of x (Current State)
    sigma_p: standard deviation of the sampling distribution
    targetdist: target distribution

    Returns
    ========================
    x1: new value of x (Next State)
    a: 1 if the new value of x is accepted, 0 otherwise
    """

    # generating a candidate value of x from a normal distribution centered around x0 and with supplied standard deviation
    x = np.random.normal(loc=x0, scale=sigma_p)
    # Computing acceptance probability
    A = targetdist(x)/targetdist(x0)

    # Acceptance and Rejection Step
    u = np.random.uniform(0, 1)
    if u <= A:
        x1 = x  # accepting
        a = 1  # storing acceptance
    else:
        x1 = x0  # rejecting
        a = 0  # storing rejection

    return x1, a

# Running the Metropolis Hasting algorithm


# Parameters
nsamples = 1000000  # number of samples in the chain
x0 = 3  # starting value of the chain
burnin = 1  # number of burn-in steps
sigma1 = 1  # standard deviation for MH Candidate generation

# Storing the chain
X_MC = np.zeros(nsamples)
X_MC[0] = x0
acceptance = 0

# running the chain
for i in tqdm(range(nsamples), "Number of Samples: "):
    # doing burin steps
    for j in range(burnin):
        x, a = MHstep(X_MC[i-1], sigma1, targetdist)
    if a == 1:
        X_MC[i] = x
        acceptance = acceptance + 1
    else:
        X_MC[i] = X_MC[i-1]

# Computing the mean and variance of the chain
mean_MC = np.mean(X_MC)
var_MC = np.var(X_MC)

# plotting the results
plt.figure(figsize=(14, 6), dpi=100)

bins = plt.hist(X_MC, bins=100, density=True, color='#4f71b4', rwidth=0.9,
                label='MH Stationary Distribution\nIteration: %2d\nAcceptance= %2.3f \n$\\mu,\\sigma$: %2.3f, %2.3f'
                % (nsamples, acceptance/nsamples, mean_MC, np.sqrt(var_MC))
                )
# plotting the target distribution
x = np.linspace(min(bins[1]), max(bins[1]), 100)
plt.plot(x, targetdist(x), c='red',
         label='Target Distribution $\\mathcal{N}$ (%2.1f, %2.2f)' % (mu, sigma))
plt.legend(loc='best', fontsize=12)

plt.title('Metropolis Hasting Algorithm', fontsize=18)
plt.ylabel('Probability Density, $p(x)$', fontsize=14)
plt.xlabel('x', fontsize=14)

#plt.savefig('WS5_2_a.jpg', bbox_inches='tight', dpi=300)

# Running metropolis hasting for different number of samples

N = [1000, 10000, 100000, 1000000]
MC = []  # master markov chain array
A = []  # acceptance array

print("\n\n Computing distribution for different N")
for nsamples in tqdm(N):
    MC.append(np.zeros(nsamples))
    MC[-1][0] = x0
    acceptance = 0
    for i in range(nsamples):
        for j in range(burnin):
            x, a = MHstep(MC[-1][i-1], sigma1, targetdist)
        if a == 1:
            MC[-1][i] = x
            acceptance = acceptance + 1
        else:
            MC[-1][i] = MC[-1][i-1]
    mean_MC = np.mean(MC[-1])
    var_MC = np.var(MC[-1])
    A.append(acceptance/nsamples)
    print('\n\nIteration: %2d\nAcceptance= %2.3f \nmu,sigma : %2.3f, %2.3f'
          % (nsamples, acceptance/nsamples, mean_MC, np.sqrt(var_MC)))

# plotting the distributions in 4 subplots
fig, ax = plt.subplots(4, 1, figsize=(14, 20), dpi=100)

# ploting histograms and distribution in subplots
for i in range(4):
    mean_MC = np.mean(MC[i])
    var_MC = np.var(MC[i])
    bins = ax[i].hist(MC[i], bins=100, density=True, color='#4f71b4', rwidth=0.75,
                      label='MH Stationary Distribution\nIteration: %2d\nAcceptance= %2.4f \n$\\mu,\\sigma$: %2.3f, %2.3f'
                      % (N[i], A[i], mean_MC, np.sqrt(var_MC))
                      )
    # plotting the target distribution
    x = np.linspace(min(bins[1]), max(bins[1]), 100)
    ax[i].plot(x, targetdist(x), c='red',
               label='Target Distribution $\\mathcal{N}$ (%2.1f, %2.2f)' % (mu, sigma))
    ax[i].legend(loc='best', fontsize=12)
    ax[i].set_ylabel('$p(x)$', fontsize=14)
    ax[i].set_xlabel('x', fontsize=14)

ax[0].set_title("Metropolis Hasting Algorithm for different N", fontsize=20)
# plt.savefig('WS6_2_b.jpg', bbox_inches='tight', dpi=300)
plt.show()
