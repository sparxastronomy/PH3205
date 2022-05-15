# importing librarires
import numpy as np
import matplotlib.pyplot as plt

# definning function for genralized RK4 integration


def RK4(q, derivs, t, h):
    """
    Generalized RK4 integrator.

    Parameters:
    ==========================================
    q:      initial parameters
    derivs: function to compute the derivatives
    t:      Current time
    h:      step size
    """

    k1 = h*derivs(t, q)
    k2 = h*derivs(t + 0.5*h, q + 0.5*k1)
    k3 = h*derivs(t + 0.5*h, q + 0.5*k2)
    k4 = h*derivs(t + h, q + k3)

    q1 = q + (k1 + 2*k2 + 2*k3 + k4)/6
    return np.array([q1])

# defining derivative function for lorenz system


def lorenz_derivs(t, q):
    """
    Lorenz system of ODEs.

    Parameters:
    ==========================================
    q:      initial parameters
    t:      Current time
    """
    # initializing the parameters
    sigma = 10
    rho = 28
    beta = 8/3

    # storing currrent values of q in x, y, z
    x, y, z = q

    # computing the derivatives
    return np.array([sigma*(y - x), x*(rho - z) - y, x*y - beta*z])


# initializing the initial paramters
q = np.array([[0, 1, 0]])

# time array
t_start = 0
t_end = 50
N = 10000  # number of time stamps
h = (t_end - t_start)/N
t = np.linspace(t_start, t_end, N+1)

# updating the parameters using RK4
for i in range(N):
    q1 = RK4(q[-1], lorenz_derivs, t[i], h)
    q = np.concatenate((q, q1), axis=0)


# plotting the results
fig = plt.figure(figsize=(15, 10), dpi=100)
grid = plt.GridSpec(3, 4, wspace=0.4, hspace=0.3)

# slecting grids for timeplots
ax3 = fig.add_subplot(grid[2, 0:2])
ax1 = fig.add_subplot(grid[0, 0:2])
ax2 = fig.add_subplot(grid[1, 0:2])
# selecting grid for x-z plot
ax4 = fig.add_subplot(grid[1:, 2:])

ax1.plot(t, q[:, 0], 'r-')
ax1.set_ylabel('x', rotation=0, fontsize=15)
ax1.set_title('Plot of $X_i(s)$ vs time', fontsize=15)
ax2.plot(t, q[:, 1], 'b-')
ax2.set_ylabel('y', rotation=0, fontsize=15)
ax3.plot(t, q[:, 2], 'g-')
ax3.set_ylabel('z  ', rotation=0, fontsize=15)
ax3.set_xlabel('time', fontsize=15)

ax4.plot(q[:, 0], q[:, 2], color='orange')
ax4.set_xlabel('x', fontsize=15)
ax4.set_ylabel('z', rotation=0, fontsize=15)
ax4.set_title("Chaotic Attactor", fontsize=17)

#plt.savefig('Problem1.jpg', bbox_inches='tight', dpi=300)
plt.show()
