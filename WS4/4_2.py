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

# defining euler integration for the heat PDE


def euler_heat(q, derivs, t, h):
    """
    Euler integrator for heat equation.

    Parameters:
    ==========================================
    q:      initial parameters
    derivs: function to compute the derivatives
    t:      Current time
    h:      step size
    """
    q1 = derivs(t, q)
    return np.array(q1)

# defining derivative function for heat equation


def heat_derivs(t, q):
    """
    Returns the next time step for q[1:N-1]
    """
    q1 = np.zeros(len(q))
    for i in range(len(q)-2):
        q1[i+1] = 0.5*(q[i] + q[i+2])
    return q1

# defining derivative function for heat equation for Rk4


def heat_derivs_rk4(t, q):
    """
    Returns the next time step for q[1:N-1]
    """
    q1 = np.zeros(len(q))
    for i in range(len(q)-2):
        q1[i+1] = (0.5/dt)*(q[i] + q[i+2]-2*q[i+1])
    return q1


# time array
t_start = 1
t_end = 100
dt_samples = [0.01, 0.05, 0.1]   # time step size

# array to store finte differnce grid for different time steps
P_euler = []
P_rk4 = []

# iterating over the time step size
for dt in dt_samples:
    t = np.arange(t_start, t_end, dt)

    # initial gaussian solution
    dx = 0.01  # spatial step size
    x0 = np.arange(-2, 2, dx)
    # defining constant
    D = (dx**2)/(2*dt)
    p0 = (1/(np.sqrt(2*np.pi)*2*D))*np.exp(-x0**2/(4*D))

    # defining finite difference grid
    # this already sets the boundary values to 0
    p = np.zeros((len(t), len(x0)))
    p[0, :] = p0/max(p0)
    p_rk4 = p.copy()

    # obtaining the solution with euler integration
    print("Euler integration for time step :", dt)
    for i in range(len(t)-1):
        p[i+1, :] = euler_heat(p[i, :], heat_derivs, t[i], dt)
    P_euler.append(p)

    # obtaining the solution with RK4
    print("Obtaining the solution with RK4 for time step :", dt)
    for i in range(len(t)-1):
        p_rk4[i+1, :] = RK4(p_rk4[i, :], heat_derivs_rk4, t[i], dt)
    P_rk4.append(p_rk4)


# plotting the results
# ploting the solution for t=1,2,20,80
T = [2, 10, 60]

# creating subplots for ploting the solution for different time steps for euler and rk4 with error
fig, axs = plt.subplots(3, 3, figsize=(24, 16), dpi=72, gridspec_kw={
                        'hspace': 0.31, 'wspace': 0.25})


# ploting euler solution for different time steps in first row
for i in range(3):
    axs[0, i].plot(x0, P_euler[i][0, :], '--', c='orange', label='t=1 s')
    # selcting time index
    for j in range(3):
        axs[0, i].plot(
            x0, P_euler[i][int((T[j]-1)/dt_samples[i]), :], label='t=%d s' % T[j])
    axs[0, i].legend(loc='best')
    axs[0, i].set_title(
        'Euler integration: $\\Delta x=0.01, \\Delta t=%2.2f$' % dt_samples[i], fontsize=14)
    axs[0, i].set_ylabel('$p(x,t)$', fontsize=10)
    axs[0, i].set_xlabel('$x$', fontsize=10)

# ploting rk4 solution for different time steps in second row
for i in range(3):
    axs[1, i].plot(x0, P_rk4[i][0, :], '--', c='orange', label='t=1 s')
    # selcting time index
    for j in range(3):
        axs[1, i].plot(x0, P_rk4[i][int((T[j]-1)/dt_samples[i]),
                       :], label='t=%d s' % T[j])
    axs[1, i].legend(loc='best')
    axs[1, i].set_title('RK4: $\\Delta x=0.01, \\Delta t=%2.2f$' %
                        dt_samples[i], fontsize=14)
    axs[1, i].set_ylabel('$p(x,t)$', fontsize=10)
    axs[1, i].set_xlabel('$x$', fontsize=10)

# plotting difference between euler and rk4 solution for different time steps in third row
for i in range(3):
    axs[2, i].plot(x0, P_rk4[i][0, :]-P_euler[i][0, :],
                   '--', c='orange', label='t=1 s')
    # selcting time index
    for j in range(3):
        axs[2, i].plot(x0, P_rk4[i][int((T[j]-1)/dt_samples[i]), :]-P_euler[i]
                       [int((T[j]-1)/dt_samples[i]), :], label='t=%d s' % T[j])
    axs[2, i].legend(loc='best')
    axs[2, i].set_title(
        'Difference (RK4 - Euler): $\\Delta x=0.01, \\Delta t=%2.2f$' % dt_samples[i], fontsize=14)
    axs[2, i].set_ylabel('$\\Delta p(x,t)$', fontsize=10)
    axs[2, i].set_xlabel('$x$', fontsize=10)

#plt.savefig("Problem2.jpg", bbox_inches='tight', dpi=300)
plt.show()
