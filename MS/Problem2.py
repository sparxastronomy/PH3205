import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["font.family"] = "serif"

# defining the function to calculate the derivatives


def derivs(q, t):
    """
    Returns the derivatives of velocities.
    d(p_x)/dt = -x - 2xy
    d(p_y)/dt = y^2 -x -x^2

    Parameters:
    ========================
    q: Current value in format [x, y]
    t: time
    """

    x, y = q
    v1 = -x - (2*x*y)  # d(p_x)/dt
    v2 = (y**2) - y - (x**2)  # d(p_y)/dt

    return np.array([v1, v2])

# leap frog method with velocity verlet


def leap_frog(q, derivs, t, h):
    """
    Returns the next value of q using the leap frog method.

    Parameters:
    ========================
    q: Current value in format [x, y, p_x, p_y]
    derivs: function to calculate the derivatives
    t: time
    h: step size
    """

    # velocity half step
    p_half = q[2:] + 0.5*h*derivs(q[0:2], t)
    # position full step
    q_full = q[0:2] + h*p_half
    # velocity full step
    p_full = p_half + 0.5*h*derivs(q_full, t+h)

    # returning the next value of q
    return np.concatenate((q_full, p_full))

# defining the energy function


def energy(q):
    """
    Returns the energy of the system. In this case it is the hamiltonina
    E = 1/2*(v_x^2 + v_y^2) + 1/2*(x^2 + y^2) + x^2y - (1/3)y^3

    Parameters:
    ========================
    q: Current value in format [x, y, v_x, v_y]
    """
    x, y, p_x, p_y = q

    T1 = 0.5*(p_x**2 + p_y**2)
    T2 = 0.5*(x**2 + y**2)
    T3 = (x**2)*y
    T4 = -(1/3)*(y**3)
    return T1 + T2 + T3 + T4


# initializing values
q0 = np.array([0.2, 0.2, 0.1, 0.1])  # initial values of [x, y, p_x, p_y]

# time array
t_start = 0
t_end = 10
N = 10000  # number of time stamps
h = (t_end - t_start)/N
t = np.linspace(t_start, t_end, N+1)

# array for storing the values
E = np.zeros(N+1)
q = np.zeros((N+1, 4))
q[0] = q0
E[0] = energy(q0)

# updating the values using leap frog method
q_current = q0
for i in tqdm(range(N), 'Iterations'):
    q_next = leap_frog(q_current, derivs, t[i], h)
    E[i+1] = energy(q_next)  # storring the energy
    # storing the values
    q[i+1] = q_next
    # updating the values
    q_current = q_next


# plotting energy variation with time

plt.figure(figsize=(10, 5), dpi=100)
plt.plot(t, E)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Energy', fontsize=14)

#plt.savefig('Problem_2.jpg', bbox_inches='tight', dpi=300)

print("Approximated value of energy upto 3 decimal places: ", round(np.mean(E), 3))
plt.show()
