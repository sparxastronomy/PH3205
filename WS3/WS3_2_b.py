import numpy as np
import matplotlib.pyplot as plt

# definging velocity derivative for an-harmonic oscillator


def anh_deriv(x, v, l, t):
    return -x - l*x**3

# energy of an-harmonic oscillator


def anh_energy(x, v, l):
    return 0.5*(v**2) + 0.5*(x**2) + l*(x**4)/4

# leap frog method with velocity verlet


def leap_frog(dy, x, v, l, t, h):
    """
    Returns the next value of x and v after a step of size h
    using leap frog method.

    Leap frog update method is defined for following 2nd order ODE
    coupled system of ODEs is as follows
    dv/dt = -x - l*x**3
    dx/dt = v

    Parameters
    ===========================================================
    dy: derivative function for velocity
    x: current value of x
    v: current value of v
    l: parameter for anharmonic oscillator
    t: current time
    h: step size
    """
    # half step for velocity
    v1 = v + dy(x, v, l, t)*h/2

    # full step for position
    x1 = x + v1*h
    # full step for velocity
    v2 = v1 + dy(x1, v1, l, t+h)*h/2

    return np.array([x1, v2])


# initializing the values
x0 = 1
v0 = 0

# array of paramters for anharmonic oscillator
l_arr = np.array([0, 0.2, 0.5, 10, 50, 100])

fig, ax = plt.subplots(2, 3, figsize=(20, 12), dpi=100,
                       gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
m, n = 0, 0  # counter for subplots
for L in l_arr:
    # array for storing the values
    x_arr = np.array([x0])
    v_arr = np.array([v0])

    # time array
    t_start = 0
    t_end = 10
    N = 1000
    h = (t_end - t_start)/N
    t = np.linspace(t_start, t_end, N+1)

    # updating the values using leap frog
    for i in range(N):
        [x_update, v_update] = leap_frog(
            anh_deriv, x_arr[-1], v_arr[-1], L, t[i], h)
        x_arr = np.append(x_arr, x_update)
        v_arr = np.append(v_arr, v_update)

    # plotting phase-space in subplots
    ax[m, n].plot(x_arr, v_arr, lw=1, ls='-', c='b', alpha=0.5)
    # plotting the starting and ending points
    ax[m, n].scatter(x0, v0, label='Starting Postion', color='r')
    ax[m, n].scatter(x_arr[-1], v_arr[-1], label='Final Position', color='g')
    ax[m, n].legend()
    ax[m, n].set_xlabel('Position $(x)$', fontsize=14)
    ax[m, n].set_ylabel('Velocity $(\\dot{x})$', fontsize=14)
    ax[m, n].set_title('$\\lambda=%2.1f$' % L)

    # updating subpolt counters
    n += 1
    if n == 3:
        m += 1
        n = 0
plt.suptitle(
    'Phase-space of Anharmonic Oscillator (varying with $\\lambda$), $x_0=1,~V_0=0$', fontsize=20)
#plt.savefig('anh_osc_phase_space.jpg', bbox_inches='tight', dpi=200)
plt.show()
