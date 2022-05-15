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


# This is the plot for lambda=0
# initializing the values
x0 = 1
v0 = 0

# array of paramters for anharmonic oscillator
l_arr = np.array([0, 0.2, 0.5, 1, 5, 10])

# array for storing the values
x_arr = np.array([x0])
v_arr = np.array([v0])
E_arr = np.array([anh_energy(x0, v0, l_arr[0])])

# time array
t_start = 0
t_end = 20
N = 1000  # number of time stamps
h = (t_end - t_start)/N
t = np.linspace(t_start, t_end, N+1)

# updating the values using leap frog
for i in range(N):
    [x_update, v_update] = leap_frog(
        anh_deriv, x_arr[-1], v_arr[-1], l_arr[0], t[i], h)
    x_arr = np.append(x_arr, x_update)
    v_arr = np.append(v_arr, v_update)
    E_arr = np.append(E_arr, anh_energy(x_update, v_update, l_arr[0]))

# plotting the energy of anharmonic oscillator
fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
# plotting position and velocity in subplot 1
ax[0].plot(t, x_arr, c='b', label='Position')
ax[0].plot(t, v_arr, c='g', label='Velocity')
ax[0].set_xlabel('time',  fontsize=14)
ax[0].set_ylabel('Position and Velocity Values',  fontsize=13)
ax[0].set_title('Position and Velocity v/s Time')
ax[0].legend()

# plotting phase-plot in subplot 2
ax[1].plot(x_arr, v_arr, lw=1, ls='--', c='b', alpha=0.5)
# scatter plot of the initial and final values
ax[1].scatter(x0, v0, label='Starting Postion', color='r')
ax[1].scatter(x_arr[-1], v_arr[-1], label='final Position', color='g')
ax[1].set_xlabel('Position', fontsize=14)
ax[1].set_ylabel('Velocity', fontsize=14)
ax[1].set_title('Phase Space Plot')
ax[1].legend()

plt.suptitle('Anharmonic Oscillator, $\\lambda=0, ~x_0=1, ~v_0=0$', fontsize=16)
#plt.savefig('anh_osc_L0.jpg', bbox_inches='tight', dpi=200)
plt.show()
