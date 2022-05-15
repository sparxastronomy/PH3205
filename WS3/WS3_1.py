import numpy as np
import matplotlib.pyplot as plt

# intializing the paramters
omega0 = 5  # Angurlar frequecy
lambda0 = 0.2  # vicious damping parameter

# derivative for damped oscillator


def damped_deriv(x, v, t):
    return -((2*lambda0*omega0)*v + (omega0**2)*x)

# Runge-Kutta method RK4


def RK4(dy, x, v, t, h):
    """
    Returns the next value of x and v after a step of size h
    using RK4 method.

    RK4 update method is defined for following 2nd order ODE
    coupled system of ODEs is as follows
    dv/dt = -(2*lambda0*lambda0)*v + (omega0**2)*x
    dx/dt = v

    Parameters
    ===========================================================
    dy: derivative function for velocity
    x: current value of x
    v: current value of v
    t: current time
    h: step size
    """
    k1v = dy(x, v, t)*h                   # k1 for velocity
    k1x = v * h                           # k1 for position

    k2v = dy(x+k1x/2, v+k1v/2, t+(h/2))*h  # k2 for velocity
    k2x = (v + k1v/2)*h                    # k2 for position

    k3v = dy(x+k2x/2, v+k2v/2, t+(h/2))*h  # k3 for velocity
    k3x = (v + k2v/2) * h                # k3 for position

    k4v = dy(x+k3x, v+k3v, t+h)*h        # k4 for velocity
    k4x = (v + k3v)*h                     # k4 for position

    # updating the values
    v1 = v + (k1v + 2*k2v + 2*k3v + k4v)/6
    x1 = x + (k1x + 2*k2x + 2*k3x + k4x)/6
    return np.array([x1, v1])


# Array of lambda values
# Change or add values here to change the range of lambda values
# It follows that:
# lambda0<=1 : Underdamped
# lambda0>1  : Overdamped
L = np.array([0.2, 1.2])

# initializing the values
x0 = 0
v0 = 1


for l in L:
    lambda0 = l

    # array for storing the values
    x_arr = np.array([x0])
    v_arr = np.array([v0])

    # time array
    t_start = 0
    t_end = 10
    N = 1000  # number of time stamps
    h = (t_end - t_start)/N
    t = np.linspace(t_start, t_end, N+1)

    # updating the values using RK4
    for i in range(N):
        [x_update, v_update] = RK4(damped_deriv, x_arr[-1], v_arr[-1], t[i], h)
        x_arr = np.append(x_arr, x_update)
        v_arr = np.append(v_arr, v_update)

    # plotting the values
    fig, ax = plt.subplots(1, 2, figsize=(20, 5), dpi=100)
    # ploting postion and velocity in subplot 1
    ax[0].plot(t, x_arr, c='b', label='position')
    ax[0].plot(t, v_arr, c='g', label='velocity')
    ax[0].set_xlabel('time', fontsize=14)
    ax[0].set_ylabel('Position and Velocity Values', fontsize=14)
    ax[0].set_title('Position and Velocity v/s Time')
    ax[0].legend()

    # plotting the phase space in subplot 2
    ax[1].plot(x_arr, v_arr, lw=1, ls='--', c='b', alpha=0.5)
    # scatter plot of the initial and final values
    ax[1].scatter(x0, v0, label='Initial Position', color='r')
    ax[1].scatter(x_arr[-1], v_arr[-1], label='Final Position', color='g')
    ax[1].set_xlabel('Position', fontsize=14)
    ax[1].set_ylabel('Velocity', fontsize=14)
    ax[1].set_title('Phase Space Diagram')
    ax[1].legend()

    if l <= 1:
        t_string = 'Under-damped'
    elif l > 1:
        t_string = 'Over-damped'

    plt.suptitle('Damped Oscillator, '+t_string +
                 ' $\\omega=%2.2f, \\lambda=%2.2f$' % (omega0, lambda0), fontsize=16)
    #plt.savefig('Damped_oscillator_'+ t_string+'.jpg', bbox_inches='tight', dpi=200)
    plt.show()
