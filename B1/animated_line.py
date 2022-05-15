import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# setting seriefs fontstyle
plt.rcParams['font.family'] = 'serif'

# setting matplotlib style
plt.style.use('seaborn-darkgrid')

# animating 1D gaussian distribution for different sigma values


def gaussian_1D(x, mu, sigma):
    # normalized gaussian distribution using scipy
    return norm.pdf(x, mu, sigma)


# array of sigma values
sigma = np.linspace(0.1, 2, 100)

# creating figure for animation
x = np.linspace(-5, 5, 500)
y = gaussian_1D(x, 0, sigma[0])
fig = plt.figure(figsize=(7, 5), dpi=100)
ln, = plt.plot(x, y, 'r-')
plt.xlabel('x', fontsize=16)
plt.ylabel('p(x)', fontsize=16)
plt.title('Gaussian distribution for different sigma values', fontsize=16)

# creating function for animation


def animate(i):
    x = np.linspace(-5, 5, 500)
    y = gaussian_1D(x, 0, sigma[i])
    ln.set_data(x, y)
    return ln,

# defining init function


def init():
    x = np.linspace(-5, 5, 500)
    y = gaussian_1D(x, 0, sigma[0])
    ln.set_data(x, y)
    return ln,


# creating animation with blit=True and interval=50
anim = FuncAnimation(fig, animate, init_func=init,
                     frames=len(sigma), interval=100, blit=True)


# saving animation at 24fps
anim.save('gaussian_1D.gif', fps=24, dpi=100)
