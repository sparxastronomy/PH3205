import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# setting seriefs fontstyle
plt.rcParams['font.family'] = 'serif'

# animating 2D gaussian distribution for different sigma values
frn = 200
fps = 45

# array of covariance values
# changing the spread along x and y axis
x_change = np.linspace(0.5, 1, 200)
y_change = np.linspace(0.5, 1, 200)

# creating covariance matrix for different sigma values
cov = np.array([[[x_change[i], 0], [0, y_change[i]]]
               for i in range(len(x_change))])

x = np.linspace(-3, 3, 500)
y = np.linspace(-3, 3, 500)
X, Y = np.meshgrid(x, y)

x1 = X.flatten()
y1 = Y.flatten()
xy = np.vstack((x1, y1)).T

# obtaining 2D gaussian distribution in the grid for different sigma values
Z_master = np.zeros((len(cov), 500, 500))
for i in range(len(cov)):
    Z_temp = multivariate_normal([0, 0], cov[i])
    z = Z_temp.pdf(xy)
    z = z.reshape(500, 500)
    Z_master[i] = z


# creating figure for animation
fig = plt.figure(figsize=(12, 9), dpi=100)
ax = fig.add_subplot(111, projection='3d')
plot = [ax.plot_surface(X, Y, Z_master[0], cmap='viridis',
                        edgecolor='none', antialiased=True)]
ax.set_zlim(0, 0.4)
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('y', fontsize=16)
ax.zaxis.set_rotate_label(False)
ax.zaxis.labelpad = 10
ax.set_zlabel('p(x,y)', rotation=90, fontsize=16)
ax.view_init(27, 40)
# title for the animation
ax.set_title('2D Gaussian Distribution', pad=-10, fontsize=20)
plt.tight_layout()

# creating function for animation


def animate3D(i, Z_master, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(
        X, Y, Z_master[i], cmap='viridis', edgecolor='none')


# creating animation with blit=True and interval=50
anim = FuncAnimation(fig, animate3D, frames=len(
    cov), fargs=(Z_master, plot), interval=frn/fps)

plt.show()

# saving animation at 30fps
anim.save('gaussian_2D.gif', fps=fps, dpi=100)
