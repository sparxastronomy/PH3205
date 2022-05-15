import matplotlib.pyplot as plt
import matplotlib.animation as animation

# setting up the problem
g = 9.81    # acceleration due to gravity (m/s^2)
x_max = 7   # The maximum x-range of ball's trajectory to plot
cor = 0.75  # The coefficient of restitution for bounces (-v_up/v_down)
dt = 0.005  # The time step for the animation

# Initial position and velocity vectors.
x0, y0 = 0, 4
vx0, vy0 = 1, 2


# Set up a new Figure, with equal aspect ratio so the ball appears round.
fig, ax = plt.subplots()
ax.set_aspect('equal')

# Objects that should be tracked.
# Array to store the ball's position at each time step.
xdata, ydata = [], []
line, = ax.plot(xdata, ydata, lw=2)     # Tracjectroy of the ball
# An MPL object to track the height of the ball.
ball = plt.Circle((x0, y0), 0.08, color='red')
ax.add_patch(ball)      # Add the ball to the axes.
# Text to display the height of the ball.
height_text = ax.text(x_max*0.5, y0*0.8, f'Height: {y0:.1f} m')


def init():
    """Initialize the animation figure."""
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y0+0.2*vy0)
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    line.set_data(xdata, ydata)
    ball.set_center((x0, y0))
    height_text.set_text(f'Height: {y0:.1f} m')
    return line, ball, height_text


def get_pos(t=0):
    """A funtion yielding the ball's position at time t."""
    x, y, vx, vy = x0, y0, vx0, vy0
    while x < x_max:
        t += dt
        x += vx * dt
        y += vy * dt
        vy -= g * dt
        if y < 0:
            # bouncing if y<0
            y = 0
            vy = -vy * cor
        yield x, y


def animate(pos):
    """For each frame, advance the animation to the new position, pos."""
    x, y = pos
    xdata.append(x)
    ydata.append(y)
    line.set_data(xdata, ydata)
    ball.set_center((x, y))
    height_text.set_text(f'Height: {y:.1f} m')
    return line, ball, height_text


interval = 1000*dt
ani = animation.FuncAnimation(fig, animate, get_pos, blit=True,
                              interval=interval, repeat=False, init_func=init)
plt.show()
