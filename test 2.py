import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define Lorenz system of equations
def lorenz(t, X, sigma=10, beta=8/3, rho=28):
    x, y, z = X
    dxdt = sigma*(y - x)
    dydt = x*(rho - z) - y
    dzdt = x*y - beta*z
    return [dxdt, dydt, dzdt]

# Set initial conditions and time span
X0 = [1, 1, 1]
t_span = [0, 100]
t_eval = np.linspace(0, 100, 10000)

# Solve the system of equations
sol = solve_ivp(lorenz, t_span, X0, t_eval=t_eval)

# Set up the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim((-25, 25))
ax.set_ylim((-35, 35))
ax.set_zlim((5, 55))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Create empty line object to be updated in the animation
line, = ax.plot([], [], [], lw=0.5)

# Define update function for the animation
def update(frame):
    # Update the data for the line
    line.set_data(sol.y[0:2, :frame])
    line.set_3d_properties(sol.y[2, :frame])
    # Update the title to show the current time
    ax.set_title(f"Lorenz Attractor (t = {sol.t[frame]:.2f})")
    # Return the line object
    return line,

# Create the animation
anim = FuncAnimation(fig, update, frames=len(sol.t), interval=0.00000000000000000000001, blit=True)

plt.show()