import numpy as np;
from bwsi_grader.python.heat_dispersion import grader;
import matplotlib.pyplot as plt;
from matplotlib.animation import FuncAnimation;
import time;

# Fast heat disperson function.
def evolve_heat_fast(u):
    ret = np.copy(u);
    ret[1:-1,1:-1] = u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2];
    ret[1:-1,1:-1] /= 4;
    return ret;
grader(evolve_heat_fast, grade_ver=2)

# Creating an example heat dispersion array.
rows = 80;
columns = 96;
u = np.zeros((rows, columns));
u[0] = np.linspace(0, 300, columns);                # top row runs 0 -> 300
u[1:,0] = np.linspace(90, 0, rows-1)                # left side goes 0 -> 90 bottom to top
u[-1,:columns//2] = np.linspace(0, 80, columns//2); # 0 (left) to 80 (middle) along the bottom
u[-1,columns//2:] = np.linspace(80, 0, columns//2); # 80 (middle) to 0 (left) along the bottom
u[:,-1] = np.linspace(300,0,rows);                  # 0 -> 300 bottom to top along the right

# Calculating time to run 5000 iterations
fast = u.copy();
start = time.time();
all_frames = [];
for _ in range(5000):
    all_frames.append(fast.copy());
    fast = evolve_heat_fast(fast);
t = round(time.time() - start, 1);
print("`evolve_heat_fast` took {} seconds to complete 5000 iterations".format(t));

# Visualizing the initial and final states.
fig, ax = plt.subplots();
ax.imshow(u, cmap="hot");
figs, axs = plt.subplots();
axs.imshow(fast, cmap='hot');

# Animating the process.
fig = plt.figure()
t = u.copy()
im = plt.imshow(t, animated=True, cmap='hot')
def updatefig(*args):
    im.set_array(all_frames[args[0]])
    return im,
ani = FuncAnimation(fig, updatefig, range(5000), interval=1, blit=True, repeat=True,
                    repeat_delay=1000)
plt.show()
