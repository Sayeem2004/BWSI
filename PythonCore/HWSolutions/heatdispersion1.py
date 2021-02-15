import numpy as np;
import matplotlib.pyplot as plt;
import time;
from bwsi_grader.python.heat_dispersion import grader

# Slow heat disperson function.
def evolve_heat_slow(u):
    M,N = np.shape(u);
    ret = np.copy(u);
    for x in range(1,M-1):
        for y in range(1,N-1):
            ret[x,y] = u[x-1,y] + u[x+1,y] + u[x,y-1] + u[x,y+1];
            ret[x,y] /= 4;
    return ret;
grader(evolve_heat_slow, grade_ver=1)

# Creating an example heat dispersion array.
rows = 80;
columns = 96;
u = np.zeros((rows, columns));
u[0] = np.linspace(0, 300, columns);                # top row runs 0 -> 300
u[1:,0] = np.linspace(90, 0, rows-1)                # left side goes 0 -> 90 bottom to top
u[-1,:columns//2] = np.linspace(0, 80, columns//2); # 0 (left) to 80 (middle) along the bottom
u[-1,columns//2:] = np.linspace(80, 0, columns//2); # 80 (middle) to 0 (left) along the bottom
u[:,-1] = np.linspace(300,0,rows);                  # 0 -> 300 bottom to top along the right

# Calculting time to calculate 5000 iterations.
slow = u.copy();
start = time.time();
for _ in range(5000):
    slow = evolve_heat_slow(slow);
t = round(time.time() - start, 1);
print("`evolve_heat_slow` took {} seconds to complete 5000 iterations".format(t));

# Visualizing the initial and final states.
fig, ax = plt.subplots();
ax.imshow(u, cmap='hot');
figs, axs = plt.subplots();
axs.imshow(slow,cmap="hot");
plt.show();
