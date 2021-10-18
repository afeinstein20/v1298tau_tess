"""
Generate a figure from the simulation results.

"""
import numpy as np
import matplotlib.pyplot as plt


# Load the simulation results
X = np.loadtxt("simulation_results.dat")

# Plot the figure
fig, ax = plt.subplots(1)
ax.imshow(X, cmap="viridis")
ax.axis("off")
fig.savefig("my_figure.pdf", bbox_inches="tight")