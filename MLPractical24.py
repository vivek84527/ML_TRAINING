# Customizing matplotlib Graphics
# Changing colors and line widths

import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

# Create a figure of size 8x6 inches, 80 dots per inch
plt.figure(figsize=(8, 6), dpi=80)

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
S = np.sin(X)
C = np.cos(X)

# Plot sine with a green continuous line
#  of width 15 (pixels)
plt.plot(X, S, color="red", linewidth=5.0,
         linestyle="--", label="sine curve")

# Plot cosine with a blue continuous line of width 7 (pixels)
plt.plot(X, C, color="blue", linewidth=7.0,
         linestyle="-", label="cosine curve")

# Location of Legends
plt.legend(loc='upper left')

# Set x limits
plt.xlim(-4.0, 4.0)

# Set x ticks
plt.xticks(np.linspace(-4, 4, 9, endpoint=True))

# Set y limits
plt.ylim(-1.0, 1.0)

# Set y ticks
plt.yticks(np.linspace(-1, 1, 9, endpoint=True))
plt.grid(True)

# Show result on screen
plt.show()
