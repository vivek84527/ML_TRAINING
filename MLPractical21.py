import matplotlib.pyplot as plt
import numpy as np

# evenly sampled time at 200ms intervals
t = np.arange(0.0, 5.0, 0.2)
print(t)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r*-', t, t + 3, 'bs-',
         t, t + 6, 'g-', t, t + 6, 'ro',
         markersize=7)

plt.show()
