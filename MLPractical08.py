import numpy as np

# We can create numpy array from range
ar = np.arange(1, 6)

print("ar = ", ar)
# [1, 2, 3, 4, 5]

ar[3] = 16

print("After updating, ar = ", ar)
# [1, 2, 3, 16, 5]
