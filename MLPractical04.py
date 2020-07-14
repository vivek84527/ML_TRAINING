arr = [1, 2, 3, 4]

# list do not support Operation Broadcasting
print("arr * 5 = ", arr * 5)

for a in arr:
    print(a * 5)

import numpy as np

nparr = np.array([1, 2, 3, 4])

print(nparr)  # [1 2 3 4]
print("nparr * 5 = ", nparr * 5)
print("nparr + 5 = ", nparr + 5)

print("nparr ** 3 = ", nparr ** 3)
