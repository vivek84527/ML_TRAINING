import numpy as np

n4 = np.array([10, -1, 0, 90, 300, 3, -6, 2])

print("Before : ", n4)

print(sorted(n4))  # External Sorting

print("After : ", n4)

n4.sort()  # In-place sorting  or Internal Sorting
print("After n4.sort() = ", n4)
