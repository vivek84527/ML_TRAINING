import numpy as np

n4 = np.array([777, 555, 222, 111, 999, 666])

print("n4   =      :", n4)
print("n4.argsort():", n4.argsort())  # Sorted list of positions
# [3,2,1,5,0,4]


indxArr = n4.argsort()  # [3,2,1,5,0,4]

print("Min=", n4[indxArr[0]])

print(n4[indxArr[len(indxArr) - 1]])

print(n4[indxArr[:]])

# Trick-1: To print the array in sorted order
for i in n4.argsort():  # [3,2,1,5,0,4]
    print(n4[i], end=" ")

# Trick-2: To print the array in sorted order
print(n4[n4.argsort()])  # Original values in n4 will not change
print("n4 = ", n4)

