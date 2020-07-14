import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5],
         'go-', label='line 1', linewidth=2)

plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25],
         'rs--', label='line 2', linewidth=4)

plt.axis([0, 6, 0, 26])
plt.legend(loc="upper right")
plt.show()
